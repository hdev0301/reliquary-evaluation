"""Reliquary CLI — mine and validate commands (v4 logging-hardened variant).

DEPLOYMENT: this file is the v4 variant of reliquary's CLI ``main.py``.
Drop it over the upstream file on the miner box (alongside engine-v4 and
math-v4):

    cp main-v4.py        /root/reliquary/reliquary/cli/main.py
    cp vllm_adapter-v4.py /root/reliquary/vllm_adapter.py
    cp engine-v4.py      /root/reliquary/reliquary/miner/engine.py
    cp math-v4.py        /root/reliquary/reliquary/environment/math.py

What v4 fixes
=============

The v3 CLI looked frozen between major milestones: the vLLM build is
~60-90 s of CUDA graph capture, ``.generate()`` is 10-60 s per attempt,
and HF model loads add another 10-15 s — and v3 emitted nothing between
"begin" / "done" markers. Worse, when stderr is block-buffered (any
non-TTY: systemd, ``nohup ... > log.txt``, docker without ``-t``), even
the markers don't appear until either the buffer fills (~4 KB) or the
process exits. The result is a CLI that looks completely silent for
minutes at a time even though everything is fine.

v4 addresses the three root causes:

1. **Buffered stdout/stderr.** The very first thing this module does is
   reconfigure ``sys.stdout`` and ``sys.stderr`` to be line-buffered
   (works on Python 3.7+). The logging subsystem additionally installs
   a ``FlushingStreamHandler`` that calls ``flush()`` after every emit —
   even if line buffering somehow gets lost, every log line still hits
   the descriptor immediately.

2. **No heartbeats during long blocking calls.** ``setup_logging`` pins
   ``vllm_adapter`` at the chosen level so v4 adapter's per-second build
   heartbeats and per-10s generation heartbeats actually reach the root
   handler. The CLI itself adds explicit "stage" log lines around every
   blocking call (model load, env load, subtensor init) so the operator
   sees motion every few seconds.

3. **No millisecond resolution.** v3's ``%H:%M:%S`` timestamps made it
   impossible to tell whether two consecutive lines are 10 ms apart or
   900 ms apart — material when diagnosing why a submission lost the
   SUPERSEDED race. v4 adds milliseconds via ``msecs`` in the formatter.

Everything else (the ``mine`` / ``validate`` command surfaces, all
flags, the single-GPU CUDA-graph-capture serialisation, the seed-from-
validator checkpoint logic) is unchanged from v3.
"""

# ---------------------------------------------------------------------------
# CRITICAL: must run BEFORE any third-party import that touches stdout/stderr.
# ---------------------------------------------------------------------------

# Tell vLLM not to reconfigure Python logging on import. vLLM 0.9+ calls
# ``logging.config.dictConfig`` from its own logger module, which replaces
# the root logger's handlers and silently kills any INFO logs we set up
# via basicConfig. Setting this env var BEFORE any transitive vLLM import
# keeps vLLM out of our logging config entirely; vLLM still emits its own
# messages (the ``(EngineCore pid=X)`` lines), they just don't clobber ours.
import os
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

# Force unbuffered output on the current process. ``PYTHONUNBUFFERED=1``
# only affects child processes when set externally; setting it here is
# documentation, not behaviour. The real fix is ``reconfigure(line_buffering=True)``
# below.
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import sys

# Line-buffer stdout/stderr so every \n triggers a flush regardless of
# whether the descriptor is a TTY. Without this, stderr is block-buffered
# under nohup / docker / systemd and an entire vLLM init + first
# .generate() call can elapse before any logs surface.
#
# ``reconfigure(line_buffering=True)`` was added in Python 3.7. We guard
# against AttributeError (older Python) and ValueError (descriptor doesn't
# support reconfigure, e.g. already wrapped). Both are non-fatal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import asyncio
import logging
import time
from pathlib import Path

import typer

from reliquary.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_HF_REPO_ID,
    ENVIRONMENT_NAME,
    VALIDATOR_HTTP_PORT,
)

# Allow ``from vllm_adapter import VLLMAdapter`` when the file lives at the
# reliquary repo root (sibling of the reliquary/ package directory).
#   __file__               = /root/reliquary/reliquary/cli/main.py
#   .parent.parent.parent  = /root/reliquary   ← repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


app = typer.Typer(name="reliquary", help="Reliquary — Verifiable Inference Subnet")


# ---------------------------------------------------------------------------
# Logging — Flushing handler + millisecond timestamps + line-buffered output
# ---------------------------------------------------------------------------

class _FlushingStreamHandler(logging.StreamHandler):
    """A ``StreamHandler`` that flushes after every emit.

    The default ``StreamHandler`` only flushes when its buffer is full
    or the process exits. On a non-TTY descriptor that's any number of
    minutes between actually seeing a log line. Calling ``flush()``
    after each record adds the cost of one syscall per log line — a
    perfectly acceptable price for getting live output.

    Wraps ``emit`` (not ``handle``) so we don't double-flush against
    filters that might cause a record to be skipped.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
            self.flush()
        except Exception:
            # ``StreamHandler.emit`` already swallows; we re-raise nothing
            # so a broken stream doesn't kill the miner loop.
            self.handleError(record)


class _MillisecondFormatter(logging.Formatter):
    """Formatter with msec timestamps so ordering between two close log
    lines is unambiguous.

    Python's default ``asctime`` truncates to second precision. When two
    submissions for the same prompt arrive 80 ms apart and only one wins
    the SUPERSEDED race, you need to see those 80 ms.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        return f"{s}.{int(record.msecs):03d}"


_PINNED_RELIQUARY_LOGGERS = (
    "reliquary",
    "reliquary.miner.engine",
    "reliquary.miner.submitter",
    "reliquary.infrastructure.chain",
    "reliquary.infrastructure.drand",
    "reliquary.cli",
    "vllm_adapter",
)
_QUIETED_VLLM_LOGGERS = ("vllm", "vllm.engine", "vllm.executor", "vllm.config")


def _install_root_handler(
    level_int: int,
    *,
    log_file: str | None = None,
) -> None:
    """Strip whatever handlers exist on the root logger and install ours.

    Idempotent: safe to call repeatedly. Used by both the initial
    ``setup_logging`` call AND by ``reseat_logging`` AFTER third-party
    imports (bittensor, vllm) to undo any handler clobber they did.

    If ``log_file`` is given, also installs a flushing FileHandler so
    a permanent record exists even when stderr is dropped by the
    surrounding shell/systemd.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    stream_handler = _FlushingStreamHandler(sys.stderr)
    stream_handler.setLevel(level_int)
    stream_handler.setFormatter(_MillisecondFormatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    root.addHandler(stream_handler)

    if log_file:
        try:
            file_handler = _FlushingStreamHandler(open(log_file, "a"))
            file_handler.setLevel(level_int)
            file_handler.setFormatter(_MillisecondFormatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            ))
            root.addHandler(file_handler)
        except OSError as e:
            print(
                f"[reliquary.cli] WARN could not open log file {log_file!r}: {e}",
                file=sys.stderr, flush=True,
            )

    root.setLevel(level_int)

    for name in _PINNED_RELIQUARY_LOGGERS:
        _lg = logging.getLogger(name)
        _lg.setLevel(level_int)
        _lg.propagate = True
        _lg.disabled = False

    for noisy in _QUIETED_VLLM_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Install our flushing root handler + pin reliquary loggers.

    Why we don't use ``logging.basicConfig``: basicConfig is a no-op
    if the root already has handlers, AND it doesn't flush on emit.
    bittensor's import installs handlers; vLLM's import sometimes
    runs ``logging.config.dictConfig``. We do the equivalent of
    ``basicConfig(force=True)`` manually using our flushing handler.

    IMPORTANT: this is fragile against post-import handler clobbering.
    Always pair this with ``reseat_logging()`` AFTER third-party imports
    that touch the logging system (bittensor, vllm). See _run() in
    ``mine()`` for the canonical pattern.
    """
    level_int = getattr(logging, level.upper(), logging.INFO)
    _install_root_handler(level_int, log_file=log_file)
    logging.getLogger("reliquary.cli").info(
        "logging initialised level=%s handler=FlushingStreamHandler "
        "stream=stderr msec_timestamps=on log_file=%s",
        level, log_file or "<none>",
    )


def reseat_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Re-install our handler AFTER bittensor / vllm imports.

    bittensor 0.9+ silently replaces the root handler on import (and
    vLLM 0.9+ may call ``logging.config.dictConfig`` during model
    construction). Both kill the FlushingStreamHandler we installed in
    setup_logging. Call this AFTER all third-party imports complete to
    re-take ownership of the root logger.

    Emits a sentinel line ``[reseat]`` so an operator can confirm the
    second handler install actually fired.
    """
    level_int = getattr(logging, level.upper(), logging.INFO)
    _install_root_handler(level_int, log_file=log_file)
    logging.getLogger("reliquary.cli").info(
        "[reseat] root handler reinstalled after third-party imports "
        "level=%s handlers=%d", level, len(logging.getLogger().handlers),
    )


def logging_probe() -> None:
    """Emit a sentinel line via every channel we care about.

    If you can see all of these in your log, every channel is healthy:

        ``LOGGING_PROBE: root``        — root logger
        ``LOGGING_PROBE: <name>``      — each pinned reliquary logger
        ``LOGGING_PROBE: direct``      — direct stderr write (bypasses
                                         the logging framework entirely)

    If the direct line shows but the logger lines don't, a third-party
    library clobbered the root handler. If even the direct line is
    missing, stderr itself is being dropped by your shell/systemd.
    """
    logging.getLogger().info("LOGGING_PROBE: root")
    for name in _PINNED_RELIQUARY_LOGGERS:
        logging.getLogger(name).info("LOGGING_PROBE: %s", name)
    print("LOGGING_PROBE: direct (bypasses logging framework)",
          file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Stage helper — explicit progress for slow blocking calls
# ---------------------------------------------------------------------------

class _Stage:
    """Bracket a blocking call with start / done log lines + elapsed time.

    Usage:
        with _Stage(logger, "loading tokenizer"):
            tokenizer = AutoTokenizer.from_pretrained(path)

    Emits:
        [stage:start] loading tokenizer
        [stage:done]  loading tokenizer elapsed_ms=842

    Failures emit ``[stage:fail]`` with the exception type and re-raise
    so the caller can decide what to do.
    """

    def __init__(self, logger: logging.Logger, label: str) -> None:
        self.logger = logger
        self.label = label
        self.t0 = 0.0

    def __enter__(self) -> "_Stage":
        self.t0 = time.monotonic()
        self.logger.info("[stage:start] %s", self.label)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_ms = (time.monotonic() - self.t0) * 1000.0
        if exc_type is None:
            self.logger.info("[stage:done]  %s elapsed_ms=%.0f", self.label, elapsed_ms)
        else:
            self.logger.error(
                "[stage:fail]  %s elapsed_ms=%.0f exc=%s: %s",
                self.label, elapsed_ms, exc_type.__name__, exc,
            )


# ---------------------------------------------------------------------------
# Model build helpers (top-level so they're picklable across asyncio.to_thread)
# ---------------------------------------------------------------------------

def _build_vllm_adapter(
    local_path: str,
    vllm_gpu: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
):
    """Construct VLLMAdapter (the ~30-60s blocking vLLM init).

    All visible progress during the vLLM build itself comes from the
    adapter's internal heartbeat thread (v4 adapter emits every second
    for the first 15s, then every 5s). We just bracket the call.
    """
    from vllm_adapter import VLLMAdapter
    logger = logging.getLogger("reliquary.cli")
    with _Stage(logger, f"vllm.LLM build gpu={vllm_gpu} max_model_len={max_model_len}"):
        return VLLMAdapter(
            model_path=local_path,
            gpu_id=vllm_gpu,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )


def _build_hf_proof_model(local_path: str, proof_gpu: int, attn_impl: str):
    """Construct the HF model used for GRAIL proof construction.

    MUST use the same attention impl the validator uses (flash_attention_2
    by default) — sketches are bit-sensitive to attention kernel variance.
    """
    import torch
    from transformers import AutoModelForCausalLM
    logger = logging.getLogger("reliquary.cli")
    with _Stage(logger, f"hf proof model build gpu={proof_gpu} attn={attn_impl}"):
        m = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to(f"cuda:{proof_gpu}").eval()
    _log_gpu_mem(proof_gpu, label="after hf proof model load")
    return m


def _build_hf_generation_model(local_path: str, gen_gpu: int, attn_impl: str):
    """Construct the HF generation model (upstream fallback path)."""
    import torch
    from transformers import AutoModelForCausalLM
    logger = logging.getLogger("reliquary.cli")
    with _Stage(logger, f"hf generation model build gpu={gen_gpu} attn={attn_impl}"):
        m = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to(f"cuda:{gen_gpu}").eval()
    _log_gpu_mem(gen_gpu, label="after hf generation model load")
    return m


def _log_gpu_mem(gpu_id: int, *, label: str = "") -> None:
    """Best-effort GPU memory snapshot to the CLI log.

    Helps an operator confirm models actually loaded on the expected
    device and how much headroom they have left.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info(gpu_id)
        used = total - free
        logging.getLogger("reliquary.cli").info(
            "gpu_mem gpu=%d %s used=%.1fGB free=%.1fGB total=%.1fGB",
            gpu_id, label,
            used / (1024 ** 3), free / (1024 ** 3), total / (1024 ** 3),
        )
    except Exception:
        pass


def _gpu_sync_and_clear(gpu_id: int) -> None:
    """Synchronize GPU and clear caches to release vLLM context locks.

    In single-GPU mode, vLLM may still hold CUDA context locks or use
    background streams after initialization. This makes the device idle
    before loading the HF proof model on the same GPU.
    """
    import torch
    logger = logging.getLogger("reliquary.cli")
    try:
        with _Stage(logger, f"gpu sync+clear gpu={gpu_id}"):
            torch.cuda.set_device(gpu_id)
            torch.cuda.synchronize(gpu_id)
            torch.cuda.empty_cache()
            time.sleep(0.5)
            torch.cuda.synchronize(gpu_id)
    except Exception as e:
        logger.warning("GPU sync/clear failed: %s", e)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def mine(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    environment: str = typer.Option(ENVIRONMENT_NAME, help="Environment name"),
    validator_url: str = typer.Option(
        "",
        help=(
            "Override the validator URL (otherwise discovered from the metagraph). "
            "Useful for local testing — e.g. http://127.0.0.1:8888"
        ),
    ),
    use_vllm: bool = typer.Option(
        False,
        "--use-vllm",
        help=(
            "Use vLLM for generation (5-10× faster than HF .generate(); "
            "closes the window_mismatch race on busy validator cycles)."
        ),
    ),
    vllm_gpu: int = typer.Option(
        0, help="GPU index for generation (default: 0). Used by both HF and vLLM paths."
    ),
    proof_gpu: int = typer.Option(
        1, help="GPU index for HF GRAIL proofs (default: 1)."
    ),
    gpu_memory_utilization: float = typer.Option(
        0.85,
        help=(
            "vLLM gpu_memory_utilization (default: 0.85). Lower to ~0.55 "
            "when vLLM and proof model share one GPU."
        ),
    ),
    max_model_len: int = typer.Option(
        8192,
        help="vLLM max_model_len (default: 8192 = MAX_NEW_TOKENS_PROTOCOL_CAP).",
    ),
    enforce_eager: bool = typer.Option(
        False,
        "--enforce-eager",
        help="Disable vLLM CUDA graphs (slower but lower memory).",
    ),
    stats_path: str = typer.Option(
        ".reliquary_miner_stats.json",
        help=(
            "Beta-posterior persistence file (default: %(default)s). "
            "Pass an empty string to disable persistence between runs."
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
    log_file: str = typer.Option(
        "",
        help=(
            "Optional path to a permanent log file. Recommended when "
            "running under systemd / nohup / docker — gives you a "
            "stderr-immune record. Default: stderr only."
        ),
    ),
):
    """Run Reliquary miner (HF by default; pass --use-vllm for vLLM backend)."""
    # Earliest possible signs of life — these go straight to stderr in
    # case anything below this point hangs before setup_logging completes.
    print(
        f"[reliquary.cli] mine starting backend={'vllm' if use_vllm else 'hf'} "
        f"network={network} netuid={netuid} log_level={log_level} "
        f"log_file={log_file or '<none>'}",
        file=sys.stderr, flush=True,
    )

    setup_logging(log_level, log_file=(log_file or None))
    logging_probe()
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    logger.info(
        "Starting Reliquary miner (network=%s, netuid=%d, env=%s, backend=%s)",
        network, netuid, environment, "vllm" if use_vllm else "hf",
    )

    async def _run():
        # IMPORTANT: bittensor (and sometimes vllm during a later
        # transitive import) reinstalls the root logger handlers, which
        # silently kills our FlushingStreamHandler. We re-take ownership
        # right after these imports complete with reseat_logging().
        print("[reliquary.cli] _run() entered — importing third-party libs",
              file=sys.stderr, flush=True)
        import bittensor as bt
        import torch
        from transformers import AutoTokenizer

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.environment import load_environment
        from reliquary.infrastructure.chain import (
            get_subtensor, get_metagraph, NETUID,
        )
        from reliquary.miner.engine import MiningEngine
        from reliquary.miner.submitter import (
            discover_validator_url, get_window_state_v2,
        )

        # Reseat: undo any handler clobber from bittensor / vllm /
        # transformers imports above. Emit a probe so the operator can
        # verify the second install fired and all channels reach stderr.
        reseat_logging(log_level, log_file=(log_file or None))
        logging_probe()

        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
        try:
            coldkey_ss58 = wallet.coldkeypub.ss58_address[:12]
        except Exception:
            coldkey_ss58 = "?"
        logger.info(
            "wallet ready: coldkey=%s hotkey=%s",
            coldkey_ss58,
            wallet.hotkey.ss58_address[:12],
        )
        with _Stage(logger, f"subtensor connect network={network}"):
            subtensor = await get_subtensor()

        # --- Resolve initial checkpoint from validator if available ---
        initial_path = checkpoint
        try:
            if validator_url:
                url = validator_url
            else:
                with _Stage(logger, "metagraph fetch (for validator discovery)"):
                    metagraph = await get_metagraph(subtensor, NETUID)
                url = discover_validator_url(metagraph)
            logger.info("resolved validator url=%s", url)

            import httpx
            from huggingface_hub import snapshot_download
            with _Stage(logger, f"GET /state from {url}"):
                async with httpx.AsyncClient(timeout=30) as client:
                    state = await get_window_state_v2(url, client=client)
            logger.info(
                "validator state: window_n=%d valid=%d checkpoint_n=%d "
                "repo=%s rev=%s",
                state.window_n, state.valid_submissions, state.checkpoint_n,
                state.checkpoint_repo_id or "-",
                (state.checkpoint_revision or "")[:12],
            )
            if state.checkpoint_repo_id and state.checkpoint_revision:
                logger.info(
                    "Validator at %s is on checkpoint %d (%s@%s). "
                    "Downloading to seed the miner model.",
                    url, state.checkpoint_n, state.checkpoint_repo_id,
                    state.checkpoint_revision[:12],
                )
                with _Stage(logger,
                            f"snapshot_download {state.checkpoint_repo_id}@{state.checkpoint_revision[:12]}"):
                    initial_path = snapshot_download(
                        repo_id=state.checkpoint_repo_id,
                        revision=state.checkpoint_revision,
                    )
                logger.info("Using initial checkpoint path: %s", initial_path)
            else:
                logger.info(
                    "Validator has no published checkpoint yet — using --checkpoint=%s",
                    checkpoint,
                )
        except Exception as e:
            logger.warning(
                "Could not fetch validator checkpoint (%s); falling back to "
                "--checkpoint=%s",
                e, checkpoint,
            )

        # --- Load tokenizer + models from resolved path ---
        with _Stage(logger, f"tokenizer load from {initial_path}"):
            tokenizer = AutoTokenizer.from_pretrained(initial_path)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise RuntimeError("no CUDA devices visible — miner needs at least 1 GPU")
        effective_proof_gpu = proof_gpu if gpu_count >= 2 else 0
        if effective_proof_gpu != proof_gpu:
            logger.info(
                "Only %d GPU visible; forcing --proof-gpu=0 (was %d). "
                "Generation and proofs will share cuda:0.",
                gpu_count, proof_gpu,
            )
        single_gpu = (vllm_gpu == effective_proof_gpu)
        logger.info(
            "device topology: gpu_count=%d vllm_gpu=%d proof_gpu=%d single_gpu=%s",
            gpu_count, vllm_gpu, effective_proof_gpu, single_gpu,
        )

        if use_vllm:
            try:
                from vllm_adapter import VLLMAdapter  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "--use-vllm requires vllm_adapter.py at the reliquary "
                    f"repo root. Searched: {_REPO_ROOT}. Original error: {e}"
                )

            if single_gpu:
                # Serialize: while vLLM captures CUDA graphs, any other tensor
                # op on the same device fails with "operation not permitted
                # when stream is capturing". Serial loading costs ~10-15s vs
                # parallel; well worth the reliability.
                logger.info(
                    "single-GPU mode (vllm_gpu=proof_gpu=%d) — serializing "
                    "model loads to avoid CUDA graph capture conflict",
                    vllm_gpu,
                )
                vllm_model = await asyncio.to_thread(
                    _build_vllm_adapter,
                    initial_path, vllm_gpu, max_model_len,
                    gpu_memory_utilization, enforce_eager,
                )
                logger.info("vllm adapter ready — synchronizing GPU before HF proof model load")
                await asyncio.to_thread(_gpu_sync_and_clear, vllm_gpu)
                logger.info("GPU synchronized — starting HF proof model load")
                hf_model = await asyncio.to_thread(
                    _build_hf_proof_model,
                    initial_path, effective_proof_gpu, ATTN_IMPLEMENTATION,
                )
                logger.info("HF proof model ready on cuda:%d", effective_proof_gpu)
            else:
                logger.info(
                    "2-GPU mode: vllm on cuda:%d, proofs on cuda:%d — "
                    "loading concurrently",
                    vllm_gpu, effective_proof_gpu,
                )
                vllm_task = asyncio.to_thread(
                    _build_vllm_adapter,
                    initial_path, vllm_gpu, max_model_len,
                    gpu_memory_utilization, enforce_eager,
                )
                hf_task = asyncio.to_thread(
                    _build_hf_proof_model,
                    initial_path, effective_proof_gpu, ATTN_IMPLEMENTATION,
                )
                vllm_model, hf_model = await asyncio.gather(vllm_task, hf_task)
        else:
            logger.info(
                "HF backend: generation on cuda:%d, proofs on cuda:%d",
                vllm_gpu, effective_proof_gpu,
            )
            vllm_model = _build_hf_generation_model(
                initial_path, vllm_gpu, ATTN_IMPLEMENTATION,
            )
            hf_model = _build_hf_proof_model(
                initial_path, effective_proof_gpu, ATTN_IMPLEMENTATION,
            )

        with _Stage(logger, f"environment load: {environment}"):
            env = load_environment(environment)
        logger.info("environment loaded: %s (n_prompts=%d)", environment, len(env))

        logger.info(
            "constructing MiningEngine (backend=%s, vllm_gpu=%d, proof_gpu=%d, "
            "stats_path=%s)",
            "vllm" if use_vllm else "hf",
            vllm_gpu, effective_proof_gpu, stats_path or "<disabled>",
        )

        class _PatchedMiningEngine(MiningEngine):
            def _load_checkpoint(self, *args, **kwargs):
                if single_gpu:
                    logger.info("Synchronizing GPU before checkpoint reload (single-GPU mode)")
                    _gpu_sync_and_clear(vllm_gpu)
                return super()._load_checkpoint(*args, **kwargs)

        engine = _PatchedMiningEngine(
            vllm_model,
            hf_model,
            tokenizer,
            wallet,
            env,
            vllm_gpu=vllm_gpu,
            proof_gpu=effective_proof_gpu,
            validator_url_override=validator_url or None,
            stats_path=(stats_path or None),
        )
        logger.info("MiningEngine constructed (patched for GPU sync on reload)")

        if initial_path != checkpoint:
            engine._loaded_checkpoint_path = initial_path
            logger.info(
                "seeded engine._loaded_checkpoint_path=%s "
                "(skips redundant first checkpoint reload)",
                initial_path,
            )

        logger.info(
            "Miner ready (backend=%s). Entering main loop.",
            "vllm" if use_vllm else "hf",
        )
        try:
            await engine.mine_window(subtensor, 0, use_drand=use_drand)
        except KeyboardInterrupt:
            logger.info("Miner interrupted by user")
        except Exception as e:
            logger.error("Mining loop crashed: %s", e, exc_info=True)
            raise

    asyncio.run(_run())


@app.command()
def validate(
    train: bool = typer.Option(
        True,
        "--train/--no-train",
        help=(
            "Run full trainer mode (default). "
            "Pass --no-train for weight-only mode: reads R2 archives, "
            "computes EMA, submits weights. No GPU, no HF, no HTTP server."
        ),
    ),
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(
        DEFAULT_BASE_MODEL,
        help="HF repo id or local path of the model to load (trainer mode only)",
    ),
    environment: str = typer.Option(
        ENVIRONMENT_NAME, help="Environment name (trainer mode only)",
    ),
    http_host: str = typer.Option(
        "0.0.0.0", help="HTTP bind address (trainer mode only)",
    ),
    http_port: int = typer.Option(
        VALIDATOR_HTTP_PORT, help="HTTP listen port (trainer mode only)",
    ),
    external_ip: str = typer.Option(
        "",
        help=(
            "Public IP this validator is reachable at. Published on-chain via "
            "serve_axon so miners can discover it through the metagraph. "
            "Leave empty to skip publishing (miners then need --validator-url). "
            "Trainer mode only."
        ),
    ),
    external_port: int = typer.Option(
        0,
        help="Public port to advertise on-chain; defaults to --http-port when 0. Trainer mode only.",
    ),
    hf_repo_id: str = typer.Option(
        DEFAULT_HF_REPO_ID,
        help="HuggingFace repo ID to publish checkpoints to (must be writable with HF_TOKEN). Trainer mode only.",
    ),
    resume_from: str = typer.Option(
        os.getenv("RELIQUARY_RESUME_FROM", ""),
        help=(
            "Resume trainer from a checkpoint instead of the base model. "
            "Accepts 'sha:<40-hex>' (HF commit on --hf-repo-id) or "
            "'path:<dir>' (local ckpt_<N> directory). Trainer mode only."
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run Reliquary validator (trainer mode by default; --no-train for weight-only)."""
    print(
        f"[reliquary.cli] validate starting train={train} "
        f"network={network} netuid={netuid} log_level={log_level}",
        file=sys.stderr, flush=True,
    )
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    if train:
        logger.info(
            "Starting Reliquary validator [trainer] (network=%s, netuid=%d, env=%s, http=%s:%d)",
            network, netuid, environment, http_host, http_port,
        )
    else:
        logger.info(
            "Starting Reliquary validator [weight-only] (network=%s, netuid=%d)",
            network, netuid,
        )

    async def _run():
        import bittensor as bt

        from reliquary.infrastructure.chain import get_subtensor

        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
        with _Stage(logger, f"subtensor connect network={network}"):
            subtensor = await get_subtensor()

        if train:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from reliquary.constants import ATTN_IMPLEMENTATION
            from reliquary.environment import load_environment
            from reliquary.validator.service import ValidationService

            with _Stage(logger, f"tokenizer load from {checkpoint}"):
                tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

            with _Stage(logger, f"trainer model load: {checkpoint}"):
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=ATTN_IMPLEMENTATION,
                ).to("cuda:0").eval()
            _log_gpu_mem(0, label="after trainer model load")

            with _Stage(logger, f"environment load: {environment}"):
                env = load_environment(environment)

            service = ValidationService(
                wallet,
                model,
                tokenizer,
                env,
                netuid,
                use_drand=use_drand,
                http_host=http_host,
                http_port=http_port,
                external_ip=external_ip or None,
                external_port=(external_port or http_port) if external_ip else None,
                hf_repo_id=hf_repo_id,
                resume_from=resume_from or None,
            )
            from reliquary.validator.weight_only import WeightOnlyValidator
            weights_worker = WeightOnlyValidator(wallet=wallet, netuid=netuid)
            await asyncio.gather(
                service.run(subtensor),
                weights_worker.run(subtensor),
            )
        else:
            from reliquary.validator.weight_only import WeightOnlyValidator

            validator = WeightOnlyValidator(wallet=wallet, netuid=netuid)
            await validator.run(subtensor)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
