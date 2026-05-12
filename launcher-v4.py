"""Standalone launcher for the vLLM-backed Reliquary miner (v4 logging-hardened).

DEPLOYMENT (full v4 swap on the miner box):

    /root/reliquary/
    ├── launcher-v4.py              ← THIS FILE  (run with ``python launcher.py``)
    ├── vllm_adapter-v4.py          ← cp to vllm_adapter.py
    ├── reliquary/
    │   ├── miner/engine.py         ← cp engine-v4.py over this
    │   ├── miner/submitter.py      ← cp submitter-v4.py over this (v4.2)
    │   └── environment/math.py     ← cp math-v4.py over this
    └── ...

v4.2 adds multi-validator broadcast and fail-fast HTTP via the
submitter-v4 overlay; see ``--max-validators`` and ``--http-timeout``.

Run command (2-GPU H200 box, vLLM on cuda:0, proofs on cuda:1):

    cd /root/reliquary
    source .venv/bin/activate
    python launcher.py \
        --network finney --netuid 81 \
        --wallet-name <name> --hotkey <hk> \
        --checkpoint Qwen/Qwen3-4B-Instruct-2507 \
        --validator-url http://86.38.238.30:8080 \
        --vllm-gpu 0 --proof-gpu 1 \
        --gpu-memory-utilization 0.85 \
        --log-level INFO

What changed vs v3 launcher
===========================

Same problem the CLI had: stderr is block-buffered under non-TTY (any
``python launcher.py 2>&1 | tee log.txt`` setup, systemd, docker, etc.)
and v3 emitted nothing between major stages. v4:

1. **Line-buffered stdout/stderr at file-import time.** First thing this
   module does — before any other import — is reconfigure both streams.
2. **FlushingStreamHandler.** Replaces the default handler so each log
   record reaches the descriptor immediately, even if line buffering
   somehow gets lost.
3. **Millisecond timestamps.** SUPERSEDED races land within ~100 ms of
   each other; second-resolution timestamps make ordering ambiguous.
4. **Stage brackets around every blocking call** (subtensor connect,
   snapshot_download, tokenizer load, model build). Each emits a
   start/done pair with elapsed_ms.
5. **Earliest-possible diagnostic print** to stderr before logging is
   set up, so a hang during the first 100 ms of imports is still visible.

Engine wiring is unchanged: the v4 engine + v4 adapter handle the
heartbeats during the long-running blocking calls (LLM construct, model
reload, generate). This file just gets the logging-system *itself* out
of the way.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CRITICAL: must run BEFORE any third-party import that touches stdout/stderr.
# ---------------------------------------------------------------------------

import os

# Tell vLLM not to reconfigure Python logging on import. vLLM 0.9+ calls
# ``logging.config.dictConfig`` from its own logger module, which replaces
# the root logger's handlers and silently kills any INFO logs we set up.
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import sys

# Line-buffer stdout/stderr so every \n triggers a flush, regardless of
# whether the descriptor is a TTY. Without this, under nohup/systemd/docker
# stderr accumulates an entire vLLM init + first generate before any logs
# surface.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# Earliest possible sign of life — visible even if every import below
# hangs.
print(
    "[reliquary-vllm-launcher] module import begin "
    "(stderr line-buffered, VLLM_CONFIGURE_LOGGING=0)",
    file=sys.stderr, flush=True,
)

import argparse
import asyncio
import logging
import signal
import time
from pathlib import Path

logger = logging.getLogger("reliquary-vllm-launcher")


# ---------------------------------------------------------------------------
# Logging — FlushingStreamHandler with msec timestamps (mirrors main-v4)
# ---------------------------------------------------------------------------

class _FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes on every emit. See main-v4 for rationale."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
            self.flush()
        except Exception:
            self.handleError(record)


class _MillisecondFormatter(logging.Formatter):
    """Adds msec precision to ``asctime`` — needed for SUPERSEDED race diagnosis."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        return f"{s}.{int(record.msecs):03d}"


_PINNED_LOGGERS = (
    "reliquary",
    "reliquary.miner.engine",
    "reliquary.miner.submitter",
    "reliquary.infrastructure.chain",
    "reliquary.infrastructure.drand",
    "reliquary-vllm-launcher",
    "vllm_adapter",
)
_QUIETED_VLLM = ("vllm", "vllm.engine", "vllm.executor", "vllm.config")


def _install_root_handler(level_int: int, log_file: str | None = None) -> None:
    """Strip + install our flushing root handler. Idempotent."""
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
                f"[launcher] WARN could not open log file {log_file!r}: {e}",
                file=sys.stderr, flush=True,
            )

    root.setLevel(level_int)
    for name in _PINNED_LOGGERS:
        _lg = logging.getLogger(name)
        _lg.setLevel(level_int)
        _lg.propagate = True
        _lg.disabled = False
    for noisy in _QUIETED_VLLM:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _setup_logging(level_name: str, log_file: str | None = None) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    _install_root_handler(level, log_file=log_file)
    logger.info(
        "logging initialised level=%s handler=FlushingStreamHandler "
        "stream=stderr msec_timestamps=on log_file=%s",
        level_name, log_file or "<none>",
    )


def _reseat_logging(level_name: str, log_file: str | None = None) -> None:
    """Re-install our handler AFTER bittensor / vllm imports clobber it."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    _install_root_handler(level, log_file=log_file)
    logger.info(
        "[reseat] root handler reinstalled after third-party imports "
        "level=%s handlers=%d", level_name, len(logging.getLogger().handlers),
    )


def _logging_probe() -> None:
    """Emit known sentinel via every channel. Grep for 'LOGGING_PROBE' to verify."""
    logging.getLogger().info("LOGGING_PROBE: root")
    for name in _PINNED_LOGGERS:
        logging.getLogger(name).info("LOGGING_PROBE: %s", name)
    print("LOGGING_PROBE: direct (bypasses logging framework)",
          file=sys.stderr, flush=True)


class _Stage:
    """Bracket a blocking call with start / done log lines + elapsed time.

    See main-v4 for the same helper.
    """

    def __init__(self, lg: logging.Logger, label: str) -> None:
        self.logger = lg
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


def _log_gpu_mem(gpu_id: int, *, label: str = "") -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info(gpu_id)
        used = total - free
        logger.info(
            "gpu_mem gpu=%d %s used=%.1fGB free=%.1fGB total=%.1fGB",
            gpu_id, label,
            used / (1024 ** 3), free / (1024 ** 3), total / (1024 ** 3),
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reliquary miner (vLLM backend, v4 logging)")
    p.add_argument("--network", default="finney",
                   help="Bittensor network (default: finney)")
    p.add_argument("--netuid", type=int, default=81,
                   help="Subnet UID (default: 81)")
    p.add_argument("--wallet-name", required=True)
    p.add_argument("--hotkey", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="HF repo id or local path to seed checkpoint")
    p.add_argument("--validator-url", required=False, default="",
                   help="http://ip:port of the validator HTTP server. "
                        "Comma-separated to broadcast to multiple validators "
                        "in parallel (v4.2). Empty = auto-discover from "
                        "metagraph (recommended).")
    p.add_argument("--max-validators", type=int, default=5,
                   help="Maximum number of permitted validators to broadcast "
                        "each /submit to (v4.2 multi-validator). Set to 1 to "
                        "restore v3 single-validator behaviour. Ignored if "
                        "--validator-url is set explicitly.")
    p.add_argument("--http-timeout", type=float, default=30.0,
                   help="Per-request HTTP timeout in seconds (v4.2). Fails "
                        "fast on slow validators so the OPEN window isn't "
                        "wedged by a single doomed POST.")
    p.add_argument("--vllm-gpu", type=int, default=0,
                   help="GPU index for vLLM generation (default: 0)")
    p.add_argument("--proof-gpu", type=int, default=1,
                   help="GPU index for HF GRAIL proofs (default: 1)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                   help="vLLM gpu_memory_utilization (default: 0.85). Lower "
                        "to ~0.55 if vLLM and proof model share one GPU.")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="vLLM max_model_len (default: 8192 = MAX_NEW_TOKENS_PROTOCOL_CAP)")
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable CUDA graphs (slower but lower memory).")
    p.add_argument("--no-drand", action="store_true",
                   help="Skip drand beacon. MUST match validator's --use-drand.")
    p.add_argument("--stats-path", default=".reliquary_miner_stats.json",
                   help="Posterior persistence file (default: %(default)s). "
                        "Pass empty string to disable.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", default="",
                   help="Optional path to a permanent log file. Recommended "
                        "when running under systemd/nohup/docker for a "
                        "stderr-immune record. Default: stderr only.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bootstrap helpers (v3 logic + v4 stage brackets)
# ---------------------------------------------------------------------------

def _resolve_checkpoint(checkpoint: str) -> str:
    p = Path(checkpoint)
    if p.exists() and p.is_dir():
        logger.info("using local checkpoint at %s", p)
        return str(p)

    with _Stage(logger, f"snapshot_download {checkpoint}"):
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(
            repo_id=checkpoint,
            allow_patterns=["model.safetensors", "config.json",
                            "*.json", "tokenizer*", "*.txt"],
        )
    logger.info("checkpoint cached at %s", local_path)
    return local_path


def _build_vllm_adapter(local_path: str, args: argparse.Namespace):
    """Construct VLLMAdapter. v4 adapter prints its own progress heartbeats."""
    from vllm_adapter import VLLMAdapter
    with _Stage(logger,
                f"vllm.LLM build gpu=cuda:{args.vllm_gpu} "
                f"max_model_len={args.max_model_len} "
                f"mem_util={args.gpu_memory_utilization:.2f}"):
        return VLLMAdapter(
            model_path=local_path,
            gpu_id=args.vllm_gpu,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
        )


def _build_hf_proof_model(local_path: str, proof_gpu: int):
    """Build the HF model used for GRAIL proof construction.

    MUST use the same attention impl the validator uses
    (flash_attention_2 by default) — sketches are bit-sensitive to
    attention kernel variance.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from reliquary.constants import ATTN_IMPLEMENTATION
    with _Stage(logger,
                f"hf proof model build gpu=cuda:{proof_gpu} "
                f"attn={ATTN_IMPLEMENTATION}"):
        m = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to(f"cuda:{proof_gpu}").eval()
    _log_gpu_mem(proof_gpu, label="after hf proof model load")
    return m


def _build_tokenizer(local_path: str):
    from transformers import AutoTokenizer
    with _Stage(logger, f"tokenizer load from {local_path}"):
        tok = AutoTokenizer.from_pretrained(local_path)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
    return tok


def _build_env():
    """Default to the math env, mirroring constants.ENVIRONMENT_NAME."""
    with _Stage(logger, "environment build (math)"):
        from reliquary.environment.math import MATHEnvironment
        env = MATHEnvironment()
    logger.info("environment ready: n_prompts=%d", len(env))
    return env


def _build_wallet(name: str, hotkey: str):
    import bittensor as bt
    with _Stage(logger, f"wallet open name={name} hotkey={hotkey}"):
        w = bt.Wallet(name=name, hotkey=hotkey)
    return w


async def _build_subtensor(network: str):
    import bittensor as bt
    with _Stage(logger, f"subtensor connect network={network}"):
        subtensor = bt.AsyncSubtensor(network=network)
        await asyncio.wait_for(subtensor.initialize(), timeout=120.0)
    return subtensor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _amain(args: argparse.Namespace) -> int:
    from reliquary.miner.engine import MiningEngine

    # MiningEngine import transitively pulls bittensor.config / vllm_adapter
    # logger modules. Re-take ownership of the root handler now that those
    # imports are complete.
    _reseat_logging(args.log_level, log_file=(args.log_file or None))
    _logging_probe()

    local_path = await asyncio.to_thread(_resolve_checkpoint, args.checkpoint)
    tokenizer = await asyncio.to_thread(_build_tokenizer, local_path)
    env = await asyncio.to_thread(_build_env)
    wallet = _build_wallet(args.wallet_name, args.hotkey)
    # Second reseat — _build_wallet imports bittensor at runtime, which is
    # the most aggressive logger-config rewriter in the stack.
    _reseat_logging(args.log_level, log_file=(args.log_file or None))

    # Build models. On a 2-GPU box (vllm_gpu != proof_gpu) we run them
    # concurrently so the slow vLLM init overlaps with the HF proof
    # model load. On a single-GPU box we MUST serialize to avoid the
    # CUDA-graph-capture-while-other-tensor-op race.
    if args.vllm_gpu == args.proof_gpu:
        logger.info(
            "single-GPU mode (vllm_gpu=proof_gpu=%d) — serializing model loads",
            args.vllm_gpu,
        )
        vllm_model = await asyncio.to_thread(_build_vllm_adapter, local_path, args)
        hf_model = await asyncio.to_thread(
            _build_hf_proof_model, local_path, args.proof_gpu,
        )
    else:
        logger.info(
            "2-GPU mode: vllm on cuda:%d, proofs on cuda:%d — loading concurrently",
            args.vllm_gpu, args.proof_gpu,
        )
        vllm_task = asyncio.to_thread(_build_vllm_adapter, local_path, args)
        hf_task = asyncio.to_thread(_build_hf_proof_model, local_path, args.proof_gpu)
        vllm_model, hf_model = await asyncio.gather(vllm_task, hf_task)

    stats_path = args.stats_path or None
    # Parse comma-separated --validator-url for multi-validator broadcast
    # (v4.2). Trims whitespace and drops empty entries.
    validator_urls_list: list[str] | None = None
    if args.validator_url:
        validator_urls_list = [
            u.strip() for u in args.validator_url.split(",") if u.strip()
        ]
        logger.info(
            "validator URL override: %d explicit URL(s) — %s",
            len(validator_urls_list),
            ",".join(validator_urls_list),
        )
    else:
        logger.info(
            "validator discovery: auto (up to max_validators=%d "
            "from metagraph, http_timeout=%.1fs)",
            args.max_validators, args.http_timeout,
        )
    engine = MiningEngine(
        vllm_model=vllm_model,
        hf_model=hf_model,
        tokenizer=tokenizer,
        wallet=wallet,
        env=env,
        vllm_gpu=args.vllm_gpu,
        proof_gpu=args.proof_gpu,
        max_new_tokens=args.max_model_len,
        validator_url_override=None,
        validator_urls_override=validator_urls_list,
        max_validators=args.max_validators,
        http_timeout_s=args.http_timeout,
        stats_path=stats_path,
    )

    subtensor = await _build_subtensor(args.network)
    _val_str = args.validator_url or f"<auto-discover up to {args.max_validators}>"
    logger.info(
        "miner ready: hotkey=%s validator=%s http_timeout=%.1fs",
        wallet.hotkey.ss58_address, _val_str, args.http_timeout,
    )

    try:
        await engine.mine_window(
            subtensor,
            window_start=0,
            use_drand=not args.no_drand,
        )
    except asyncio.CancelledError:
        logger.info("mine_window cancelled — shutting down cleanly")
    except KeyboardInterrupt:
        logger.info("interrupted by user")
    finally:
        try:
            await subtensor.close()
        except Exception:
            pass
    return 0


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level, log_file=(args.log_file or None))
    _logging_probe()

    # Diagnostic prints so an operator can immediately tell, from the
    # first lines of the log, which engine.py is actually loaded and
    # whether v4 (picker/probe/submit/cohort) logs will fire.
    try:
        import reliquary.miner.engine as _eng
        import inspect
        src = inspect.getsource(_eng)
        logger.info(
            "engine.py loaded from: %s "
            "(v4=%s, has_cohort=%s, has_batched_proof=%s, has_superseded_blacklist=%s)",
            _eng.__file__,
            "_PromptStats" in src and "_cohort_counts" in src,
            "_cohort_prior" in src,
            "_build_rollout_submissions_batched" in src,
            "_superseded_in_window" in src,
        )
    except Exception:
        logger.exception("could not introspect reliquary.miner.engine")

    try:
        import vllm_adapter as _vad
        import inspect
        src = inspect.getsource(_vad)
        logger.info(
            "vllm_adapter loaded from: %s "
            "(v4=%s, has_build_heartbeat=%s, has_stderr_print=%s)",
            _vad.__file__,
            "_Heartbeat" in src,
            "label='LLM(...) construct'" in src,
            "_stderr_print" in src,
        )
    except Exception:
        logger.exception("could not introspect vllm_adapter")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    main_task = loop.create_task(_amain(args))

    def _shutdown(*_):
        logger.info("signal received → cancelling main task")
        main_task.cancel()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _shutdown)
            except NotImplementedError:
                # add_signal_handler is unavailable on Windows; KeyboardInterrupt
                # path in _amain still catches Ctrl+C.
                pass
        loop.run_until_complete(main_task)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
