"""Reliquary CLI — mine and validate commands (v3 vLLM-aware variant).

DEPLOYMENT: this file is the v3 variant of reliquary's CLI ``main.py`` with
added vLLM support for the ``mine`` command. Copy it over the upstream file
on the miner box:

    cp main-v3.py /root/reliquary/reliquary/cli/main.py

After the swap, ``reliquary mine`` gains the same vLLM backend that
``launcher.py`` uses — but exposed through the canonical CLI entry point
so operators don't need a separate launcher script. Pass ``--use-vllm`` to
opt in; without the flag the command behaves identically to the upstream
HF-only path (so existing miner setups don't change behavior on upgrade).

New ``mine`` flags (all no-ops when ``--use-vllm`` is not passed):

  --use-vllm                       Switch generation backend from HF to vLLM
  --vllm-gpu INT                   GPU index for vLLM (default 0)
  --proof-gpu INT                  GPU index for HF GRAIL proofs (default 1)
  --gpu-memory-utilization FLOAT   vLLM gpu_memory_utilization (default 0.85;
                                   drop to ~0.55 when sharing one GPU)
  --max-model-len INT              vLLM max_model_len (default 8192)
  --enforce-eager                  Disable vLLM CUDA graphs (slower, less mem)
  --stats-path STR                 Beta-posterior persistence file
                                   (empty string disables persistence)

The vLLM path requires ``vllm_adapter.py`` to be importable. Since this
file lives at ``reliquary/cli/main.py`` after deployment, we add the
reliquary repo root (``../../../`` relative to __file__, i.e.
``/root/reliquary``) to sys.path so the sibling ``vllm_adapter.py`` at
the repo root is reachable. This works under both ``pip install -e .``
(editable, which is how launcher.py expects the layout) and a normal
``pip install .`` — for the latter, ensure vllm_adapter.py is in the
PYTHONPATH some other way (sym-link into site-packages, or put it next to
the venv).

The ``validate`` command is unchanged from upstream.

Single-GPU note: when ``--vllm-gpu == --proof-gpu``, the two model loads
are serialized to avoid the CUDA-graph-capture-while-other-tensor-op race
(observed crash: "operation not permitted when stream is capturing"
around 89% through vLLM's graph capture). Costs ~10s vs parallel; worth
the reliability.

Usage with vLLM:

    reliquary mine \\
        --use-vllm \\
        --network finney --netuid 81 \\
        --wallet-name <name> --hotkey <hk> \\
        --checkpoint Qwen/Qwen3-4B-Instruct-2507 \\
        --validator-url http://<vip>:8080 \\
        --vllm-gpu 0 --proof-gpu 0 \\
        --gpu-memory-utilization 0.55 \\
        --log-level INFO

Usage without vLLM (unchanged from upstream):

    reliquary mine \\
        --network finney --netuid 81 \\
        --wallet-name <name> --hotkey <hk> \\
        --checkpoint <path-or-repo> \\
        --validator-url http://<vip>:8080
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Tell vLLM not to reconfigure Python logging on import. vLLM 0.9+ calls
# ``logging.config.dictConfig`` from its own logger module, which replaces
# the root logger's handlers and silently kills any INFO logs we set up via
# basicConfig — same failure mode the comment in setup_logging below
# describes for bittensor, just from vLLM instead. Setting this env var
# BEFORE any transitive vLLM import keeps vLLM out of our logging config
# entirely; vLLM still emits its own messages (the ``(EngineCore pid=X)``
# lines), they just don't clobber ours.
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

import typer

from reliquary.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_HF_REPO_ID,
    ENVIRONMENT_NAME,
    VALIDATOR_HTTP_PORT,
)

# Allow ``from vllm_adapter import VLLMAdapter`` when the file lives at the
# reliquary repo root (sibling of the reliquary/ package directory). Only
# needed in editable installs — for non-editable installs, vllm_adapter.py
# must be on PYTHONPATH some other way.
#   __file__               = /root/reliquary/reliquary/cli/main.py
#   .parent                = /root/reliquary/reliquary/cli
#   .parent.parent         = /root/reliquary/reliquary
#   .parent.parent.parent  = /root/reliquary   ← repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


app = typer.Typer(name="reliquary", help="Reliquary — Verifiable Inference Subnet")


def setup_logging(level: str = "INFO"):
    """Configure root logger and pin per-package levels.

    ``force=True`` is load-bearing: bittensor calls ``logging.basicConfig()``
    during its own import (visible as "Enabling default logging" at miner
    start), which without ``force=True`` makes our basicConfig a silent
    no-op — root logger keeps whatever bittensor set, and our INFO logs
    vanish even though they "should" propagate via the root.
    """
    pinned_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=pinned_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    # Belt-and-suspenders: explicitly pin reliquary + vllm_adapter loggers
    # to the chosen level. Even if some part of reliquary or vLLM later
    # calls ``setLevel(WARNING)`` on a sibling logger (which has been
    # observed for bittensor), these guarantees stay in effect.
    for name in (
        "reliquary",
        "reliquary.miner.engine",
        "reliquary.cli",
        "vllm_adapter",
    ):
        _lg = logging.getLogger(name)
        _lg.setLevel(pinned_level)
        # Force propagation in case some import path (vllm's dictConfig,
        # bittensor's import-time basicConfig) silently set propagate=False
        # on an ancestor logger and our records can't reach the root handler.
        _lg.propagate = True
        # And re-enable the logger in case dictConfig disabled it.
        _lg.disabled = False
    # Quiet vLLM's own noisy startup banner without dropping its warnings.
    for noisy in ("vllm", "vllm.engine", "vllm.executor"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
    """Construct VLLMAdapter (the ~30-60s blocking vLLM init)."""
    from vllm_adapter import VLLMAdapter
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
    by default) — GRAIL sketches are bit-sensitive to attention kernel
    variance, so a different impl produces hidden states that don't
    match the validator's recomputed states and every commitment fails.
    """
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(f"cuda:{proof_gpu}").eval()


def _gpu_sync_and_clear(gpu_id: int) -> None:
    """Synchronize GPU and clear caches to release vLLM context locks.

    In single-GPU mode, vLLM may still hold CUDA context locks or use
    background streams after initialization. This makes the device idle
    before loading the HF proof model on the same GPU.
    """
    import time
    import torch
    logger = logging.getLogger("reliquary.cli")
    try:
        torch.cuda.set_device(gpu_id)
        torch.cuda.synchronize(gpu_id)
        torch.cuda.empty_cache()
        time.sleep(0.5)
        torch.cuda.synchronize(gpu_id)
        logger.debug("GPU %d synchronized and cleared", gpu_id)
    except Exception as e:
        logger.warning("GPU sync/clear failed: %s", e)


def _build_hf_generation_model(local_path: str, gen_gpu: int, attn_impl: str):
    """Construct the HF generation model (upstream fallback path)."""
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(f"cuda:{gen_gpu}").eval()


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
    # ----------------------- vLLM-specific options -------------------------
    use_vllm: bool = typer.Option(
        False,
        "--use-vllm",
        help=(
            "Use vLLM for generation (5-10× faster than HF .generate(); "
            "closes the window_mismatch race on busy validator cycles). "
            "Requires vllm_adapter.py at the reliquary repo root."
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
):
    """Run Reliquary miner (HF by default; pass --use-vllm for vLLM backend)."""
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    logger.info(
        "Starting Reliquary miner (network=%s, netuid=%d, env=%s, backend=%s)",
        network, netuid, environment, "vllm" if use_vllm else "hf",
    )

    async def _run():
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

        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        # --- Resolve initial checkpoint from validator if available ---
        initial_path = checkpoint  # fallback to --checkpoint arg
        try:
            if validator_url:
                url = validator_url
            else:
                metagraph = await get_metagraph(subtensor, NETUID)
                url = discover_validator_url(metagraph)

            import httpx
            from huggingface_hub import snapshot_download
            async with httpx.AsyncClient(timeout=30) as client:
                state = await get_window_state_v2(url, client=client)
            if state.checkpoint_repo_id and state.checkpoint_revision:
                logger.info(
                    "Validator at %s is on checkpoint %d (%s@%s). "
                    "Downloading to seed the miner model.",
                    url, state.checkpoint_n, state.checkpoint_repo_id,
                    state.checkpoint_revision[:12],
                )
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
        logger.info("Loading tokenizer from %s ...", initial_path)
        tokenizer = AutoTokenizer.from_pretrained(initial_path)
        if tokenizer.pad_token_id is None:
            # Required by the engine's batched generate path. Mirroring HF
            # convention: when no explicit pad token, use eos.
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Single-GPU detection — applies to both backends.
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise RuntimeError("no CUDA devices visible — miner needs at least 1 GPU")
        effective_proof_gpu = proof_gpu if gpu_count >= 2 else 0
        if effective_proof_gpu != proof_gpu:
            logger.info(
                "Only 1 GPU visible; forcing --proof-gpu=0 (was %d). "
                "Generation and proofs will share cuda:0.",
                proof_gpu,
            )
        single_gpu = (vllm_gpu == effective_proof_gpu)

        if use_vllm:
            # Import surfaced here so we get a clean error if vllm_adapter
            # isn't reachable from sys.path before we burn time on the
            # tokenizer load above.
            try:
                from vllm_adapter import VLLMAdapter  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "--use-vllm requires vllm_adapter.py at the reliquary "
                    f"repo root. Searched: {_REPO_ROOT}. Original error: {e}"
                )

            if single_gpu:
                # On a single-GPU box we MUST serialize: while vLLM is
                # capturing CUDA graphs, any other tensor op on the same
                # device fails with "operation not permitted when stream
                # is capturing". The HF .to(cuda:N) would race against
                # vLLM's graph capture and crash mid-capture (~89% complete
                # in the observed case). Serial loading costs ~10-15s vs
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
                # 2-GPU box: overlap the slow vLLM init (~60-90s) with the
                # HF proof model load (~10-15s).
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
            # Upstream HF-on-both-cards path (backwards compat). Two
            # AutoModelForCausalLM instances, generation on vllm_gpu and
            # proofs on proof_gpu (or both on cuda:0 if only one GPU).
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

        logger.info("loading environment: %s", environment)
        env = load_environment(environment)
        logger.info("environment loaded: %s (n_prompts=%d)", environment, len(env))

        logger.info(
            "constructing MiningEngine (backend=%s, vllm_gpu=%d, proof_gpu=%d, "
            "stats_path=%s)",
            "vllm" if use_vllm else "hf",
            vllm_gpu, effective_proof_gpu, stats_path or "<disabled>",
        )
        # Wrap MiningEngine to inject GPU sync before checkpoint reloads in single-GPU mode
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

        # Seed engine's _loaded_checkpoint_path so the first
        # maybe_pull_checkpoint sees we're already synced and skips a
        # redundant reload of the same weights.
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
        subtensor = await get_subtensor()

        if train:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from reliquary.constants import ATTN_IMPLEMENTATION
            from reliquary.environment import load_environment
            from reliquary.validator.service import ValidationService

            logger.info("Loading model from %s...", checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to("cuda:0").eval()

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
            # Run training + scoring as two independent concurrent loops so
            # set_weights fires once per subnet epoch instead of being
            # gated by the trainer's window timeouts.
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
