"""Reliquary CLI — mine and validate commands."""

import asyncio
import logging
import os
import threading

import typer

from reliquary.constants import (
    DEFAULT_BASE_MODEL, DEFAULT_HF_REPO_ID, ENVIRONMENT_NAME,
    MAX_NEW_TOKENS_PROTOCOL_CAP, VALIDATOR_HTTP_PORT,
)

app = typer.Typer(name="reliquary", help="Reliquary — Verifiable Inference Subnet")

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    # ``%(threadName)s`` distinguishes the main asyncio loop from the
    # dedicated ``weight-setter`` thread (see ``validate`` below) when
    # tailing logs.
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def mine(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    environment: str = typer.Option(
        os.getenv("RELIQUARY_ENVIRONMENT_NAME", ENVIRONMENT_NAME),
        help="Environment name (env: RELIQUARY_ENVIRONMENT_NAME)",
    ),
    validator_url: str = typer.Option(
        "",
        help=(
            "Override the validator URL (otherwise discovered from the metagraph). "
            "Useful for local testing — e.g. http://127.0.0.1:8888"
        ),
    ),
    max_new_tokens: int = typer.Option(
        MAX_NEW_TOKENS_PROTOCOL_CAP,
        help=(
            "Max new tokens per rollout (protocol cap is "
            f"{MAX_NEW_TOKENS_PROTOCOL_CAP}). Lower trades completion "
            "length for cycle time / VRAM — at risk of BAD_TERMINATION "
            "rejects when the model can't finish in budget."
        ),
    ),
    vllm_gpu_memory_fraction: float = typer.Option(
        0.5,
        help=(
            "Fraction of free GPU memory vLLM may allocate for weights + "
            "KV cache. The HF proof model and its activations share the "
            "same device on single-GPU boxes, so leave a margin: 0.5 keeps "
            "~half the GPU for GRAIL proofs."
        ),
    ),
    sigma_filter: bool = typer.Option(
        True,
        "--sigma-filter/--no-sigma-filter",
        help=(
            "Drop rollout groups locally when reward σ < SIGMA_MIN (the "
            "validator's OUT_OF_ZONE threshold). Saves submission slots "
            "but silences all validator-side feedback for filtered "
            "groups. Disable to surface the validator's verdicts on "
            "every group (most will be OUT_OF_ZONE, but you'll see the "
            "race-time / freshness behaviour clearly)."
        ),
    ),
    prompt_cache: bool = typer.Option(
        True,
        "--prompt-cache/--no-prompt-cache",
        help=(
            "Track per-prompt-idx σ history on disk and bias sampling "
            "toward indices that have previously landed in σ-zone. "
            "Off-by-default sampling is uniform over the math/augmented_math "
            "subset; with cache on, sampling shifts to the small minority "
            "of prompts Qwen3 doesn't solve all-correct."
        ),
    ),
    prompt_cache_path: str = typer.Option(
        "",
        help=(
            "Path to the prompt-σ JSONL cache. Empty uses "
            "~/.cache/reliquary/prompt_sigma.jsonl. Ignored if "
            "--no-prompt-cache."
        ),
    ),
    prompt_cache_hot_bias: float = typer.Option(
        0.5,
        help=(
            "Probability of drawing from the in-zone history set vs. "
            "exploring the full pool. 0 = always explore, 1 = always "
            "exploit. Default 0.5 splits between exploit and explore."
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run Reliquary miner."""
    setup_logging(log_level)
    logger = logging.getLogger("reliquary.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    logger.info(
        "Starting Reliquary miner (network=%s, netuid=%d, env=%s)",
        network, netuid, environment,
    )

    async def _run():
        import bittensor as bt
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.environment import load_environment
        from reliquary.infrastructure.chain import get_subtensor, get_metagraph, NETUID
        from reliquary.miner.engine import MiningEngine
        from reliquary.miner.submitter import discover_validator_url, get_window_state_v2

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

        # --- Load models from resolved path ---
        logger.info("Loading models from %s...", initial_path)
        tokenizer = AutoTokenizer.from_pretrained(initial_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Use 2 GPUs when available (vLLM on 0, HF proof on 1). On a
        # single-GPU box they share device 0 — load HF proof model FIRST
        # so vLLM sizes its KV cache against the remaining headroom.
        proof_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"

        hf_model = AutoModelForCausalLM.from_pretrained(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to(proof_device).eval()

        # Cap vLLM's max sequence length at the protocol's new-token cap
        # plus a generous prompt budget. Math prompts are short (~few
        # hundred tokens); 1024 of headroom is plenty.
        vllm_max_model_len = max_new_tokens + 1024
        vllm_kwargs = {
            "dtype": "bfloat16",
            "gpu_memory_utilization": vllm_gpu_memory_fraction,
            "max_model_len": vllm_max_model_len,
            "enforce_eager": False,
            "disable_log_stats": True,
        }
        from vllm import LLM
        vllm_model = LLM(model=initial_path, **vllm_kwargs)

        env = load_environment(environment)
        # If the CLI snapshot_download'd the validator's current revision
        # and used it as ``initial_path``, seed the engine so the first
        # maybe_pull_checkpoint sees we're already synced and skips an
        # immediate (and expensive) vLLM teardown + rebuild.
        seeded_n = 0
        seeded_hash = ""
        if initial_path != checkpoint:
            try:
                async with httpx.AsyncClient(timeout=30) as _c:
                    _st = await get_window_state_v2(url, client=_c)
                seeded_n = _st.checkpoint_n
                seeded_hash = _st.checkpoint_revision or ""
            except Exception:
                seeded_n, seeded_hash = 0, ""

        cache_obj = None
        if prompt_cache:
            from reliquary.miner.prompt_cache import (
                DEFAULT_CACHE_PATH, PromptSigmaCache,
            )
            cache_obj = PromptSigmaCache(
                path=prompt_cache_path or DEFAULT_CACHE_PATH,
                hot_bias=prompt_cache_hot_bias,
            )

        engine = MiningEngine(
            vllm_model,
            hf_model,
            tokenizer,
            wallet,
            env,
            proof_gpu=0 if proof_device == "cuda:0" else 1,
            max_new_tokens=max_new_tokens,
            validator_url_override=validator_url or None,
            vllm_kwargs=vllm_kwargs,
            initial_checkpoint_n=seeded_n,
            initial_checkpoint_hash=seeded_hash,
            sigma_filter=sigma_filter,
            prompt_cache=cache_obj,
        )
        if initial_path != checkpoint:
            engine._loaded_checkpoint_path = initial_path

        logger.info("Miner ready. Entering main loop.")
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
    checkpoint: str = typer.Option(DEFAULT_BASE_MODEL, help="HF repo id or local path of the model to load (trainer mode only)"),
    environment: str = typer.Option(
        os.getenv("RELIQUARY_ENVIRONMENT_NAME", ENVIRONMENT_NAME),
        help="Environment name (trainer mode only; env: RELIQUARY_ENVIRONMENT_NAME)",
    ),
    http_host: str = typer.Option("0.0.0.0", help="HTTP bind address (trainer mode only)"),
    http_port: int = typer.Option(VALIDATOR_HTTP_PORT, help="HTTP listen port (trainer mode only)"),
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
            # Run the weight setter in a dedicated OS thread with its own
            # event loop. asyncio is single-threaded, so any sync blocking
            # call on the trainer's loop (e.g. /state acquiring a lock the
            # GRAIL verifier is holding) would stall set_weights too. The
            # weight setter's own subtensor (see WeightOnlyValidator.run)
            # plus its own loop here means neither side can block the other.
            from reliquary.validator.weight_only import WeightOnlyValidator

            def _run_weight_setter() -> None:
                try:
                    worker = WeightOnlyValidator(wallet=wallet, netuid=netuid)
                    asyncio.run(worker.run())
                except Exception:
                    logger.exception("weight-setter thread crashed")

            threading.Thread(
                target=_run_weight_setter,
                name="weight-setter",
                daemon=True,
            ).start()
            await service.run(subtensor)
        else:
            from reliquary.validator.weight_only import WeightOnlyValidator

            validator = WeightOnlyValidator(wallet=wallet, netuid=netuid)
            await validator.run()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
