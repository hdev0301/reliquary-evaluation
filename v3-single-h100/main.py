"""Reliquary CLI — mine and validate commands."""

import asyncio
import logging
import os
import threading

import typer

from reliquary.constants import DEFAULT_BASE_MODEL, DEFAULT_HF_REPO_ID, ENVIRONMENT_NAME, VALIDATOR_HTTP_PORT

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
    # Suppress per-request HTTP traffic from chatty third-party libraries.
    # The /state poll fires every ~1 s and the HF snapshot download fires
    # many requests per file; their INFO lines drown out the miner's own
    # pregen/submit events. WARNING+ still surfaces real errors.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
        int(os.getenv("RELIQUARY_MAX_NEW_TOKENS", "1024")),
        help=(
            "Per-rollout generation cap. Protocol ceiling is 8192. Math "
            "completions typically finish in 200-600 tokens; the default "
            "of 1024 leaves headroom while keeping gen latency low so the "
            "pregen pipeline can stay ahead of the window. Set via "
            "RELIQUARY_MAX_NEW_TOKENS env var."
        ),
    ),
    pregen_capacity: int = typer.Option(
        int(os.getenv("RELIQUARY_PREGEN_CAPACITY", "12")),
        help=(
            "Number of pre-generated rollout batches held in the queue. "
            "Each batch is one prompt × 8 rollouts ready to sketch + fire "
            "as soon as the next window OPENs with randomness. Higher = "
            "more candidate prompts buffered (better resilience to OUT_OF_ZONE "
            "or PROMPT_FULL rejections), lower = less GPU memory pressure."
        ),
    ),
    bootstrap: bool = typer.Option(
        os.getenv("RELIQUARY_BOOTSTRAP", "0") in ("1", "true", "True"),
        help=(
            "Use the relaxed σ ≥ 0.33 threshold for the in-zone predictor "
            "(matches the validator's BOOTSTRAP_SIGMA_MIN). Only enable "
            "during the first BOOTSTRAP_WINDOWS=100 windows of a subnet."
        ),
    ),
    prescreen_rollouts: int = typer.Option(
        int(os.getenv("RELIQUARY_PRESCREEN_ROLLOUTS", "8")),
        help=(
            "Speculative pre-screen size. Generate N short rollouts (at "
            "--prescreen-max-tokens) on every fresh prompt and skip the "
            "full M=8 × 8192 gen if k=0 (model hopeless) or k=N (model "
            "trivially solves → likely OUT_OF_ZONE). 8 matches M_ROLLOUTS "
            "so the pre-screen batch saturates the H100's per-call "
            "throughput (3-rollout batches were memory-bandwidth-starved). "
            "Set to 0 to disable."
        ),
    ),
    prescreen_max_tokens: int = typer.Option(
        int(os.getenv("RELIQUARY_PRESCREEN_MAX_TOKENS", "512")),
        help=(
            "Max tokens per pre-screen rollout. Lower = cheaper but more "
            "noise. 512 catches the boxed answer on most solvable Qwen3 "
            "math completions while costing ~1/16th of the full gen."
        ),
    ),
    pregen_capacity_arg: int = typer.Option(
        int(os.getenv("RELIQUARY_PREGEN_CAPACITY_OVERRIDE", "0")),
        "--pregen-capacity-override",
        help=(
            "If > 0, override --pregen-capacity. Deep queue lets pregen "
            "keep generating across windows without dropping fresh in-zone "
            "batches. 12 is a good default on H100/80GB."
        ),
    ),
    share_model_copies: bool = typer.Option(
        os.getenv("RELIQUARY_SHARE_MODEL", "1") in ("1", "true", "True"),
        help=(
            "Use a single model instance for both generation and GRAIL "
            "proof (default on). Saves ~8 GB VRAM on the 4B base model — "
            "headroom for deeper pregen queue. Off only if you need the "
            "two-GPU split."
        ),
    ),
    gen_batch_prompts: int = typer.Option(
        int(os.getenv("RELIQUARY_GEN_BATCH_PROMPTS", "2")),
        help=(
            "K = number of distinct prompts batched into ONE .generate() "
            "call. With K=2 on H100/80GB and Qwen3-4B, the full gen runs "
            "shape (K × M, max_new_tokens) = (16, 8192). Same model-weight "
            "load per decode step, ~2× the useful work — ~25-40 %% better "
            "per-prompt throughput than K=1 (single-prompt batches). Bump "
            "to 3-4 if VRAM allows; lower to 1 to disable cross-prompt "
            "batching entirely."
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

        # Use 2 GPUs when available (vllm on 0, HF proof on 1). Fall back to
        # sharing GPU 0 for test boxes that only expose one device.
        proof_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"

        if share_model_copies and proof_device == "cuda:0":
            # ONE model instance backs both generation and GRAIL proof
            # paths. They never run concurrently (gen, then sketch), so
            # sharing is safe and saves ~8 GB VRAM on the 4B base —
            # crucial headroom for wider gen batches on H100/80GB
            # where two-copy + KV cache at max_new_tokens=8192 nearly
            # fills the device.
            shared_model = AutoModelForCausalLM.from_pretrained(
                initial_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to("cuda:0").eval()
            vllm_model = shared_model
            hf_model = shared_model
            logger.info("Loaded single shared model copy on cuda:0 (share_model_copies=True)")
        else:
            vllm_model = AutoModelForCausalLM.from_pretrained(
                initial_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to("cuda:0").eval()

            hf_model = AutoModelForCausalLM.from_pretrained(
                initial_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(proof_device).eval()

        env = load_environment(environment)
        engine = MiningEngine(
            vllm_model,
            hf_model,
            tokenizer,
            wallet,
            env,
            proof_gpu=0 if proof_device == "cuda:0" else 1,
            validator_url_override=validator_url or None,
            max_new_tokens=max_new_tokens,
            pregen_capacity=pregen_capacity,
            bootstrap=bootstrap,
            prescreen_rollouts=prescreen_rollouts,
            prescreen_max_tokens=prescreen_max_tokens,
            gen_batch_prompts=gen_batch_prompts,
        )

        # Seed engine's _loaded_checkpoint_path so the first
        # maybe_pull_checkpoint sees we're already synced (skips redundant reload).
        if initial_path != checkpoint:
            engine._loaded_checkpoint_path = initial_path

        logger.info(
            "Miner ready (max_new_tokens=%d pregen_capacity=%d "
            "prescreen=%dx%d gen_batch_prompts=%d bootstrap=%s). "
            "Entering main loop.",
            max_new_tokens, pregen_capacity,
            prescreen_rollouts, prescreen_max_tokens,
            gen_batch_prompts, bootstrap,
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
