"""Reliquary CLI — mine and validate commands."""

import asyncio
import logging
import os
import threading

import typer

from reliquary.constants import (
    DEFAULT_BASE_MODEL,
    DEFAULT_HF_REPO_ID,
    ENVIRONMENT_NAME,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    SIGMA_MIN,
    VALIDATOR_HTTP_PORT,
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
            "Max completion tokens per rollout. The validator caps at "
            f"{MAX_NEW_TOKENS_PROTOCOL_CAP} (protocol cap) and allows at most 1 "
            "truncated rollout per group of 8 before rejecting BAD_TERMINATION. "
            "At the protocol cap Qwen3-4B on OpenMathInstruct truncates ~1/8 "
            "statistically — already at the limit. Lower this only if your model "
            "EOSes earlier on the active env; otherwise expect BAD_TERMINATION "
            "rejects to dominate. Faster window cycle is achieved primarily via "
            "--use-vllm and --pregen-buffer-size, not by shortening this."
        ),
    ),
    min_sigma: float = typer.Option(
        SIGMA_MIN,
        help=(
            "Skip prompts whose reward σ across the 8 rollouts falls below this. "
            f"Default {SIGMA_MIN} = validator's steady-state SIGMA_MIN; lower to 0.33 "
            "if mining against a bootstrap-window validator."
        ),
    ),
    prescreen_k: int = typer.Option(
        4,
        help=(
            "Probe rollouts to generate before committing to a prompt. If all probe "
            "rollouts return the same reward (all-correct or all-incorrect), the "
            "prompt is skipped before paying for the remaining (M-k) generations + "
            "the 8× GRAIL forward pass. 0 disables the probe. Must be < 8."
        ),
    ),
    prescreen_max_tokens: int = typer.Option(
        0,
        help=(
            "Max completion tokens for probe rollouts. 0 (default) = match "
            "--max-new-tokens for length parity. Set lower (e.g. 1024) to "
            "trade some false-negative skips for a faster probe on extreme prompts."
        ),
    ),
    difficulty_blacklist_size: int = typer.Option(
        4096,
        help=(
            "Bounded in-memory cache of prompts the miner saw go OUT_OF_ZONE under "
            "the current checkpoint. Avoids re-trying them every window. Cleared on "
            "checkpoint pull (new policy → new judgments). 0 disables the cache."
        ),
    ),
    use_vllm: bool = typer.Option(
        True,
        "--use-vllm/--no-vllm",
        help=(
            "Load the generation model with vLLM (continuous batching, paged "
            "attention) — typically 5-10× faster than HuggingFace .generate() for "
            "our M=8 sample-per-prompt workload. Falls back to HuggingFace if vLLM "
            "is not installed."
        ),
    ),
    vllm_gpu_memory_utilization: float = typer.Option(
        0.85,
        help=(
            "Fraction of generation-GPU memory vLLM may use for KV cache + weights. "
            "vLLM-only. Leave headroom on a single-GPU box for the HF proof model."
        ),
    ),
    pregen_buffer_size: int = typer.Option(
        4,
        help=(
            "Number of rollout groups to pre-generate during the validator's non-OPEN "
            "phase (TRAINING / PUBLISHING / READY). Pre-gen entries are finalized + "
            "POSTed in a burst the instant OPEN flips, racing for early drand-round "
            "slots. Capped at 8 (per-hotkey submission cap). 0 disables pre-gen."
        ),
    ),
    state_poll_ms: int = typer.Option(
        100,
        help=(
            "How often (ms) to poll the validator's /state endpoint when waiting for "
            "OPEN with a primed buffer. Lower = earlier detection = bigger slot share, "
            "but more load on the validator."
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
            # Retry /state on transient 503 — happens between window seal
            # and next window OPEN, ~few seconds. Skipping this puts us
            # in the OOM-on-rebuild trap: we load --checkpoint (Qwen3 base),
            # then mine_window pulls the validator's actual checkpoint, and
            # vLLM has to be torn down + rebuilt holding ~90 GiB of stale
            # memory.
            import asyncio as _asyncio
            state = None
            _last_exc = None
            async with httpx.AsyncClient(timeout=30) as client:
                for attempt in range(6):  # ~60 s budget
                    try:
                        state = await get_window_state_v2(url, client=client)
                        break
                    except Exception as exc:
                        _last_exc = exc
                        logger.warning(
                            "/state attempt %d failed: %s (retrying in 10s)",
                            attempt + 1, exc,
                        )
                        await _asyncio.sleep(10)
                if state is None:
                    raise _last_exc if _last_exc else RuntimeError("/state never succeeded")
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

        # Use 2 GPUs when available (gen on 0, HF proof on 1). Fall back to
        # sharing GPU 0 for test boxes that only expose one device. With
        # vLLM enabled, the generation model is a ``vllm.LLM`` instance on
        # GPU 0; with --no-vllm it's a HF AutoModelForCausalLM, same
        # placement.
        proof_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"

        gen_model = None
        gen_backend = "hf"
        if use_vllm:
            try:
                from vllm import LLM
                logger.info(
                    "Loading vLLM engine on cuda:0 "
                    "(gpu_memory_utilization=%.2f)...",
                    vllm_gpu_memory_utilization,
                )
                # max_model_len: Qwen3-4B's config exposes 262144 (256K YaRN-
                # extended). vLLM's KV cache scales linearly with this, so
                # leaving it at config default needs ~128 GiB just for one
                # sequence — exceeds our gpu_memory_utilization budget. Cap
                # at a value comfortably above our actual usage:
                #   prompt (~500 tokens) + max_new_tokens (8192) + slack
                # = 16384 fits everything the miner generates. Bumping this
                # only matters if you want vLLM to accept longer prompts
                # than the env supplies.
                _vllm_max_model_len = min(
                    16384,
                    max_new_tokens + 4096,
                )
                gen_model = LLM(
                    model=initial_path,
                    dtype="bfloat16",
                    gpu_memory_utilization=vllm_gpu_memory_utilization,
                    enforce_eager=False,
                    trust_remote_code=True,
                    max_model_len=_vllm_max_model_len,
                    # Cap concurrent sequences vLLM may schedule. M_ROLLOUTS=8
                    # per submission, pre-gen buffer up to 4 → max 32 in
                    # flight on the heaviest schedule. 16 is safe and keeps
                    # KV cache pressure predictable.
                    max_num_seqs=16,
                )
                gen_backend = "vllm"
            except ImportError:
                logger.warning(
                    "vLLM not installed; falling back to HuggingFace generation. "
                    "Install with `pip install vllm` for ~5-10× faster rollouts."
                )
            except Exception:
                logger.exception(
                    "vLLM init failed; falling back to HuggingFace generation"
                )

        if gen_model is None:
            gen_model = AutoModelForCausalLM.from_pretrained(
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
            gen_model,
            hf_model,
            tokenizer,
            wallet,
            env,
            proof_gpu=0 if proof_device == "cuda:0" else 1,
            validator_url_override=validator_url or None,
            max_new_tokens=max_new_tokens,
            min_sigma=min_sigma,
            prescreen_k=prescreen_k,
            prescreen_max_tokens=prescreen_max_tokens,
            difficulty_blacklist_size=difficulty_blacklist_size,
            pregen_buffer_size=pregen_buffer_size,
            state_poll_ms=state_poll_ms,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        logger.info(
            "Mining tuning: backend=%s, max_new_tokens=%d, min_sigma=%.3f, "
            "prescreen_k=%d, prescreen_max_tokens=%d, "
            "difficulty_blacklist_size=%d, pregen_buffer_size=%d, "
            "state_poll_ms=%d",
            gen_backend, max_new_tokens, min_sigma, prescreen_k,
            prescreen_max_tokens, difficulty_blacklist_size,
            pregen_buffer_size, state_poll_ms,
        )

        # Seed engine state so the first maybe_pull_checkpoint short-
        # circuits when the validator's published checkpoint matches what
        # we already loaded at boot. Without ``_initial_local_n``, the
        # mine loop starts with local_n=0 and always triggers a redundant
        # vLLM rebuild on the first /state poll — OOMing the GPU because
        # the old vLLM hasn't released its memory yet.
        if initial_path != checkpoint:
            engine._loaded_checkpoint_path = initial_path
        # ``state`` may be unbound if the earlier validator-discovery try
        # block raised before assigning it. Guard with locals().
        _boot_state = locals().get("state", None)
        if _boot_state is not None:
            try:
                if _boot_state.checkpoint_repo_id and _boot_state.checkpoint_revision:
                    engine._initial_local_n = int(_boot_state.checkpoint_n)
                    engine._initial_local_hash = str(_boot_state.checkpoint_revision)
            except (AttributeError, TypeError, ValueError):
                pass

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
