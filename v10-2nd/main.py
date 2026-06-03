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
    gpu_memory_utilization: float = typer.Option(
        0.55,
        help=(
            "Fraction of GPU memory vLLM may use. On a single GPU the HF proof "
            "model + forward run in the remainder, so keep this below ~0.7."
        ),
    ),
    pool_size: int = typer.Option(
        64, help="Target number of prepared in-zone groups kept ready (≈ windows × 8 slots)."
    ),
    gen_batch: int = typer.Option(
        8, help="Candidate prompts sampled per vLLM pregeneration batch."
    ),
    oversample: int = typer.Option(
        int(os.getenv("RELIQUARY_OVERSAMPLE", "8")),
        help=(
            "Completions sampled per prompt; keep the first 8 that terminate "
            "with EOS. Raise above 8 for low-termination (long-CoT) checkpoints "
            "(env: RELIQUARY_OVERSAMPLE)."
        ),
    ),
    prompt_sources: str = typer.Option(
        os.getenv("RELIQUARY_PROMPT_SOURCES", "gsm8k,augmented_gsm8k"),
        help=(
            "Comma-separated OMI problem_source values to mine. Long-CoT "
            "checkpoints only terminate on GSM8K-style prompts; empty = all "
            "(env: RELIQUARY_PROMPT_SOURCES)."
        ),
    ),
    frontier: bool = typer.Option(
        os.getenv("RELIQUARY_FRONTIER", "1") == "1",
        "--frontier/--no-frontier",
        help=(
            "Online frontier predictor for prompt selection: learns which "
            "prompts yield in-zone (2-6/8) groups on the CURRENT checkpoint "
            "from observed outcomes, and biases sampling toward them."
        ),
    ),
    max_new_tokens: int = typer.Option(
        int(os.getenv("RELIQUARY_MAX_NEW_TOKENS", "8192")),
        help="Max completion tokens per rollout (protocol cap 8192; env: RELIQUARY_MAX_NEW_TOKENS).",
    ),
    prompt_idx_file: str = typer.Option(
        os.getenv("RELIQUARY_PROMPT_IDX_FILE", ""),
        help=(
            "Path to a JSON list of OMI prompt indices to mine EXCLUSIVELY "
            "(overrides --prompt-sources/--frontier). Use a curated set of prompts "
            "confirmed in-zone (2-6/8) on the current checkpoint "
            "(env: RELIQUARY_PROMPT_IDX_FILE)."
        ),
    ),
    decool_snipe: bool = typer.Option(
        os.getenv("RELIQUARY_DECOOL_SNIPE", "0") == "1",
        "--decool-snipe/--no-decool-snipe",
        help=(
            "Prioritise mining prompts that just EXITED the validator's cooldown "
            "(recently rewarded = likely still in-zone and now submittable), with "
            "broad exploration as fallback (env: RELIQUARY_DECOOL_SNIPE)."
        ),
    ),
    two_stage: bool = typer.Option(
        os.getenv("RELIQUARY_TWO_STAGE", "0") == "1",
        "--two-stage/--no-two-stage",
        help=(
            "Two-stage discovery funnel: cheap screen (small oversample, short cap) "
            "to find prompts that terminate AND are ~50/50, then deep-mine only "
            "those at full oversample. Rejects ramblers/8-8 cheaply, multiplying "
            "distinct-prompt throughput (env: RELIQUARY_TWO_STAGE)."
        ),
    ),
    symbolic_only: bool = typer.Option(
        os.getenv("RELIQUARY_SYMBOLIC_ONLY", "0") == "1",
        "--symbolic-only/--no-symbolic-only",
        help=(
            "Restrict the candidate pool to prompts with SYMBOLIC expected answers "
            "(expressions/LaTeX/variables, not plain numbers). These are where the "
            "converged model is ~50/50 (in-zone); numeric answers it solves 8/8. "
            "~3x in-zone density (env: RELIQUARY_SYMBOLIC_ONLY)."
        ),
    ),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run Reliquary miner (vLLM generation + HF GRAIL proof + pregeneration)."""
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.environment import load_environment
        from reliquary.infrastructure.chain import get_subtensor, get_metagraph, NETUID
        from reliquary.miner.engine import MiningEngine
        from reliquary.miner.pregen import Pregenerator
        from reliquary.miner.submitter import discover_validator_url, get_window_state_v2
        from reliquary.miner.vllm_backend import VLLMGenerator
        from reliquary.protocol.grail_verifier import GRAILVerifier
        from reliquary.shared.hf_compat import resolve_hidden_size

        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        # --- Resolve initial checkpoint from validator if available ---
        initial_path = checkpoint  # fallback to --checkpoint arg
        init_repo_id: str | None = None
        init_revision: str = ""    # used verbatim as checkpoint_hash; "" = bootstrap sentinel
        init_n: int = 0
        init_cooldown: list[int] = []  # live recent-winners = current frontier seed
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
            init_cooldown = list(state.cooldown_prompts or [])
            if init_cooldown:
                logger.info("seeding frontier predictor with %d live cooldown winners", len(init_cooldown))
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
                init_repo_id = state.checkpoint_repo_id
                init_revision = state.checkpoint_revision
                init_n = state.checkpoint_n
                logger.info("Using initial checkpoint path: %s", initial_path)
            else:
                logger.info(
                    "Validator has no published checkpoint yet — using --checkpoint=%s",
                    checkpoint,
                )
        except Exception as e:
            logger.warning(
                "Could not fetch validator checkpoint (%s); falling back to "
                "--checkpoint=%s", e, checkpoint,
            )

        # --- Tokenizer + EOS set (matches validator._eos_set_from_model) ---
        tokenizer = AutoTokenizer.from_pretrained(initial_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        eos_set: set[int] = set()
        try:
            gcfg = GenerationConfig.from_pretrained(initial_path)
            geos = getattr(gcfg, "eos_token_id", None)
            if isinstance(geos, int):
                eos_set.add(int(geos))
            elif geos:
                eos_set.update(int(e) for e in geos)
        except Exception:
            pass
        if tokenizer.eos_token_id is not None:
            eos_set.add(int(tokenizer.eos_token_id))

        ndev = torch.cuda.device_count()
        proof_device = "cuda:1" if ndev >= 2 else "cuda:0"
        # On 2 GPUs vLLM owns GPU0 fully; on 1 GPU it shares with the HF proof.
        gmu = 0.9 if ndev >= 2 else gpu_memory_utilization

        # --- vLLM generator FIRST so it reserves its memory fraction, then the
        #     HF proof model loads into the remainder (single-GPU) or GPU1. ---
        # Bound max_model_len so vLLM doesn't try to size the KV cache for
        # Qwen3's full context window under a reduced gpu_memory_utilization
        # (a common "max seq len larger than KV cache" init failure). Prompts
        # in OpenMathInstruct are short; allow ~1024 tokens of prompt headroom.
        vllm_max_len = int(max_new_tokens) + 1024
        logger.info(
            "Initialising vLLM generator (gpu_mem_util=%.2f, max_model_len=%d) on %s",
            gmu, vllm_max_len, initial_path,
        )
        vllm_gen = VLLMGenerator(
            initial_path,
            eos_token_ids=eos_set,
            oversample=oversample,
            gpu_memory_utilization=gmu,
            max_model_len=vllm_max_len,
        )

        logger.info("Loading HF proof model on %s ...", proof_device)
        hf_model = AutoModelForCausalLM.from_pretrained(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to(proof_device).eval()

        env = load_environment(environment)
        verifier = GRAILVerifier(hidden_dim=resolve_hidden_size(hf_model))

        # The live cooldown is ALL-TIME accumulated (permanent horizon), so it is
        # dominated by historical augmented_gsm8k winners from earlier/weaker
        # checkpoints. The CURRENT converged checkpoint solves those 8/8; the live
        # in-zone frontier is augmented_math (confirmed via top-miner accepts).
        # Seed the frontier only from cooled exemplars whose source is in the
        # mined set, so it learns the CURRENT signature, not the stale one.
        if init_cooldown:
            try:
                _src = env._dataset["problem_source"]
                _N = len(_src)
                _mine = {s.strip() for s in prompt_sources.split(",") if s.strip()}
                _filtered = [i for i in init_cooldown if _src[i % _N] in _mine]
                logger.info("frontier seed filtered to mined sources %s: %d/%d cooled exemplars",
                            sorted(_mine), len(_filtered), len(init_cooldown))
                init_cooldown = _filtered or init_cooldown
            except Exception:
                logger.exception("seed-source filter failed; using full cooldown seed")

        explicit_pool_idxs = None
        if prompt_idx_file:
            import json as _json
            with open(prompt_idx_file) as _f:
                explicit_pool_idxs = [int(x) for x in _json.load(_f)]
            logger.info(
                "explicit prompt-idx pool: %d prompts from %s (overrides sources/frontier)",
                len(explicit_pool_idxs), prompt_idx_file,
            )

        pregen = Pregenerator(
            vllm_gen=vllm_gen,
            hf_model=hf_model,
            tokenizer=tokenizer,
            env=env,
            verifier=verifier,
            proof_device=proof_device,
            checkpoint_n=init_n,
            checkpoint_hash=init_revision,
            model_path=initial_path,
            model_name=getattr(hf_model, "name_or_path", initial_path),
            repo_id=init_repo_id,
            target_ready=pool_size,
            gen_batch_size=gen_batch,
            max_new_tokens=max_new_tokens,
            prompt_sources={s.strip() for s in prompt_sources.split(",") if s.strip()} or None,
            use_frontier=frontier,
            winners_path=os.getenv("RELIQUARY_WINNERS_PATH", "/root/wf_data/winners.jsonl"),
            controls_path=os.getenv("RELIQUARY_CONTROLS_PATH", "/root/wf_data/controls.jsonl"),
            frontier_save_path=os.getenv("RELIQUARY_FRONTIER_MODEL", "/root/frontier_model.npz"),
            seed_positive_idxs=init_cooldown,
            explicit_pool_idxs=explicit_pool_idxs,
            decool_snipe=decool_snipe,
            two_stage=two_stage,
            symbolic_only=symbolic_only,
        )

        engine = MiningEngine(
            pregen,
            wallet,
            env,
            validator_url_override=validator_url or None,
        )

        logger.info("Miner ready. Pregeneration thread starting; entering submit loop.")
        try:
            await engine.mine_window(subtensor, 0, use_drand=use_drand)
        except KeyboardInterrupt:
            logger.info("Miner interrupted by user")
        except Exception as e:
            logger.error("Mining loop crashed: %s", e, exc_info=True)
            raise
        finally:
            pregen.stop()

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
