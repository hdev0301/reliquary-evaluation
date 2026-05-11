"""Standalone launcher for the vLLM-backed Reliquary miner.

Why this exists: the stock ``reliquary mine`` CLI instantiates
``MiningEngine`` with an HF ``AutoModelForCausalLM`` as ``vllm_model``
(the name is a historical misnomer). This launcher bypasses the CLI and
constructs ``MiningEngine`` directly, swapping the HF generation path
for a real vLLM ``LLM`` (wrapped by ``VLLMAdapter``) — 5–10× throughput
gain that closes the ``window_mismatch`` race.

File placement (on the miner box):
    /root/reliquary/                            ← reliquary repo root
    ├── launcher.py                             ← THIS FILE
    ├── vllm_adapter.py                         ← VLLMAdapter wrapper
    ├── requirements-vllm.txt                   ← pip install instructions
    └── reliquary/
        └── miner/
            └── engine.py                       ← replace with engine-v3.py content

Because the engine lives at the canonical ``reliquary.miner.engine`` path,
we import it with a normal ``from reliquary.miner.engine import MiningEngine``
— the dynamic-load workaround used in earlier drafts is no longer needed.
The stock ``reliquary mine`` CLI also picks up the new engine, but the
two backends are differentiated by the adapter-detection sentinel in
``_load_checkpoint`` (HF model → HF reload path, ``VLLMAdapter`` → vLLM
rebuild path), so the CLI keeps working as an HF fallback if you ever
need it.

Topology for a 2-GPU box:
    cuda:0  → vLLM engine for generation (8 rollouts/batch via SamplingParams.n=8)
    cuda:1  → HF AutoModelForCausalLM for GRAIL proof construction
              (must match the validator's flash_attention_2 kernels for
              GRAIL sketches to verify — vLLM uses its own attention so
              its forward CAN'T be used for proofs)

Expected throughput change from the HF baseline:
    HF .generate() 8 rollouts × Qwen3-4B × ~5000 tokens ≈ 6–7 min/attempt
    vLLM same workload                                   ≈ 60–90 sec/attempt

That brings per-attempt latency under the validator's 2–3 min window
cycle, so we stop losing every submission to window_mismatch.

Usage:
    cd /root/reliquary
    source .venv/bin/activate
    python launcher.py \\
        --network finney \\
        --netuid 81 \\
        --wallet-name ronnywebdev \\
        --hotkey hdev0301 \\
        --checkpoint Qwen/Qwen3-4B-Instruct-2507 \\
        --validator-url http://86.38.238.30:8080 \\
        --vllm-gpu 0 \\
        --proof-gpu 1 \\
        --gpu-memory-utilization 0.85 \\
        --log-level INFO

Single-GPU mode (vLLM + HF sharing one card) — works but you'll need
``--gpu-memory-utilization 0.55`` or lower to leave room for HF, and
proof construction will compete with generation for SMs:

    python launcher.py ... --vllm-gpu 0 --proof-gpu 0 \\
        --gpu-memory-utilization 0.55
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path

logger = logging.getLogger("reliquary-vllm-launcher")


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reliquary miner (vLLM backend)")
    p.add_argument("--network", default="finney",
                   help="Bittensor network (default: finney)")
    p.add_argument("--netuid", type=int, default=81,
                   help="Subnet UID (default: 81)")
    p.add_argument("--wallet-name", required=True)
    p.add_argument("--hotkey", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="HF repo id or local path to seed checkpoint")
    p.add_argument("--validator-url", required=True,
                   help="http://ip:port of the validator HTTP server")
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
                   help="Skip drand beacon when deriving window randomness. "
                        "MUST match the validator's --use-drand setting.")
    p.add_argument("--stats-path", default=".reliquary_miner_stats.json",
                   help="Posterior persistence file (default: %(default)s). "
                        "Pass empty string to disable.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ----------------------------------------------------------------------
# Bootstrap helpers
# ----------------------------------------------------------------------

def _resolve_checkpoint(checkpoint: str) -> str:
    """Return a local filesystem path for *checkpoint*.

    If *checkpoint* is already a local directory, return it as-is.
    Otherwise treat it as a HF repo id and snapshot-download it.
    """
    p = Path(checkpoint)
    if p.exists() and p.is_dir():
        logger.info("using local checkpoint at %s", p)
        return str(p)

    logger.info("snapshot_download from HF: %s", checkpoint)
    from huggingface_hub import snapshot_download
    local_path = snapshot_download(
        repo_id=checkpoint,
        allow_patterns=["model.safetensors", "config.json",
                        "*.json", "tokenizer*", "*.txt"],
    )
    logger.info("checkpoint cached at %s", local_path)
    return local_path


def _build_vllm_adapter(local_path: str, args: argparse.Namespace):
    from vllm_adapter import VLLMAdapter
    logger.info(
        "building vLLM adapter: gpu=cuda:%d max_model_len=%d gpu_mem_util=%.2f",
        args.vllm_gpu, args.max_model_len, args.gpu_memory_utilization,
    )
    return VLLMAdapter(
        model_path=local_path,
        gpu_id=args.vllm_gpu,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )


def _build_hf_proof_model(local_path: str, proof_gpu: int):
    """Build the HF model used for GRAIL proof construction.

    MUST use the same attention impl the validator uses (flash_attention_2
    by default) — sketches are bit-sensitive to attention kernel variance.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from reliquary.constants import ATTN_IMPLEMENTATION
    logger.info(
        "loading HF proof model on cuda:%d (attn=%s)",
        proof_gpu, ATTN_IMPLEMENTATION,
    )
    return AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to(f"cuda:{proof_gpu}").eval()


def _build_tokenizer(local_path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(local_path)
    if tok.pad_token_id is None:
        # Required by the engine's batched generate path. Mirroring HF
        # convention: when no explicit pad token, use eos.
        tok.pad_token_id = tok.eos_token_id
    return tok


def _build_env():
    """Default to the math env, mirroring constants.ENVIRONMENT_NAME."""
    from reliquary.environment.math import MATHEnvironment
    return MATHEnvironment()


def _build_wallet(name: str, hotkey: str):
    import bittensor as bt
    return bt.Wallet(name=name, hotkey=hotkey)


async def _build_subtensor(network: str):
    import bittensor as bt
    subtensor = bt.AsyncSubtensor(network=network)
    await asyncio.wait_for(subtensor.initialize(), timeout=120.0)
    return subtensor


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

async def _amain(args: argparse.Namespace) -> int:
    # The engine lives at the canonical reliquary path now (we replaced
    # the upstream miner/engine.py with engine-v3.py's content), so a
    # normal package import works — no dynamic loading needed.
    from reliquary.miner.engine import MiningEngine

    local_path = await asyncio.to_thread(_resolve_checkpoint, args.checkpoint)
    tokenizer = await asyncio.to_thread(_build_tokenizer, local_path)
    env = await asyncio.to_thread(_build_env)
    wallet = _build_wallet(args.wallet_name, args.hotkey)

    # Build models. On a 2-GPU box (vllm_gpu != proof_gpu) we run them
    # concurrently so the slow vLLM init (~60-90s, includes CUDA graph
    # capture) overlaps with the HF proof model load (~10-15s).
    #
    # On a single-GPU box (vllm_gpu == proof_gpu) we MUST serialize: while
    # vLLM is capturing CUDA graphs, any other tensor op on the same
    # device fails with "operation not permitted when stream is
    # capturing". The HF .to(cuda:N) would race against vLLM's graph
    # capture and crash mid-capture (~89% complete in the observed case).
    # Serial loading costs ~10-15s vs parallel; well worth the
    # reliability on single-GPU setups.
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
        vllm_task = asyncio.to_thread(_build_vllm_adapter, local_path, args)
        hf_task = asyncio.to_thread(_build_hf_proof_model, local_path, args.proof_gpu)
        vllm_model, hf_model = await asyncio.gather(vllm_task, hf_task)

    stats_path = args.stats_path or None
    engine = MiningEngine(
        vllm_model=vllm_model,
        hf_model=hf_model,
        tokenizer=tokenizer,
        wallet=wallet,
        env=env,
        vllm_gpu=args.vllm_gpu,
        proof_gpu=args.proof_gpu,
        max_new_tokens=args.max_model_len,
        validator_url_override=args.validator_url,
        stats_path=stats_path,
    )

    subtensor = await _build_subtensor(args.network)
    logger.info(
        "miner ready: hotkey=%s validator=%s",
        wallet.hotkey.ss58_address, args.validator_url,
    )

    try:
        await engine.mine_window(
            subtensor,
            window_start=0,             # ignored in v2.1+
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
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    # Quiet vLLM's noisy startup banner without dropping its warnings.
    for noisy in ("vllm", "vllm.engine", "vllm.executor"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Cancel the main task on SIGTERM/SIGINT so cleanup runs.
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
