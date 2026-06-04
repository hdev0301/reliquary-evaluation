"""Pregen -> Supabase producer (GPU box).

Builds the SAME stack as `reliquary.cli.main mine` (vLLM + HF proof + Pregenerator)
but: (a) runs honest CURATE=0 generation, (b) swaps the in-memory PregenStore for one
that mirrors every prepared group into Supabase (full GRAIL artifacts), and (c) NEVER
submits — submission is the consumer's job (submit_from_supabase). Keep RELIQUARY_CURATE=0.

Env: RELIQUARY_CHECKPOINT, RELIQUARY_VALIDATOR_URL, RELIQUARY_OVERSAMPLE,
     RELIQUARY_GEN_BATCH, RELIQUARY_POOL_SIZE, RELIQUARY_MAX_NEW_TOKENS,
     RELIQUARY_PROMPT_SOURCES, RELIQUARY_PROMPT_IDX_FILE, RELIQUARY_FRONTIER,
     RELIQUARY_SUPABASE_URL/KEY/TABLE, BT_HOTKEY (target hotkey ss58 tag).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import sb  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("sb.producer")


def _envf(name, default):  # float env
    try: return float(os.environ.get(name, default))
    except Exception: return float(default)


def _envi(name, default):  # int env
    try: return int(os.environ.get(name, default))
    except Exception: return int(default)


async def _run():
    import bittensor as bt
    import torch
    from transformers import AutoTokenizer, GenerationConfig

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.environment import load_environment
    from reliquary.miner.pregen import Pregenerator, PregenStore
    from reliquary.miner.submitter import get_window_state_v2
    from reliquary.miner.vllm_backend import VLLMGenerator
    from reliquary.protocol.grail_verifier import GRAILVerifier
    from reliquary.shared.hf_compat import resolve_hidden_size
    from reliquary.shared.modeling import load_text_generation_model
    from huggingface_hub import snapshot_download
    import httpx

    if os.environ.get("RELIQUARY_CURATE", "0") != "0":
        log.warning("RELIQUARY_CURATE != 0 — this producer is meant for HONEST groups; forcing 0")
        os.environ["RELIQUARY_CURATE"] = "0"

    checkpoint = os.environ.get("RELIQUARY_CHECKPOINT", "Qwen/Qwen3.5-4B")
    validator_url = os.environ.get("RELIQUARY_VALIDATOR_URL", "")
    hotkey_ss58 = os.environ["PRODUCER_TARGET_SS58"]   # the stardev ss58 the consumer submits under
    environment = os.environ.get("RELIQUARY_ENVIRONMENT_NAME", "openmathinstruct")
    oversample = _envi("RELIQUARY_OVERSAMPLE", 40)
    gen_batch = _envi("RELIQUARY_GEN_BATCH", 40)
    pool_size = _envi("RELIQUARY_POOL_SIZE", 96)
    max_new_tokens = _envi("RELIQUARY_MAX_NEW_TOKENS", 1024)
    gpu_mem = _envf("RELIQUARY_GPU_MEM", 0.78)
    prompt_sources = os.environ.get("RELIQUARY_PROMPT_SOURCES", "gsm8k,augmented_gsm8k")
    frontier = os.environ.get("RELIQUARY_FRONTIER", "1") == "1"
    prompt_idx_file = os.environ.get("RELIQUARY_PROMPT_IDX_FILE", "")

    # --- resolve validator's published checkpoint (same as mine()) ---
    initial_path = checkpoint
    init_repo_id = None
    init_revision = ""
    init_n = 0
    init_cooldown = []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            state = await get_window_state_v2(validator_url, client=client)
        init_cooldown = list(state.cooldown_prompts or [])
        if state.checkpoint_repo_id and state.checkpoint_revision:
            log.info("validator on checkpoint %s (%s@%s); downloading",
                     state.checkpoint_n, state.checkpoint_repo_id, state.checkpoint_revision[:12])
            initial_path = snapshot_download(repo_id=state.checkpoint_repo_id,
                                             revision=state.checkpoint_revision)
            init_repo_id = state.checkpoint_repo_id
            init_revision = state.checkpoint_revision
            init_n = state.checkpoint_n
    except Exception as e:
        log.warning("could not fetch validator checkpoint (%s); using --checkpoint=%s", e, checkpoint)

    # model_name tag: use the published repo id (canonical, validator-acceptable);
    # the consumer signs with current()[2]=repo_id, so this is just informational.
    model_name = init_repo_id or checkpoint

    # --- tokenizer + EOS set (matches validator) ---
    tokenizer = AutoTokenizer.from_pretrained(initial_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_set = set()
    try:
        gcfg = GenerationConfig.from_pretrained(initial_path)
        geos = getattr(gcfg, "eos_token_id", None)
        if isinstance(geos, int): eos_set.add(int(geos))
        elif geos: eos_set.update(int(e) for e in geos)
    except Exception:
        pass
    if tokenizer.eos_token_id is not None:
        eos_set.add(int(tokenizer.eos_token_id))

    ndev = torch.cuda.device_count()
    proof_device = "cuda:1" if ndev >= 2 else "cuda:0"
    gmu = 0.9 if ndev >= 2 else gpu_mem
    vllm_max_len = int(max_new_tokens) + 1024

    log.info("init vLLM (gpu_mem=%.2f max_len=%d) on %s", gmu, vllm_max_len, initial_path)
    vllm_gen = VLLMGenerator(initial_path, eos_token_ids=eos_set, oversample=oversample,
                             gpu_memory_utilization=gmu, max_model_len=vllm_max_len)

    log.info("loading HF proof model on %s", proof_device)
    hf_model = load_text_generation_model(initial_path, dtype=torch.bfloat16,
                                          attn_implementation=ATTN_IMPLEMENTATION).to(proof_device).eval()

    env = load_environment(environment)
    hidden_dim = resolve_hidden_size(hf_model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)

    explicit_pool_idxs = None
    if prompt_idx_file:
        import json as _json
        with open(prompt_idx_file) as _f:
            explicit_pool_idxs = [int(x) for x in _json.load(_f)]
        log.info("explicit prompt-idx pool: %d prompts from %s", len(explicit_pool_idxs), prompt_idx_file)

    pregen = Pregenerator(
        vllm_gen=vllm_gen, hf_model=hf_model, tokenizer=tokenizer, env=env, verifier=verifier,
        proof_device=proof_device, checkpoint_n=init_n, checkpoint_hash=init_revision,
        model_path=initial_path, model_name=getattr(hf_model, "name_or_path", initial_path),
        repo_id=init_repo_id, target_ready=pool_size, gen_batch_size=gen_batch,
        max_new_tokens=max_new_tokens,
        prompt_sources={s.strip() for s in prompt_sources.split(",") if s.strip()} or None,
        use_frontier=frontier, seed_positive_idxs=init_cooldown,
        explicit_pool_idxs=explicit_pool_idxs, decool_snipe=False, two_stage=False, symbolic_only=False,
    )

    # --- swap the in-memory store for a Supabase-mirroring one BEFORE start() ---
    client = sb.SupabaseClient()

    class SupabaseWritingStore(PregenStore):
        def add(self, group):
            super().add(group)
            try:
                row = sb.serialize_group(group, model_name=model_name, hidden_dim=hidden_dim,
                                         miner_hotkey=hotkey_ss58, tier="honest_first8")
                client.upsert_group(row)
                log.info("supabase <- group prompt=%d sigma=%.3f n_corr=%d ckpt=%s",
                         group.prompt_idx, group.sigma, row["n_correct"], group.checkpoint_hash[:10])
            except Exception as e:
                log.warning("supabase write failed for prompt=%s: %r", getattr(group, "prompt_idx", "?"), e)

        def add_many(self, items):
            for g in items:
                self.add(g)

    pregen.store = SupabaseWritingStore()
    pregen.store.set_active_checkpoint(init_revision)
    # producer has no engine to wire providers; generate freely (consumer enforces
    # the live validator cooldown at submit). Seed exclusion with the live cooldown.
    _cool = set(int(i) for i in init_cooldown)
    pregen.set_cooldown_provider(lambda: _cool)
    pregen.set_priority_provider(lambda: [])

    log.info("starting pregen thread (table=%s, target ss58=%s, CURATE=0 honest)", client.table, hotkey_ss58)
    pregen.start()
    try:
        while True:
            await asyncio.sleep(30)
            log.info("alive | local ready=%d | supabase ready=%d",
                     pregen.store.ready_count(init_revision),
                     client.ready_count(init_revision, hotkey_ss58))
    finally:
        pregen.stop()


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        log.info("producer stopped")
