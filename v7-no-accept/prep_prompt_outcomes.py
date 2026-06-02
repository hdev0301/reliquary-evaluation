#!/usr/bin/env python3
"""Single-engine vLLM prep worker: classify + save pregen in one pass.

Each iteration:
  1. Pick prompt: cross-ckpt priors first, then random unseen (skip cooldown).
  2. Generate M=8 rollouts via vLLM at PREP_MAX_NEW_TOKENS.
  3. Compute σ; classify good / dud / oof; upsert outcome.
  4. If status="good" AND ≥3 of 8 rollouts terminated via EOS, save the
     same rollouts to pregen_batches.

vLLM is ~3-5× faster than HF transformers thanks to continuous batching
and paged attention. We chose vLLM because pregen throughput is the
binding constraint; HF-only was hitting ~3 saves/min, vLLM should hit
~10-15. The validator-acceptance of vLLM-sampled rollouts is untested
(see [[reliquary-validator-safety]] memory): theoretically vLLM's
sampling kernels produce a different reward-distribution shape than
HF, which might trip the validator's distribution_suspicious guard,
but we have no direct measurement — the user's earlier 100% reject
came from the miner's own HF code, not from vLLM-pregen consumption.

Validator-rule safety stance: we monitor the miner's reject reasons
after the miner-side pregen-consumption patch lands; if
distribution_suspicious spikes specifically on consumed pregen rows
(i.e., rows whose consumed_at is non-null), revert prep to HF.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
import time

# Quiet vLLM init logs.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

logger = logging.getLogger("prep_prompt_outcomes")


async def _peek_validator_ckpt(validator_url: str) -> tuple[int, str] | None:
    import httpx
    from reliquary.miner.submitter import get_window_state_v2
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            state = await get_window_state_v2(validator_url, client=client)
    except Exception:
        return None
    return state.checkpoint_n, (state.checkpoint_revision or "")


async def _peek_cooldown_prompts(validator_url: str) -> set[int]:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{validator_url.rstrip('/')}/state")
            r.raise_for_status()
            data = r.json()
    except Exception:
        return set()
    cd = data.get("cooldown_prompts") or []
    return set(int(x) for x in cd)


async def _resolve_checkpoint_path(default_repo: str) -> tuple[str, int, str]:
    """Return (local_path, ckpt_n, checkpoint_hash). Mirrors cli/main.py."""
    import httpx
    from huggingface_hub import snapshot_download

    validator_url = os.environ.get("RELIQUARY_VALIDATOR_URL", "").strip()
    if not validator_url:
        logger.warning("RELIQUARY_VALIDATOR_URL unset — cold-start ckpt_n=0")
        local_path = snapshot_download(repo_id=default_repo)
        return local_path, 0, ""

    from reliquary.miner.submitter import get_window_state_v2
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            state = await get_window_state_v2(validator_url, client=client)
    except Exception as e:
        logger.warning(
            "validator unreachable at %s (%s) — using cold-start %s",
            validator_url, type(e).__name__, default_repo,
        )
        local_path = snapshot_download(repo_id=default_repo)
        return local_path, 0, ""
    if not (state.checkpoint_repo_id and state.checkpoint_revision):
        logger.warning("validator has no published ckpt yet; using cold-start")
        local_path = snapshot_download(repo_id=default_repo)
        return local_path, 0, ""

    logger.info(
        "validator on ckpt_n=%d (%s@%s) — downloading",
        state.checkpoint_n, state.checkpoint_repo_id,
        state.checkpoint_revision[:12],
    )
    local_path = snapshot_download(
        repo_id=state.checkpoint_repo_id,
        revision=state.checkpoint_revision,
    )
    return local_path, state.checkpoint_n, state.checkpoint_revision


def _population_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return variance ** 0.5


async def main_async(num_prompts: int) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from reliquary.constants import (
        M_ROLLOUTS, MAX_TRUNCATED_PER_SUBMISSION, T_PROTO,
        TOP_K_PROTO, TOP_P_PROTO, MAX_NEW_TOKENS_PROTOCOL_CAP,
    )
    from reliquary.environment import load_environment
    from reliquary.miner.persistence import (
        PersistedBatch, PromptOutcome, cache_from_env, resolve_hotkey,
    )

    save_pregen = os.environ.get("PREP_SAVE_PREGEN", "1") == "1"

    hotkey = resolve_hotkey(os.environ.get("RELIQUARY_PREP_HOTKEY", "prep_worker"))
    cache = cache_from_env(miner_hotkey=hotkey)
    if not cache.enabled:
        print("ERROR: Supabase env not set", file=sys.stderr)
        sys.exit(1)

    default_repo = os.environ.get(
        "RELIQUARY_CHECKPOINT", "Qwen/Qwen3-4B-Instruct-2507",
    )
    local_path, ckpt_n, checkpoint_hash = await _resolve_checkpoint_path(default_repo)
    if not checkpoint_hash:
        checkpoint_hash = f"coldstart:{os.path.basename(local_path)}"
        logger.warning("no live ckpt; tagging rows with %s", checkpoint_hash)

    keep_n = int(os.environ.get("PREP_KEEP_LAST_CKPTS", "5"))
    purged = await asyncio.to_thread(
        cache.purge_old_checkpoints, checkpoint_hash, keep_last_n=keep_n,
    )
    if purged:
        logger.info("purged %d old ckpts", purged)

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85"))
    max_new_tokens = int(os.environ.get(
        "PREP_MAX_NEW_TOKENS",
        os.environ.get("RELIQUARY_MAX_NEW_TOKENS", "2048"),
    ))
    logger.info(
        "loading vllm engine from %s (max_model_len=%d, gpu_mem_util=%.2f, max_new_tokens=%d)",
        local_path, MAX_NEW_TOKENS_PROTOCOL_CAP, gpu_mem_util, max_new_tokens,
    )
    llm = LLM(
        model=local_path,
        dtype="bfloat16",
        max_model_len=MAX_NEW_TOKENS_PROTOCOL_CAP,
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=1,
        trust_remote_code=False,
    )

    env = load_environment("openmathinstruct")
    n_env = len(env)
    logger.info("env loaded: %d prompts", n_env)

    sigma_min = float(os.environ.get("RELIQUARY_SIGMA_MIN", "0.43"))
    existing = await asyncio.to_thread(cache.load_outcomes, checkpoint_hash)
    seen: set[int] = {o.prompt_idx for o in existing}
    n_good = sum(1 for o in existing if o.status == "good")
    n_bad = sum(1 for o in existing if o.status in ("dud", "oof"))
    logger.info(
        "hydrated ckpt=%s: %d good, %d bad already (seen=%d)",
        checkpoint_hash[:12], n_good, n_bad, len(seen),
    )

    validator_url = os.environ.get("RELIQUARY_VALIDATOR_URL", "").strip()
    if validator_url:
        cooldown_set = await _peek_cooldown_prompts(validator_url)
        if cooldown_set:
            before = len(seen)
            seen.update(cooldown_set)
            logger.info(
                "merged %d cooldown_prompts (was %d, now %d)",
                len(cooldown_set), before, len(seen),
            )

    rng = random.Random()
    top_k_val = -1 if TOP_K_PROTO <= 0 else TOP_K_PROTO  # vLLM: -1 disables

    priors: dict[int, int] = await asyncio.to_thread(cache.good_counts_across_ckpts)
    priority_queue = [idx for idx in priors.keys() if idx not in seen]
    rng.shuffle(priority_queue)
    logger.info(
        "cross-ckpt prior: %d structurally-good prompts, %d queued",
        len(priors), len(priority_queue),
    )

    STABLE_PRIOR_THRESHOLD = 5
    PROVEN_PRIOR_THRESHOLD = 1
    PROMPTS_PER_BATCH = int(os.environ.get("PREP_PROMPTS_PER_BATCH", "4"))
    # Check /state once per batch by default. The /state call is cheap
    # (~100ms) and ckpts can advance every ~5 min on this subnet; the old
    # default of 50 prompts could leave us generating on a stale ckpt for
    # 3+ min after an advance. Per-batch keeps detection latency under
    # one batch (~30-90s on vLLM).
    CKPT_POLL_EVERY = int(os.environ.get("PREP_CKPT_POLL_EVERY", str(PROMPTS_PER_BATCH)))
    min_terminated = M_ROLLOUTS - MAX_TRUNCATED_PER_SUBMISSION  # = 3

    def _next_prompt_idx() -> int | None:
        while priority_queue:
            cand = priority_queue.pop()
            if cand not in seen:
                return cand
        for _ in range(2000):
            cand = rng.randrange(n_env)
            if cand not in seen:
                return cand
        return None

    done = 0
    last_poll_done = 0
    target = num_prompts if num_prompts > 0 else None
    t0 = time.time()
    eos = tokenizer.eos_token_id

    while target is None or done < target:
        batch: list[dict] = []
        while len(batch) < PROMPTS_PER_BATCH:
            cand = _next_prompt_idx()
            if cand is None:
                break
            seen.add(cand)
            problem = env.get_problem(cand)
            prompt_tokens = tokenizer.encode(
                problem["prompt"], add_special_tokens=False,
            )
            prompt_len = len(prompt_tokens)
            if prompt_len + 256 > MAX_NEW_TOKENS_PROTOCOL_CAP:
                continue
            batch.append({
                "idx": cand,
                "problem": problem,
                "tokens": prompt_tokens,
                "prompt_len": prompt_len,
            })
        if not batch:
            logger.info("no more unseen prompts — done")
            return

        sps = [
            SamplingParams(
                n=M_ROLLOUTS,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=top_k_val,
                max_tokens=min(
                    max_new_tokens,
                    MAX_NEW_TOKENS_PROTOCOL_CAP - bp["prompt_len"],
                ),
            )
            for bp in batch
        ]
        inputs = [{"prompt_token_ids": bp["tokens"]} for bp in batch]
        gen_t0 = time.time()
        request_outputs = llm.generate(inputs, sps, use_tqdm=False)
        gen_elapsed = time.time() - gen_t0

        for bp, ro in zip(batch, request_outputs):
            problem = bp["problem"]
            rewards: list[float] = []
            trimmed_completions: list[list[int]] = []
            for comp in ro.outputs:
                gen = list(comp.token_ids)
                try:
                    first_eos = gen.index(eos)
                    gen = gen[: first_eos + 1]
                except ValueError:
                    pass
                trimmed_completions.append(gen)
                text = tokenizer.decode(gen)
                rewards.append(float(env.compute_reward(problem, text)))

            sigma = _population_std(rewards)
            k_correct = sum(1 for r in rewards if r > 0.5)
            n_terminated = sum(
                1 for ct in trimmed_completions if ct and ct[-1] == eos
            )
            if sigma >= sigma_min:
                status = "good"
            elif k_correct > 0:
                status = "dud"
            else:
                status = "oof"

            await asyncio.to_thread(
                cache.upsert_outcome,
                PromptOutcome(
                    prompt_idx=bp["idx"],
                    checkpoint_hash=checkpoint_hash,
                    k=k_correct,
                    sigma=sigma,
                    status=status,
                    miner_hotkey=hotkey,
                ),
            )

            pregen_saved = False
            if (
                save_pregen and status == "good"
                and n_terminated >= min_terminated
                and not checkpoint_hash.startswith("coldstart:")
            ):
                prior_count = priors.get(bp["idx"], 0)
                if prior_count >= STABLE_PRIOR_THRESHOLD:
                    tier = "stable"
                elif prior_count >= PROVEN_PRIOR_THRESHOLD:
                    tier = "proven"
                else:
                    tier = "exploratory"
                batch_rollouts = [
                    {
                        "tokens": bp["tokens"] + ct,
                        "prompt_length": bp["prompt_len"],
                        "reward": rewards[i],
                    }
                    for i, ct in enumerate(trimmed_completions)
                ]
                await asyncio.to_thread(
                    cache.save_batch,
                    PersistedBatch(
                        prompt_idx=bp["idx"],
                        checkpoint_hash=checkpoint_hash,
                        local_n=ckpt_n,
                        sigma=sigma,
                        k=k_correct,
                        rollouts=batch_rollouts,
                        miner_hotkey=hotkey,
                        tier=tier,
                    ),
                )
                pregen_saved = True

            done += 1
            rate = done / max(1.0, time.time() - t0)
            pregen_note = ""
            if status == "good" and save_pregen and not checkpoint_hash.startswith("coldstart:"):
                if pregen_saved:
                    pregen_note = " pregen=saved"
                else:
                    pregen_note = f" pregen=skip(truncated={M_ROLLOUTS - n_terminated}/8)"
            logger.info(
                "[%d%s] prompt=%d k=%d/%d sigma=%.3f status=%s%s "
                "batch=%d/%.1fs (%.1f prompts/min)",
                done, f"/{target}" if target else "",
                bp["idx"], k_correct, M_ROLLOUTS, sigma, status, pregen_note,
                len(batch), gen_elapsed, rate * 60,
            )

        if validator_url and (done - last_poll_done) >= CKPT_POLL_EVERY:
            last_poll_done = done
            peek = await _peek_validator_ckpt(validator_url)
            if peek is not None and peek[1] and peek[1] != checkpoint_hash:
                logger.info(
                    "ckpt advanced %s -> %s (ckpt_n %d -> %d); exiting for respawn",
                    checkpoint_hash[:12], peek[1][:12], ckpt_n, peek[0],
                )
                sys.exit(0)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-prompts", type=int, default=500, help="0 = run forever.")
    args = p.parse_args()
    try:
        asyncio.run(main_async(args.num_prompts))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)


if __name__ == "__main__":
    main()
