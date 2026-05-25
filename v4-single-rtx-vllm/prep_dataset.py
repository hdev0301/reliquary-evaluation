#!/usr/bin/env python3
"""Standalone prep job — generate prompt outcomes + pregen batches and
write them to Supabase for the miner to consume.

Runs the same model + env the miner uses, but skips everything to do
with the validator / wallet / window state / GRAIL sketches. The output
table layout matches what ``reliquary.miner.persistence`` expects, so
the miner hydrates from it transparently on its next ckpt advance.

Designed for:
  * **Cold start** — pre-populate Supabase before launching the miner.
  * **Off-box generation** — run on a second machine to widen the
    in-zone prompt pool without stealing the live miner's GPU time.
  * **Checkpoint warmup** — when the validator publishes a new ckpt,
    run this against the new ckpt to seed batches before the miner
    has a chance to enqueue them itself.

NOT designed for:
  * **Same-GPU concurrent runs with the live miner** — they will share
    the device and each will run at roughly half throughput. Either
    pause the miner or run this on a different GPU/host.

Env (same .env the miner uses):
  RELIQUARY_VALIDATOR_URL       validator base URL (queried for current ckpt)
  RELIQUARY_SUPABASE_URL        Supabase project URL
  RELIQUARY_SUPABASE_KEY        Supabase service_role key
  RELIQUARY_MAX_NEW_TOKENS      full-gen cap (default 8192)
  RELIQUARY_PRESCREEN_MAX_TOKENS prescreen cap (default 1024)
  RELIQUARY_PRESCREEN_ROLLOUTS  prescreen rollout count (default 8)
  RELIQUARY_GEN_BATCH_PROMPTS   prompts per .generate() call (default 2)

The script queries the validator's ``/state`` to learn the currently
published checkpoint (``checkpoint_repo_id`` + ``checkpoint_revision``)
and loads that exact revision. The miner uses ``checkpoint_revision``
verbatim as its ``_latest_local_hash`` — the prep script does the same,
so cache rows hydrate cleanly into the live miner on its next ckpt
advance.

Usage:
    cd /root/reliquary && source scripts/.env
    python scripts/prep_dataset.py --cuda 0 --total 1000

  --total N         : stop after N prompts considered (default: forever)
  --cuda IDX        : CUDA device index (default 0)
  --hotkey HK       : ss58 to stamp rows (default from env)
  --repo-id ID      : override validator-published repo (advanced)
  --revision REV    : override validator-published revision (advanced)
  --validator-url U : override RELIQUARY_VALIDATOR_URL
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import urllib.request
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make `reliquary` importable when the script is run from anywhere.
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

from reliquary.environment import load_environment
from reliquary.miner.persistence import (
    PersistedBatch,
    PromptOutcome,
    cache_from_env,
    resolve_hotkey,
)
from reliquary.constants import (
    BINARY_REWARD_MAX_CORRECT,
    BINARY_REWARD_MIN_CORRECT,
    BOOTSTRAP_SIGMA_MIN,
    M_ROLLOUTS,
    SIGMA_MIN,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("prep_dataset")


def _fetch_validator_ckpt(validator_url: str) -> tuple[str, str, int]:
    """Query validator /state for (repo_id, revision, checkpoint_n).
    The miner sets _latest_local_hash = checkpoint_revision verbatim,
    so the prep script must use the same string as its ckpt_hash key
    for hydration to match.
    """
    url = validator_url.rstrip("/") + "/state"
    with urllib.request.urlopen(url, timeout=15) as r:
        st = json.loads(r.read())
    repo_id = st.get("checkpoint_repo_id")
    revision = st.get("checkpoint_revision")
    n = int(st.get("checkpoint_n", 0))
    if not repo_id or not revision:
        raise RuntimeError(
            f"validator /state has no published checkpoint yet "
            f"(repo_id={repo_id!r} revision={revision!r})"
        )
    return repo_id, revision, n


def _resolve_eos_set(tokenizer, model) -> set[int]:
    """Build the full eos id set the miner trims against. Same logic as
    the post-fix `_generate_full` so prep-script batches end at the
    same token positions the validator's `has_eos_padding` accepts."""
    gen_cfg = getattr(model, "generation_config", None)
    eos_ids = getattr(gen_cfg, "eos_token_id", None) if gen_cfg else None
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        return {eos_ids}
    if eos_ids is None:
        return set()
    return {int(e) for e in eos_ids if e is not None}


def _trim_at_first_eos(tokens: list[int], eos_set: set[int]) -> list[int]:
    if not eos_set:
        return tokens
    for i, t in enumerate(tokens):
        if int(t) in eos_set:
            return tokens[: i + 1]
    return tokens


def _generate_batch(
    model, tokenizer, problems: list[dict], n_rollouts: int,
    max_new_tokens: int, eos_set: set[int], device: str,
) -> list[list[dict]]:
    """Mirror of the miner's `_generate_rollouts_multi_prompt`.

    Returns a list of length len(problems), each entry a list of
    n_rollouts dicts with keys ``tokens`` (prompt+completion ids) and
    ``prompt_length``. Trims at first EOS so trailing pad EOS doesn't
    leak into persisted batches.
    """
    prompts = [p["prompt"] for p in problems]
    # Apply chat template so the model sees the same assistant prefix
    # the validator + live miner expect. The OpenMathInstruct env builds
    # bare-prompt strings; the model is an Instruct checkpoint and
    # behaves badly on raw prompts.
    formatted = []
    for prompt_text in prompts:
        try:
            formatted.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            # Tokenizer without chat template — fall back to raw prompt.
            formatted.append(prompt_text)

    enc = tokenizer(formatted, return_tensors="pt", padding=True).to(device)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    K = input_ids.shape[0]
    max_prompt_len = input_ids.shape[1]
    # Replicate each prompt n_rollouts times so a single batched generate
    # produces all rollouts in one forward pass.
    input_ids = input_ids.repeat_interleave(n_rollouts, dim=0)
    attn = attn.repeat_interleave(n_rollouts, dim=0)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            top_k=TOP_K_PROTO,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_lengths = attn.sum(dim=1)[::n_rollouts].tolist()
    results: list[list[dict]] = [[] for _ in range(K)]
    for k in range(K):
        prompt_length = int(prompt_lengths[k])
        start_idx = max_prompt_len - prompt_length
        for r in range(n_rollouts):
            row_idx = k * n_rollouts + r
            full_seq = outputs[row_idx].tolist()
            real_seq = full_seq[start_idx:]
            prompt_part = real_seq[:prompt_length]
            gen_part = real_seq[prompt_length:]
            gen_part = _trim_at_first_eos(gen_part, eos_set)
            results[k].append({
                "tokens": prompt_part + gen_part,
                "prompt_length": prompt_length,
            })
    return results


def _reward(env, problem: dict, gen_tokens: list[int], tokenizer) -> float:
    text = tokenizer.decode(gen_tokens)
    try:
        return float(env.compute_reward(problem, text))
    except Exception:
        return 0.0


def _pick_random_prompt_idx(env) -> int:
    """OpenMathInstruct exposes a numeric index space; pick uniformly.
    Tries env.dataset_size first, then falls back to a 1M range.
    """
    size = getattr(env, "dataset_size", None)
    if not isinstance(size, int) or size <= 0:
        size = 1_000_000
    return random.randint(0, size - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=None,
                    help="stop after N prompts considered (default: forever)")
    ap.add_argument("--cuda", type=int, default=0,
                    help="CUDA device index (default 0)")
    ap.add_argument("--hotkey", type=str, default=None,
                    help="ss58 to stamp rows (default from env)")
    ap.add_argument("--max-new-tokens", type=int,
                    default=int(os.environ.get("RELIQUARY_MAX_NEW_TOKENS", "8192")))
    ap.add_argument("--prescreen-rollouts", type=int,
                    default=int(os.environ.get("RELIQUARY_PRESCREEN_ROLLOUTS", "8")))
    ap.add_argument("--prescreen-max-tokens", type=int,
                    default=int(os.environ.get("RELIQUARY_PRESCREEN_MAX_TOKENS", "1024")))
    ap.add_argument("--batch-prompts", type=int,
                    default=int(os.environ.get("RELIQUARY_GEN_BATCH_PROMPTS", "2")))
    ap.add_argument("--environment", type=str, default="openmathinstruct")
    ap.add_argument("--repo-id", type=str, default=None,
                    help="override validator-published HF repo id")
    ap.add_argument("--revision", type=str, default=None,
                    help="override validator-published HF revision")
    ap.add_argument("--validator-url", type=str,
                    default=os.environ.get("RELIQUARY_VALIDATOR_URL", ""))
    args = ap.parse_args()

    # Resolve the checkpoint the validator is currently scoring against.
    # Hand-overrides win; otherwise query the validator. The revision
    # string becomes the ckpt_hash (must match what the live miner sets
    # _latest_local_hash to from its own /state poll, otherwise the
    # miner can't hydrate from our writes).
    if args.repo_id and args.revision:
        repo_id, revision = args.repo_id, args.revision
        ckpt_n = -1
        logger.info("using --repo-id/--revision override")
    else:
        if not args.validator_url:
            logger.error(
                "no --validator-url and no RELIQUARY_VALIDATOR_URL — "
                "pass --repo-id and --revision to skip the validator query"
            )
            return 2
        try:
            repo_id, revision, ckpt_n = _fetch_validator_ckpt(args.validator_url)
        except Exception as e:
            logger.error("validator /state fetch failed: %s", e)
            return 2
        logger.info(
            "validator publishes repo=%s revision=%s ckpt_n=%d",
            repo_id, revision, ckpt_n,
        )

    device = f"cuda:{args.cuda}"
    logger.info(
        "loading repo=%s rev=%s device=%s max_new=%d prescreen=%dx%d batch_prompts=%d",
        repo_id, revision, device, args.max_new_tokens,
        args.prescreen_rollouts, args.prescreen_max_tokens, args.batch_prompts,
    )
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=os.environ.get("RELIQUARY_ATTN_IMPL", "sdpa"),
    ).to(device).eval()
    eos_set = _resolve_eos_set(tokenizer, model)
    logger.info("eos_set=%s", sorted(eos_set))

    env = load_environment(args.environment)
    resolved = resolve_hotkey(args.hotkey)
    if resolved != args.hotkey:
        logger.info("resolved hotkey %r -> %s", args.hotkey, resolved)
    args.hotkey = resolved
    cache = cache_from_env(miner_hotkey=args.hotkey)
    if not cache.enabled:
        logger.error("Supabase cache disabled — check RELIQUARY_SUPABASE_URL/KEY")
        return 2

    # Must match what the miner sets _latest_local_hash to: bare revision string.
    ckpt_hash = revision
    logger.info("ckpt_hash=%s", ckpt_hash)

    # Hydrate the set of prompts we've already classified so we don't
    # re-pay prescreen on them.
    already = {o.prompt_idx for o in cache.load_outcomes(ckpt_hash)}
    logger.info("loaded %d existing outcomes for this ckpt", len(already))

    bootstrap = False  # prep job always uses steady-state thresholds
    sigma_min = BOOTSTRAP_SIGMA_MIN if bootstrap else SIGMA_MIN

    considered = 0
    good = 0
    duds = 0
    oofs = 0
    t0 = time.time()
    while args.total is None or considered < args.total:
        # Build a batch of fresh-to-us prompts.
        problems: list[dict] = []
        pids: list[int] = []
        attempts = 0
        while len(problems) < args.batch_prompts and attempts < 50:
            attempts += 1
            pid = _pick_random_prompt_idx(env)
            if pid in already:
                continue
            try:
                problem = env.get_problem(pid)
            except Exception:
                continue
            if problem is None:
                continue
            problems.append(problem)
            pids.append(pid)
        if not problems:
            logger.info("no fresh prompts found in 50 tries, sleeping 30s")
            time.sleep(30)
            continue

        # Prescreen at the short cap.
        t_pre = time.time()
        try:
            ps_gens_per = _generate_batch(
                model, tokenizer, problems, args.prescreen_rollouts,
                args.prescreen_max_tokens, eos_set, device,
            )
        except Exception:
            logger.exception("prescreen failed; skipping batch")
            continue
        pre_secs = time.time() - t_pre

        survivors: list[int] = []
        for slot, pid in enumerate(pids):
            considered += 1
            already.add(pid)
            short = ps_gens_per[slot]
            rewards_s: list[float] = []
            for g in short:
                rewards_s.append(
                    _reward(env, problems[slot],
                            g["tokens"][g["prompt_length"]:], tokenizer)
                )
            k_s = sum(1 for r in rewards_s if r >= 0.5)
            n_s = len(rewards_s)
            avg_len = int(sum(
                len(g["tokens"]) - g["prompt_length"] for g in short
            ) / max(1, n_s))
            wrong_trunc = sum(
                1 for j, r in enumerate(rewards_s)
                if r < 0.5
                and (len(short[j]["tokens"]) - short[j]["prompt_length"])
                >= args.prescreen_max_tokens - 4
            )
            if k_s == 0 or k_s == n_s:
                # 0/n or n/n at prescreen → almost certainly OOZ at full gen
                cache.upsert_outcome(PromptOutcome(
                    prompt_idx=pid, checkpoint_hash=ckpt_hash,
                    k=k_s, sigma=0.0,
                    status="dud" if k_s == 0 else "oof",
                    avg_completion_len=avg_len, truncated_count=wrong_trunc,
                    miner_hotkey=args.hotkey,
                ))
                duds += 1
                logger.info(
                    "prescreen dud pid=%d k=%d/%d avg_len=%d pre=%.1fs",
                    pid, k_s, n_s, avg_len, pre_secs,
                )
                continue
            # All wrongs truncated at prescreen → almost always 8/8 at full → OOZ.
            wrong_indices = [j for j, r in enumerate(rewards_s) if r < 0.5]
            if (
                len(wrong_indices) >= 2
                and wrong_trunc == len(wrong_indices)
                and k_s >= 1
            ):
                cache.upsert_outcome(PromptOutcome(
                    prompt_idx=pid, checkpoint_hash=ckpt_hash,
                    k=k_s, sigma=0.0, status="dud",
                    avg_completion_len=avg_len, truncated_count=wrong_trunc,
                    miner_hotkey=args.hotkey,
                ))
                duds += 1
                logger.info(
                    "prescreen all-wrong-trunc pid=%d k=%d/%d pre=%.1fs → likely full-gen OOZ",
                    pid, k_s, n_s, pre_secs,
                )
                continue
            survivors.append(slot)
            logger.info(
                "prescreen pass pid=%d k=%d/%d avg_len=%d — full gen next",
                pid, k_s, n_s, avg_len,
            )

        if not survivors:
            continue

        # Full gen on survivors.
        sub_problems = [problems[s] for s in survivors]
        sub_pids = [pids[s] for s in survivors]
        t_full = time.time()
        try:
            gens_per = _generate_batch(
                model, tokenizer, sub_problems, M_ROLLOUTS,
                args.max_new_tokens, eos_set, device,
            )
        except Exception:
            logger.exception("full gen failed; skipping batch")
            continue
        full_secs = time.time() - t_full

        for slot, pid in enumerate(sub_pids):
            problem = sub_problems[slot]
            gens = gens_per[slot]
            rewards: list[float] = []
            for g in gens:
                rewards.append(_reward(env, problem,
                                       g["tokens"][g["prompt_length"]:],
                                       tokenizer))
            k = sum(1 for r in rewards if r >= 0.5)
            # population std (same as miner's _population_std):
            mean = sum(rewards) / max(1, len(rewards))
            var = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards))
            sigma = var ** 0.5

            if sigma < sigma_min:
                cache.upsert_outcome(PromptOutcome(
                    prompt_idx=pid, checkpoint_hash=ckpt_hash,
                    k=k, sigma=sigma, status="oof",
                    miner_hotkey=args.hotkey,
                ))
                oofs += 1
                logger.info(
                    "full OOZ pid=%d k=%d/%d sigma=%.3f gen=%.1fs",
                    pid, k, M_ROLLOUTS, sigma, full_secs,
                )
                continue
            if not (BINARY_REWARD_MIN_CORRECT <= k <= BINARY_REWARD_MAX_CORRECT):
                # Binary frontier band gate — outside [3,5] would be
                # rejected by validator as reward_distribution.
                cache.upsert_outcome(PromptOutcome(
                    prompt_idx=pid, checkpoint_hash=ckpt_hash,
                    k=k, sigma=sigma, status="oof",
                    miner_hotkey=args.hotkey,
                ))
                oofs += 1
                logger.info(
                    "full outside-frontier pid=%d k=%d/%d sigma=%.3f gen=%.1fs",
                    pid, k, M_ROLLOUTS, sigma, full_secs,
                )
                continue

            # Good — persist outcome + batch.
            cache.upsert_outcome(PromptOutcome(
                prompt_idx=pid, checkpoint_hash=ckpt_hash,
                k=k, sigma=sigma, status="good",
                miner_hotkey=args.hotkey,
            ))
            rollouts_payload = []
            for g, r in zip(gens, rewards):
                rollouts_payload.append({
                    "tokens": [int(t) for t in g["tokens"]],
                    "prompt_length": int(g["prompt_length"]),
                    "reward": float(r),
                })
            # local_n=0: prep script doesn't track ckpt revisions like the
            # miner does. Miner uses this only for stale-ckpt eviction; the
            # ckpt_hash gate already covers that, so 0 is safe.
            cache.save_batch(PersistedBatch(
                prompt_idx=pid, checkpoint_hash=ckpt_hash,
                local_n=0, sigma=sigma, k=k,
                rollouts=rollouts_payload,
                miner_hotkey=args.hotkey,
            ))
            good += 1
            logger.info(
                "GOOD pid=%d k=%d/%d sigma=%.3f gen=%.1fs",
                pid, k, M_ROLLOUTS, sigma, full_secs,
            )

        elapsed = time.time() - t0
        logger.info(
            "totals: considered=%d good=%d duds=%d oofs=%d elapsed=%.0fs",
            considered, good, duds, oofs, elapsed,
        )

    logger.info("done. considered=%d good=%d duds=%d oofs=%d",
                considered, good, duds, oofs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
