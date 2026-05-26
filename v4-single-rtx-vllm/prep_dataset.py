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
  RELIQUARY_PRESCREEN_ROLLOUTS  prescreen rollout count (default 8 — matches
                                M_ROLLOUTS so k_short maps 1:1 to k_full)
  RELIQUARY_GEN_BATCH_PROMPTS   prompts per .generate() call (default 2)
  RELIQUARY_OVERGEN_K           full-gen rollout count (default 12); cherry-
                                picks M_ROLLOUTS=8 in-band. Set to 8 to
                                disable over-generation.
  RELIQUARY_GOOD_TS_SINCE_ISO   drop prompt_outcomes rows with last_seen<this
                                ISO timestamp from cross-ckpt stats. Set after
                                a ckpt reset to ignore poisoned pre-reset
                                history (e.g. 2026-05-26T03:00:00Z).
  RELIQUARY_EXCLUDE_CKPTS       comma-separated ckpt revisions to drop from
                                cross-ckpt stats (surgical alternative to
                                RELIQUARY_GOOD_TS_SINCE_ISO).

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
import datetime
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
from reliquary.validator.dedup import compute_rollout_hash
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

    Tokenization must match the protocol exactly: raw
    ``tokenizer.encode(prompt, add_special_tokens=False)`` with no
    chat template — that is what the validator's
    ``_canonical_prompt_tokens`` and the live miner's
    ``_generate_rollouts_multi_prompt`` both produce. Applying a chat
    template here makes the persisted prompt prefix differ from what
    the validator expects → PROMPT_MISMATCH on submission.

    Padding must also be LEFT (causal-LM convention). Letting HF's
    ``tokenizer(..., padding=True)`` default to right-padding (Qwen
    default) and then slicing as if it were left-padded silently
    corrupts the prompt prefix for every non-longest prompt in the
    batch. We build the padded tensor by hand to make the alignment
    explicit and match the miner's path.
    """
    K = len(problems)
    if K == 0:
        return []

    prompt_token_lists = [
        tokenizer.encode(p["prompt"], add_special_tokens=False)
        for p in problems
    ]
    prompt_lengths = [len(t) for t in prompt_token_lists]
    max_prompt_len = max(prompt_lengths)
    pad_id = tokenizer.pad_token_id

    flat_input: list[list[int]] = []
    flat_mask: list[list[int]] = []
    for tokens, length in zip(prompt_token_lists, prompt_lengths):
        pad_count = max_prompt_len - length
        padded = [pad_id] * pad_count + tokens
        mask = [0] * pad_count + [1] * length
        for _ in range(n_rollouts):
            flat_input.append(padded)
            flat_mask.append(mask)

    input_ids = torch.tensor(flat_input, device=device)
    attn = torch.tensor(flat_mask, device=device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            top_k=TOP_K_PROTO,
            pad_token_id=pad_id,
        )

    results: list[list[dict]] = [[] for _ in range(K)]
    for k in range(K):
        prompt_length = prompt_lengths[k]
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
    ap.add_argument("--skip-prescreen-threshold", type=float,
                    default=float(os.environ.get("RELIQUARY_SKIP_PRESCREEN_THRESHOLD", "5.0")),
                    help="priority score above which prep skips prescreen "
                         "and routes the prompt straight to full-gen")
    ap.add_argument("--prescreen-max-tokens", type=int,
                    default=int(os.environ.get("RELIQUARY_PRESCREEN_MAX_TOKENS", "2048")))
    ap.add_argument("--batch-prompts", type=int,
                    default=int(os.environ.get("RELIQUARY_GEN_BATCH_PROMPTS", "2")))
    ap.add_argument("--overgen-k", type=int,
                    default=int(os.environ.get("RELIQUARY_OVERGEN_K", "12")),
                    help="full-gen rollout count to sample before "
                         "cherry-picking M_ROLLOUTS. Set to M_ROLLOUTS (8) "
                         "to disable. Higher → more chances to land an "
                         "in-band subset on prompts where the natural k "
                         "would be outside [3,5].")
    ap.add_argument("--environment", type=str, default="openmathinstruct")
    ap.add_argument("--repo-id", type=str, default=None,
                    help="override validator-published HF repo id")
    ap.add_argument("--revision", type=str, default=None,
                    help="override validator-published HF revision")
    ap.add_argument("--validator-url", type=str,
                    default=os.environ.get("RELIQUARY_VALIDATOR_URL", ""))
    ap.add_argument("--good-ts-since-iso", type=str,
                    default=os.environ.get("RELIQUARY_GOOD_TS_SINCE_ISO") or None,
                    help="drop cross-ckpt rows with last_seen<this ISO ts "
                         "(use after a ckpt reset to ignore poisoned history)")
    ap.add_argument("--exclude-ckpts", type=str,
                    default=os.environ.get("RELIQUARY_EXCLUDE_CKPTS", ""),
                    help="comma-separated ckpt revisions to drop from "
                         "cross-ckpt stats (surgical poison-ckpt exclusion)")
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

    # Cross-ckpt priority queue (graded + recency-weighted): score each
    # prompt by its history across all ckpts in Supabase. Recent good
    # observations count more than old ones; recent dud/oof penalize more.
    excluded_ckpts: set[str] = {
        s.strip() for s in (args.exclude_ckpts or "").split(",") if s.strip()
    }
    if args.good_ts_since_iso or excluded_ckpts:
        logger.info(
            "cross-ckpt filter: since_iso=%s exclude_ckpts=%d",
            args.good_ts_since_iso or "(none)", len(excluded_ckpts),
        )
    xc_stats = cache.prompt_stats_across_ckpts(
        since_iso=args.good_ts_since_iso,
        exclude_ckpts=excluded_ckpts or None,
    )

    band_centre = (BINARY_REWARD_MIN_CORRECT + BINARY_REWARD_MAX_CORRECT) / 2
    HALF_LIFE_SECS = 3 * 3600  # 3 hours: an observation older than this
    # contributes ~half as much as a fresh one.
    _now_ts = time.time()

    def _parse_iso_ts(ts_str: str) -> float:
        try:
            dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return _now_ts - HALF_LIFE_SECS * 10  # treat unparseable as very old

    def _decayed_count(ts_list: list[str]) -> float:
        return sum(
            0.5 ** ((_now_ts - _parse_iso_ts(ts)) / HALF_LIFE_SECS)
            for ts in ts_list
        )

    def _priority_score(s: dict) -> float:
        good_w = _decayed_count(s.get("good_ts", []))
        bad_w = _decayed_count(s.get("bad_ts", []))
        return (
            good_w
            - 0.5 * bad_w
            - 0.25 * (s["mean_k"] - band_centre) ** 2
        )

    scored = [
        (pid, _priority_score(s))
        for pid, s in xc_stats.items()
        if pid not in already and _priority_score(s) > 0
    ]
    scored.sort(key=lambda t: -t[1])
    priority_pids: list[int] = [pid for pid, _ in scored]

    # Structural dud blacklist: prompts that have been dud or oof multiple
    # times across ckpts and have never landed good are almost always
    # too-easy or too-hard for this base model on this env, regardless of
    # which fine-tuned ckpt is live. Skip these in random sampling so we
    # don't keep paying prescreen on hopeless candidates.
    structural_duds: set[int] = {
        pid for pid, s in xc_stats.items()
        if s["good"] == 0 and (s["dud"] + s["oof"]) >= 3
    }
    logger.info(
        "structural dud blacklist: %d prompts (dud+oof≥3, good=0)",
        len(structural_duds),
    )

    # Trusted prompts (skip prescreen, route straight to full gen) admit
    # two strictly-stronger-than-prescreen signals:
    #
    #   (a) Any pid with a confirmed `good` outcome on the CURRENT ckpt.
    #       Validator-accepted evidence on the live model — the strongest
    #       possible prior, far stronger than any cross-ckpt aggregate.
    #       Drop the cross-ckpt score threshold entirely for these.
    #   (b) pids whose cross-ckpt score ≥ threshold AND that have at
    #       least one same-ckpt good (the same-ckpt gate protects against
    #       poisoned-chain priors after a reset).
    #
    # The two sets are identical in practice today (every pid in (b) is
    # also in (a)) but the structure makes the intent explicit and keeps
    # (a) safe to extend later (e.g., admit scrape_intel hot pids without
    # routing through fresh_good_pids).
    same_ckpt_goods: set[int] = {
        pid for pid, _ts in cache.fresh_good_pids(ckpt_hash)
    }
    trusted_pids: set[int] = set(same_ckpt_goods) | {
        pid for pid, score in scored
        if score >= args.skip_prescreen_threshold and pid in same_ckpt_goods
    }
    # Filter out already-classified pids — no point routing a pid we
    # already have an outcome for through the full-gen path.
    trusted_pids -= already
    logger.info(
        "trusted prompts (skip-prescreen): %d "
        "(same-ckpt-goods=%d, cross-ckpt threshold=%.1f)",
        len(trusted_pids), len(same_ckpt_goods), args.skip_prescreen_threshold,
    )

    if priority_pids:
        top_pid = priority_pids[0]
        top_stat = xc_stats[top_pid]
        logger.info(
            "priority queue primed: %d candidates "
            "(top pid=%d good=%d dud=%d oof=%d mean_k=%.2f score=%.2f)",
            len(priority_pids), top_pid, top_stat["good"], top_stat["dud"],
            top_stat["oof"], top_stat["mean_k"], scored[0][1],
        )
    else:
        logger.info("priority queue empty — falling back to random sampling")

    bootstrap = False  # prep job always uses steady-state thresholds
    sigma_min = BOOTSTRAP_SIGMA_MIN if bootstrap else SIGMA_MIN

    considered = 0
    good = 0
    duds = 0
    oofs = 0
    t0 = time.time()
    # Live refresh: poll Supabase periodically for fresh `good` rows other
    # miners (or our own scrape_intel) have just written under the live
    # ckpt. Those are the strongest possible priors for the current
    # checkpoint and should jump to the head of the queue.
    REFRESH_EVERY_BATCHES = 10
    batches_done = 0
    refresh_since_iso: str | None = None
    while args.total is None or considered < args.total:
        if batches_done > 0 and batches_done % REFRESH_EVERY_BATCHES == 0:
            fresh = cache.fresh_good_pids(ckpt_hash, since_iso=refresh_since_iso)
            new_pids = [
                pid for pid, _ts in fresh
                if pid not in already and pid not in structural_duds
            ]
            if new_pids:
                # Dedupe vs already-queued and prepend.
                queued = set(priority_pids)
                fresh_new = [p for p in new_pids if p not in queued]
                priority_pids = fresh_new + priority_pids
                # Move the new pids into the trusted set too — they have
                # validator-accepted evidence on THIS ckpt, the strongest
                # possible prior; treat them as skip-prescreen.
                trusted_pids.update(fresh_new)
                refresh_since_iso = max(ts for _pid, ts in fresh)
                logger.info(
                    "live refresh: +%d fresh good pids on current ckpt "
                    "(queue=%d trusted=%d since=%s)",
                    len(fresh_new), len(priority_pids), len(trusted_pids),
                    refresh_since_iso,
                )
        batches_done += 1
        # Build a batch of fresh-to-us prompts. Batch is homogeneous:
        # either all "trusted" (skip prescreen) or all "screen" (need
        # prescreen). The very first pid decides the mode.
        problems: list[dict] = []
        pids: list[int] = []
        attempts = 0
        batch_mode: str | None = None
        while len(problems) < args.batch_prompts and attempts < 50:
            attempts += 1
            if priority_pids:
                pid = priority_pids.pop(0)
                from_priority = True
            else:
                pid = _pick_random_prompt_idx(env)
                from_priority = False
            if pid in already or pid in structural_duds:
                continue
            mode = "trusted" if pid in trusted_pids else "screen"
            if batch_mode is None:
                batch_mode = mode
            elif mode != batch_mode:
                # mode mismatch — push back if it came from the queue and
                # finalize this batch with what we already have.
                if from_priority:
                    priority_pids.insert(0, pid)
                break
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

        # Prescreen at the short cap — or skip entirely for trusted pids.
        survivors: list[int] = []
        if batch_mode == "trusted":
            pre_secs = 0.0
            ps_gens_per = [[] for _ in pids]  # not used
            for slot, pid in enumerate(pids):
                considered += 1
                already.add(pid)
                survivors.append(slot)
            logger.info(
                "trusted batch (skip-prescreen): n=%d pids=%s",
                len(pids), pids,
            )
        else:
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
        if batch_mode == "screen":
            classification_pids = list(enumerate(pids))
        else:
            classification_pids = []
        for slot, pid in classification_pids:
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
            # Gate purely on k_s ∈ [BINARY_REWARD_MIN_CORRECT,
            # BINARY_REWARD_MAX_CORRECT]. The previous `est_full_k =
            # k_s + wrong_trunc` rescue was empirically over-optimistic:
            # every rescued prompt in recent logs (pid 618082, 339561,
            # 967856, 550072) went OOZ at full gen, costing ~200s each.
            # And the avg_len upper-bound check was rejecting prompts with
            # k_s already in band just because rollouts hit the prescreen
            # cap — those are exactly the ones worth full-gen'ing.
            #
            # The prescreen_max_tokens default was also bumped to 2048
            # for this gate to be meaningful — at 1024 the model bails on
            # too many problems mid-chain and k_s underestimates k_full
            # systematically.
            if not (
                BINARY_REWARD_MIN_CORRECT <= k_s <= BINARY_REWARD_MAX_CORRECT
            ):
                # k_s == n_s ("oof": model saturated) is distinct from
                # k_s == 0 / 1 / 2 / 6 / 7 ("dud": off-band). The
                # distinction feeds back into priority scoring + the
                # structural_duds blacklist heuristic.
                status = "oof" if k_s == n_s else "dud"
                cache.upsert_outcome(PromptOutcome(
                    prompt_idx=pid, checkpoint_hash=ckpt_hash,
                    k=k_s, sigma=0.0, status=status,
                    avg_completion_len=avg_len, truncated_count=wrong_trunc,
                    miner_hotkey=args.hotkey,
                ))
                duds += 1
                logger.info(
                    "prescreen %s pid=%d k=%d/%d avg_len=%d trunc=%d "
                    "outside [%d,%d] pre=%.1fs",
                    status, pid, k_s, n_s, avg_len, wrong_trunc,
                    BINARY_REWARD_MIN_CORRECT, BINARY_REWARD_MAX_CORRECT,
                    pre_secs,
                )
                continue
            survivors.append(slot)
            logger.info(
                "prescreen pass pid=%d k=%d/%d avg_len=%d trunc=%d — full gen next",
                pid, k_s, n_s, avg_len, wrong_trunc,
            )

        if not survivors:
            continue

        # Full gen on survivors — generate `overgen_k` rollouts per prompt
        # and cherry-pick M_ROLLOUTS below to maximise the chance that the
        # submitted subset lands in the validator's [3,5] band.
        sub_problems = [problems[s] for s in survivors]
        sub_pids = [pids[s] for s in survivors]
        t_full = time.time()
        try:
            gens_per = _generate_batch(
                model, tokenizer, sub_problems, args.overgen_k,
                args.max_new_tokens, eos_set, device,
            )
        except Exception:
            logger.exception("full gen failed; skipping batch")
            continue
        full_secs = time.time() - t_full

        for slot, pid in enumerate(sub_pids):
            problem = sub_problems[slot]
            gens_all = gens_per[slot]
            rewards_all: list[float] = []
            for g in gens_all:
                rewards_all.append(_reward(env, problem,
                                           g["tokens"][g["prompt_length"]:],
                                           tokenizer))

            # Pre-flight HASH_DUPLICATE check. Any rollout whose sha256
            # is already in the validator's accepted set for this (pid,
            # ckpt) would reject as HASH_DUPLICATE at submit time. Drop
            # those from the cherry-pick pool BEFORE picking 8, so we
            # don't waste a submission slot on a guaranteed-loss rollout.
            # The hash function must match the validator's exactly —
            # ``compute_rollout_hash`` from validator/dedup.py is the
            # canonical impl.
            accepted_set = cache.accepted_hashes_for_prompt(pid, ckpt_hash)
            keep_idx: list[int] = []
            dropped_hashes = 0
            for i, g in enumerate(gens_all):
                try:
                    h_hex = compute_rollout_hash(g["tokens"]).hex()
                except Exception:
                    keep_idx.append(i)  # hashing failure → trust it; let
                    continue            # the validator decide
                if h_hex in accepted_set:
                    dropped_hashes += 1
                    continue
                keep_idx.append(i)
            right_idx = [i for i in keep_idx if rewards_all[i] >= 0.5]
            wrong_idx = [i for i in keep_idx if rewards_all[i] < 0.5]
            gen_k = len(right_idx)
            n_wrong = len(wrong_idx)

            # Cherry-pick a subset of M_ROLLOUTS landing k ∈
            # [BINARY_REWARD_MIN_CORRECT, BINARY_REWARD_MAX_CORRECT].
            # Target k=4 (band center → max σ). Rewards in this env are
            # binary {0,1} so σ for the picked subset is deterministic in
            # k_pick: σ = sqrt(k(M-k))/M.
            k_lo = max(BINARY_REWARD_MIN_CORRECT, M_ROLLOUTS - n_wrong)
            k_hi = min(BINARY_REWARD_MAX_CORRECT, gen_k)
            if k_lo > k_hi:
                # Infeasible — either <3 right or <3 wrong in the
                # hash-filtered pool, so no in-band subset of size M
                # exists. Persist the raw (pre-filter) gen_k so the
                # priority queue learns the prompt's difficulty signal
                # correctly (gen_k=overgen_k means "too easy",
                # gen_k=0 means "too hard").
                raw_gen_k = sum(1 for r in rewards_all if r >= 0.5)
                cache.upsert_outcome(PromptOutcome(
                    prompt_idx=pid, checkpoint_hash=ckpt_hash,
                    k=raw_gen_k, sigma=0.0, status="oof",
                    miner_hotkey=args.hotkey,
                ))
                oofs += 1
                logger.info(
                    "full overgen-infeasible pid=%d gen_k=%d/%d "
                    "(right=%d wrong=%d dropped_hashes=%d) gen=%.1fs",
                    pid, raw_gen_k, args.overgen_k, gen_k, n_wrong,
                    dropped_hashes, full_secs,
                )
                continue

            k_pick = max(k_lo, min(k_hi, 4))  # 4 is the band center
            chosen = right_idx[:k_pick] + wrong_idx[:M_ROLLOUTS - k_pick]
            gens = [gens_all[i] for i in chosen]
            rewards = [rewards_all[i] for i in chosen]
            # Recompute σ on the picked subset (deterministic for binary
            # rewards but we keep the formula symmetric with the miner's
            # `_population_std` so any future non-binary env Just Works).
            mean = sum(rewards) / M_ROLLOUTS
            var = sum((r - mean) ** 2 for r in rewards) / M_ROLLOUTS
            sigma = var ** 0.5

            # Safety nets — the math above guarantees these, but defend
            # against any future env where reward isn't strictly {0,1}.
            if sigma < sigma_min or not (
                BINARY_REWARD_MIN_CORRECT <= k_pick <= BINARY_REWARD_MAX_CORRECT
            ):
                cache.upsert_outcome(PromptOutcome(
                    prompt_idx=pid, checkpoint_hash=ckpt_hash,
                    k=k_pick, sigma=sigma, status="oof",
                    miner_hotkey=args.hotkey,
                ))
                oofs += 1
                logger.info(
                    "full post-pick band/zone fail pid=%d k=%d/%d sigma=%.3f gen=%.1fs",
                    pid, k_pick, M_ROLLOUTS, sigma, full_secs,
                )
                continue

            # Good — persist outcome + the cherry-picked subset.
            cache.upsert_outcome(PromptOutcome(
                prompt_idx=pid, checkpoint_hash=ckpt_hash,
                k=k_pick, sigma=sigma, status="good",
                miner_hotkey=args.hotkey,
            ))
            rollouts_payload = []
            for g, r in zip(gens, rewards):
                rollouts_payload.append({
                    "tokens": [int(t) for t in g["tokens"]],
                    "prompt_length": int(g["prompt_length"]),
                    "reward": float(r),
                })
            # Tier this batch by the prompt's cross-ckpt history so the
            # consumer (engine) can draw a mix of stable/proven/exploratory
            # per submission window.
            if pid in trusted_pids:
                tier = "stable"
            elif pid in xc_stats and xc_stats[pid].get("good", 0) >= 1:
                tier = "proven"
            else:
                tier = "exploratory"
            # local_n=0: prep script doesn't track ckpt revisions like the
            # miner does. Miner uses this only for stale-ckpt eviction; the
            # ckpt_hash gate already covers that, so 0 is safe.
            cache.save_batch(PersistedBatch(
                prompt_idx=pid, checkpoint_hash=ckpt_hash,
                local_n=0, sigma=sigma, k=k_pick,
                rollouts=rollouts_payload,
                miner_hotkey=args.hotkey,
                tier=tier,
            ))
            good += 1
            raw_gen_k = sum(1 for r in rewards_all if r >= 0.5)
            rescued = " RESCUED" if raw_gen_k not in range(
                BINARY_REWARD_MIN_CORRECT, BINARY_REWARD_MAX_CORRECT + 1
            ) else ""
            dropped_note = (
                f" hash_drop={dropped_hashes}" if dropped_hashes else ""
            )
            logger.info(
                "GOOD pid=%d k=%d/%d sigma=%.3f tier=%s gen_k=%d/%d%s%s gen=%.1fs",
                pid, k_pick, M_ROLLOUTS, sigma, tier,
                raw_gen_k, args.overgen_k, rescued, dropped_note, full_secs,
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
