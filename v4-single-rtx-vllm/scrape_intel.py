#!/usr/bin/env python3
"""Scrape recent R2 archive windows for accepted submissions and persist
the (prompt_idx, k, sigma) intel into the Supabase ``prompt_outcomes``
table. The live miner picks from those rows first, skipping prescreen
on prompts other miners already proved in-zone for the current ckpt.

This is the safe complement to ``prep_dataset.py``:

* ``prep_dataset.py`` runs the model locally to discover new in-zone
  prompts — costs ~160s of GPU per batch but produces fresh rollouts
  the miner can submit with unique hashes.
* ``scrape_intel.py`` reads other miners' accepted submissions from
  the public R2 archive — costs only network bandwidth, but does NOT
  give us submittable rollouts (those token sequences are already in
  the validator's hash set, so re-submitting them returns HASH_DUPLICATE
  — we'd need to re-generate locally on the same prompt to get unique
  rollouts).

R2 archive caveat: the legacy code used to assume all scraped windows
used the validator's currently-published ckpt. The v2.3+ archive now
stamps each batch entry with ``claimed_checkpoint_hash`` (the ckpt the
miner cited at submission time, which is what the validator scored
against). We use that per-row when present and fall back to the
validator's current ckpt for legacy rows.

Hash persistence: every accepted rollout exposes a ``hash`` field —
the per-rollout sha256 the validator's hash_set keys on. We mirror
those into the ``accepted_rollout_hashes`` table so the miner can
pre-flight a freshly-generated rollout against the validator's
dedup set BEFORE paying for submission — a hit means HASH_DUPLICATE
is guaranteed, so the miner re-samples with a fresh seed instead.

Usage:
    cd /root/reliquary && source scripts/.env
    python scripts/scrape_intel.py                  # last 10 windows
    python scripts/scrape_intel.py --lookback 5     # only last 5
    python scripts/scrape_intel.py --since-window 6820 --until-window 6830
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))

from reliquary.miner.persistence import PromptOutcome, cache_from_env, resolve_hotkey

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("scrape_intel")

R2_WINDOW_URL = "https://www.reliqua.ai/api/r2/window/{n}"
STATE_URL_DEFAULT = os.environ.get("RELIQUARY_VALIDATOR_URL", "")


def _fetch_window(n: int) -> dict | None:
    url = R2_WINDOW_URL.format(n=n)
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        logger.warning("window %d fetch failed: %s", n, e)
        return None


def _fetch_validator_ckpt(validator_url: str) -> tuple[str, str, int, int]:
    """Returns (repo_id, revision, ckpt_n, current_window_n)."""
    url = validator_url.rstrip("/") + "/state"
    with urllib.request.urlopen(url, timeout=15) as r:
        st = json.loads(r.read())
    return (
        st["checkpoint_repo_id"],
        st["checkpoint_revision"],
        int(st["checkpoint_n"]),
        int(st["window_n"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=10,
                    help="number of past windows to scrape (default 10)")
    ap.add_argument("--since-window", type=int, default=None,
                    help="explicit start window (overrides --lookback)")
    ap.add_argument("--until-window", type=int, default=None,
                    help="explicit end window (overrides default = current)")
    ap.add_argument("--ckpt-hash", type=str, default=None,
                    help="override the ckpt_hash tag (default: validator current)")
    ap.add_argument("--validator-url", type=str, default=STATE_URL_DEFAULT)
    ap.add_argument("--hotkey", type=str, default=None,
                    help="miner_hotkey to stamp rows (informational only)")
    ap.add_argument("--watch", type=int, default=0,
                    help="if >0, after one pass sleep and re-scrape forever")
    ap.add_argument("--include-rejected", action="store_true",
                    help="also persist OOZ/dud info from rejected submissions")
    ap.add_argument("--throttle", type=float, default=0.5,
                    help="seconds to sleep between window fetches (avoids upstream rate limit)")
    args = ap.parse_args()

    if not args.ckpt_hash:
        if not args.validator_url:
            logger.error(
                "need --ckpt-hash or --validator-url (or RELIQUARY_VALIDATOR_URL env)"
            )
            return 2
        try:
            repo_id, revision, ckpt_n, cur_win = _fetch_validator_ckpt(args.validator_url)
            ckpt_hash = revision
            logger.info(
                "validator current: repo=%s rev=%s ckpt_n=%d window_n=%d",
                repo_id, revision, ckpt_n, cur_win,
            )
        except Exception as e:
            if not args.watch:
                logger.error("validator /state fetch failed: %s", e)
                return 2
            logger.warning(
                "validator /state failed at startup: %s — entering watch loop; "
                "do_pass will retry per-iteration", e,
            )
            ckpt_hash = None  # do_pass re-queries each pass
    else:
        ckpt_hash = args.ckpt_hash
        if not args.until_window:
            cur_win = None
        else:
            cur_win = None

    resolved = resolve_hotkey(args.hotkey)
    if resolved != args.hotkey:
        logger.info("resolved hotkey %r -> %s", args.hotkey, resolved)
    args.hotkey = resolved
    cache = cache_from_env(miner_hotkey=args.hotkey)
    if not cache.enabled:
        logger.error("Supabase cache disabled — check RELIQUARY_SUPABASE_URL/KEY")
        return 2

    def do_pass() -> tuple[int, int, int, int]:
        # If neither override is set, re-fetch validator state per pass so
        # both the end window AND ckpt_hash track ckpt advances naturally.
        pass_ckpt = ckpt_hash
        if args.until_window is not None and args.ckpt_hash:
            end = int(args.until_window)
        else:
            try:
                _, revision, _, cur = _fetch_validator_ckpt(args.validator_url)
            except Exception as e:
                logger.warning("validator state failed: %s", e)
                return 0, 0, 0, 0
            if args.until_window is not None:
                end = int(args.until_window)
            else:
                end = cur - 1
            if not args.ckpt_hash:
                pass_ckpt = revision
        if pass_ckpt is None:
            logger.warning("no ckpt_hash available yet; skipping pass")
            return 0, 0, 0
        if args.since_window is not None:
            start = int(args.since_window)
        else:
            start = end - int(args.lookback) + 1

        added_good = 0
        added_oof = 0
        added_hashes = 0
        windows_seen = 0
        for n in range(start, end + 1):
            w = _fetch_window(n)
            if args.throttle > 0:
                time.sleep(args.throttle)
            if w is None:
                continue
            windows_seen += 1
            data = (w or {}).get("data") or {}
            batch = data.get("batch") or []
            hash_rows_this_window: list[dict] = []
            for item in batch:
                pid = item.get("prompt_idx")
                rollouts = item.get("rollouts") or []
                if pid is None or not rollouts:
                    continue
                # Per-row ckpt tagging — the validator stamps each batch
                # entry with the ckpt the miner cited at submission time.
                # That is the ckpt the rewards were scored under, which
                # is what we want to key the outcome on. Fall back to the
                # current validator ckpt for legacy rows missing the field.
                row_ckpt = item.get("claimed_checkpoint_hash") or pass_ckpt
                rewards = [float(r.get("reward", 0.0)) for r in rollouts]
                k = sum(1 for r in rewards if r >= 0.5)
                mean = sum(rewards) / max(1, len(rewards))
                var = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards))
                sigma = var ** 0.5
                cache.upsert_outcome(PromptOutcome(
                    prompt_idx=int(pid),
                    checkpoint_hash=row_ckpt,
                    k=int(k), sigma=float(sigma), status="good",
                    miner_hotkey=item.get("hotkey"),
                ))
                added_good += 1
                # Mirror per-rollout hashes for the miner's pre-flight
                # dedup check (skip rollouts missing the field — legacy).
                for r in rollouts:
                    h = r.get("hash")
                    if not h:
                        continue
                    hash_rows_this_window.append({
                        "rollout_hash": h,
                        "prompt_idx": int(pid),
                        "checkpoint_hash": row_ckpt,
                        "window_n": int(n),
                        "miner_hotkey": item.get("hotkey"),
                    })
            if hash_rows_this_window:
                added_hashes += cache.upsert_accepted_hashes(hash_rows_this_window)
            if args.include_rejected:
                rej = data.get("rejected") or []
                for item in rej:
                    pid = item.get("prompt_idx")
                    reason = item.get("reason") or ""
                    if pid is None:
                        continue
                    # We can only sensibly cache OOZ-style rejections —
                    # bad_termination / bad_envelope are submission-side
                    # issues not prompt-quality issues.
                    if reason not in ("out_of_zone", "reward_distribution"):
                        continue
                    row_ckpt = item.get("claimed_checkpoint_hash") or pass_ckpt
                    cache.upsert_outcome(PromptOutcome(
                        prompt_idx=int(pid),
                        checkpoint_hash=row_ckpt,
                        k=0, sigma=0.0, status="oof",
                        miner_hotkey=item.get("hotkey"),
                    ))
                    added_oof += 1
            logger.info(
                "window %d: batch=%d cumulative_good=%d cumulative_oof=%d hashes=%d",
                n, len(batch), added_good, added_oof, added_hashes,
            )
        return windows_seen, added_good, added_oof, added_hashes

    while True:
        t0 = time.time()
        ws, g, o, h = do_pass()
        logger.info(
            "pass done windows=%d good=%d oof=%d hashes=%d elapsed=%.1fs",
            ws, g, o, h, time.time() - t0,
        )
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
