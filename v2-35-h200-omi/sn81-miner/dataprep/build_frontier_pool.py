#!/usr/bin/env python3
"""build_frontier_pool.py — FRONTIER-TARGETED candidate pool for OMI mining.

THE METHOD (why this exists)
----------------------------
A reward only lands when a group is *in-zone*: the validator-recomputed binary
rewards of the 8 rollouts have k in {2..6} correct (sigma >= SIGMA_MIN = 0.43).
That requires picking prompts at the policy's LEARNING FRONTIER — problems the
current checkpoint gets right *sometimes* and wrong *sometimes*.

As the live Qwen3.5-4B checkpoint converges it solves the easy OMI sources
(``gsm8k`` / ``augmented_gsm8k``) ~100% of the time. Screened against such a
pool, every group comes back 8/8 correct -> the curator can't find the wrong
rollouts it needs and the ready-pool starves (the observed ``pregen: +0/8`` /
"pool produced nothing"). The frontier has simply moved to HARDER problems.

This builder produces the durable supply fix: a large candidate pool drawn from
the HARD OMI sources (``augmented_math`` 64% + ``math`` 18% of the 4-shard
dataset) where the converged model still has genuine 20-80% pass mass, filtered
to SHORT prompts (cheaper screen; the live token cap then drops the long-tail
ramblers). The pool is BROAD by default — numeric AND symbolic answers — because
top-OMI-miner telemetry on v23 is symbolic-LEANING (num/sym 37/63; has_var /
tuple / fraction / radical / pi all present). As the policy converged, the easy
numeric word problems became ~100%-solved and the frontier mass moved into
symbolic problems (trig / geom / poly / radicals). A numeric-only run was tried
live and starved: ``screen: 1/24 promising (drop ramble=13 extreme=10
[allcorr=10])`` — the terminating numeric prompts are nearly all all-correct.
``--numeric-only`` remains as an opt-in shorter-output variant.
Note: symbolic answers still score via the SAME ``env.compute_reward`` the
validator runs (normalized-string / value equality), so reward claims match
(no REWARD_MISMATCH); they just run longer, so pair this pool with a higher
``--max-new-tokens`` in run_miner.sh (frontier mode uses 1280).

The per-checkpoint difficulty selection is left to the LIVE two-stage screen
(RELIQUARY_SCREEN_P_LOW/P_HIGH in run_miner.sh) — that measures the current
checkpoint's pass-rate per batch and auto-drifts with each checkpoint. This
script's job is only to hand that screen a candidate stream that *contains*
frontier mass, which the easy sources no longer do.

INDEX CORRECTNESS
-----------------
The written indices are raw row positions in the OMI dataset loaded with
``RELIQUARY_OMI_SHARDS=4`` — the SAME shard set the validator loads. The
validator maps prompt_idx via ``idx % len(dataset)``; with matching shards each
written idx round-trips to the identical problem (no PROMPT_MISMATCH). Always
build with 4 shards (the bin/build_frontier_pool.sh wrapper pins this).

USAGE
-----
    RELIQUARY_OMI_SHARDS=4 python dataprep/build_frontier_pool.py
    # or override:
    python dataprep/build_frontier_pool.py \
        --sources augmented_math,math --max-chars 400 --numeric-only \
        --out /root/sn81-miner/data/inzone_pool_frontier.json

Writes a JSON list of int indices, excluding anything already burned
(submitted_idx.json), and prints source/length/answer stats + a sample.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback


def _parse_args() -> argparse.Namespace:
    sn81 = os.environ.get("SN81", "/root/sn81-miner")
    p = argparse.ArgumentParser(description="Build the frontier-targeted OMI candidate pool.")
    p.add_argument("--sources", default=os.environ.get("FRONTIER_SOURCES", "augmented_math,math"),
                   help="comma-separated problem_source values to KEEP (hard sources). "
                        "Default: augmented_math,math (the bands with frontier mass on a converged model).")
    p.add_argument("--max-chars", type=int, default=int(os.environ.get("FRONTIER_MAX_CHARS", "400")),
                   help="drop prompts longer than this many characters (cheaper screen, shorter completions).")
    # BROAD by default. Top-OMI-miner telemetry on the v23 checkpoint is symbolic-LEANING
    # (num/sym 37/63; has_var/tuple/fraction/radical/pi all well-represented): as the model
    # converged, numeric word problems became ~100%-solved (no frontier mass) and the live
    # frontier moved into symbolic answers. Restricting to numeric throws that mass away and
    # leaves mostly all-correct prompts. --numeric-only is kept as an opt-in for a short-output,
    # clean-reward variant, but it is NOT the default anymore.
    p.add_argument("--numeric-only", dest="numeric_only", action="store_true",
                   default=os.environ.get("FRONTIER_NUMERIC_ONLY", "0") == "1",
                   help="OPT-IN: keep only plain int/decimal/simple-fraction expected_answers "
                        "(shorter output, cleanest reward — but discards the symbolic frontier mass).")
    p.add_argument("--allow-symbolic", dest="numeric_only", action="store_false",
                   help="keep symbolic/LaTeX answers too.")
    p.add_argument("--brittle", dest="brittle", action="store_true",
                   default=os.environ.get("FRONTIER_BRITTLE", "1") == "1",
                   help="(DEFAULT) keep only PARSER-BRITTLE ground_truths (non-value-parseable: "
                        "LaTeX/tuple/interval/has-var). On a converged model the model reasons "
                        "right but formats variably -> natural 2-6/8 scatter -> ~2x the curatable "
                        "density of numeric answers. This is the well-trained-model lever.")
    p.add_argument("--no-brittle", dest="brittle", action="store_false",
                   help="disable the brittleness filter (keep all answer formats).")
    p.add_argument("--math-frac", type=float, default=float(os.environ.get("FRONTIER_MATH_FRAC", "0.5")),
                   help="balance sources so the 'math' source (Hendrycks MATH — hardest, highest "
                        "frontier density) is this fraction of the pool: keep ALL math, downsample "
                        "augmented_math to match. Top-miner winners are ~29%% math vs ~22%% natural. "
                        "0.5 = 50/50. Set 0 to keep the natural source mix.")
    p.add_argument("--cap", type=int, default=int(os.environ.get("FRONTIER_CAP", "0")),
                   help="if >0, randomly subsample the pool to this many idxs (0 = keep all).")
    p.add_argument("--seed", type=int, default=int(os.environ.get("FRONTIER_SEED", "81")))
    p.add_argument("--burned", default=os.environ.get("RELIQUARY_BURNED_PATH", f"{sn81}/data/submitted_idx.json"),
                   help="path to the submitted/burned-idx blocklist to exclude (anti hash_duplicate).")
    p.add_argument("--out", default=os.environ.get("FRONTIER_OUT", f"{sn81}/data/inzone_pool_frontier.json"))
    return p.parse_args()


# expected_answer forms we treat as "numeric": 45, -2, +5, 82.50, 3/4, 1,234.5
_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:/\d+)?$")


def _is_numeric(ans: str) -> bool:
    if not ans:
        return False
    s = ans.strip().replace(",", "").replace("$", "").rstrip(".")
    return bool(_NUMERIC_RE.match(s))


def _is_brittle(ans: str) -> bool:
    """True iff the validator scores this ground_truth by BRITTLE exact-string match
    rather than robust numeric value-equality — i.e. the env's own _as_number() can't
    parse it (LaTeX \\frac/\\sqrt/\\pi, tuples (a,b), intervals, matrices, has-var, text).

    On a WELL-TRAINED model these are the high-yield curatable prompts: the model reasons
    correctly but emits an equivalent form the parser rejects (0.5 vs \\frac12, reordered
    tuple, dropped degree marker) -> natural 2-6/8 reward scatter -> in-zone WITHOUT the
    model actually being wrong. (Harvest: 48%% of top-miner winners are these vs ~33%% of
    the pool.) Uses the env's exact functions so 'brittle' means brittle to the REAL parser.
    Both miner and validator run this same parser, so the wrong rollouts agree -> no
    REWARD_MISMATCH; the rollouts are genuine, only prompt selection is biased."""
    s = str(ans).strip()
    if not s:
        return False
    # Must be scored by exact-STRING match (value-parseable answers are robust, not brittle).
    try:
        from reliquary.environment.openmathinstruct import _as_number, _normalize_answer
        if _as_number(_normalize_answer(s)) is not None:
            return False
    except Exception:
        if _is_numeric(s):
            return False
    # SCATTER-brittle only: the gt has a standard equivalent the model commonly emits that the
    # parser REJECTS, so the model flips between forms (-> 2-6/8), instead of a one-off expression
    # it always writes the same wrong way (-> 0/8 waste, the allwrong=34 we saw on the broad filter).
    #   \frac/\sqrt/\pi  -> model writes a/b, a decimal, or 'pi'/spacing variants
    #   \circ / ° / degree -> model drops the marker
    #   (a,b) / [a,b] tuple/interval/coords -> model reorders or respaces
    if re.search(r"\\[dt]?frac|\\sqrt|\\pi|\\circ|°|degree", s, re.I):
        return True
    if re.match(r"^[\(\[].*,.*[\)\]]$", s):
        return True
    return False


def main() -> int:
    args = _parse_args()
    shards = os.environ.setdefault("RELIQUARY_OMI_SHARDS", "4")
    if shards != "4":
        print(f"WARNING: RELIQUARY_OMI_SHARDS={shards} != 4 — written idxs will NOT match a "
              f"4-shard validator and will PROMPT_MISMATCH. Re-run with RELIQUARY_OMI_SHARDS=4.",
              file=sys.stderr)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc
        from reliquary.environment import load_environment

        env = load_environment("openmathinstruct")
        n_total = len(env)
        tbl = env._dataset.data.table
        src = tbl["problem_source"].cast(pa.string())
        prob = tbl["problem"].cast(pa.string())
        ans = tbl["expected_answer"].cast(pa.string())

        # --- cheap vectorized filters: hard source + short prompt ---
        mask = pc.is_in(src, value_set=pa.array(sources, pa.string()))
        mask = pc.and_(mask, pc.less_equal(pc.utf8_length(prob), args.max_chars))
        keep = mask.combine_chunks().to_numpy(zero_copy_only=False)
        cand = np.nonzero(keep)[0]
        print(f"dataset(4 shards)={n_total}  after source{sources}+len<= {args.max_chars}: {len(cand)}")

        # --- answer-format filter (cheap python over the reduced candidate set) ---
        if args.brittle:
            ansl = ans.combine_chunks().to_pylist()
            cand = np.array([i for i in cand if _is_brittle(ansl[i])], dtype=np.int64)
            print(f"after BRITTLE filter (parser-ambiguous gt -> ~2x curatable density on a converged model): {len(cand)}")
        elif args.numeric_only:
            ansl = ans.combine_chunks().to_pylist()
            cand = np.array([i for i in cand if _is_numeric(ansl[i])], dtype=np.int64)
            print(f"after numeric-answer filter: {len(cand)}")

        # --- exclude already-burned (won/submitted) idxs ---
        burned: set[int] = set()
        try:
            burned = set(int(i) for i in json.load(open(args.burned)))
        except Exception:
            pass
        if burned:
            before = len(cand)
            cand = np.array([int(i) for i in cand if int(i) not in burned], dtype=np.int64)
            print(f"excluded {before - len(cand)} burned idxs (blocklist={len(burned)})")

        if len(cand) == 0:
            print("FATAL: pool is empty after filters — loosen --max-chars / --sources.", file=sys.stderr)
            return 1

        rng = np.random.default_rng(args.seed)

        # --- math-source boosting: keep ALL math, downsample the rest to hit math_frac ---
        if 0.0 < args.math_frac < 1.0 and "math" in sources:
            srcl = src.combine_chunks().to_pylist()
            math_idx = [int(i) for i in cand if srcl[i] == "math"]
            other_idx = [int(i) for i in cand if srcl[i] != "math"]
            n_other_target = int(len(math_idx) * (1.0 - args.math_frac) / args.math_frac)
            if 0 <= n_other_target < len(other_idx):
                other_idx = [int(i) for i in rng.choice(other_idx, size=n_other_target, replace=False)]
            cand = np.array(math_idx + other_idx, dtype=np.int64)
            print(f"math-boost: math={len(math_idx)} + other={len(other_idx)} "
                  f"-> math_frac~{len(math_idx)/max(1,len(cand)):.2f} (target {args.math_frac})")

        # --- optional subsample ---
        if args.cap and len(cand) > args.cap:
            cand = rng.choice(cand, size=args.cap, replace=False)
            print(f"subsampled to cap={args.cap}")
        rng.shuffle(cand)  # shuffle so a sequential walk mixes sources/difficulty

        idxs = [int(i) for i in cand]
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(idxs, open(args.out, "w"))

        # --- report: source / length / sample (so you can eyeball the band) ---
        srcl = src.combine_chunks().to_pylist()
        probl = prob.combine_chunks().to_pylist()
        ansl2 = ans.combine_chunks().to_pylist()
        import collections
        by_src = collections.Counter(srcl[i] for i in idxs[: min(len(idxs), 200000)])
        print(f"\nBUILT {args.out}  size={len(idxs)}")
        print("source mix (first 200k):", dict(by_src))
        print("samples:")
        for i in idxs[:8]:
            q = probl[i].replace("\n", " ")
            print(f"  idx={i:>8d} ans={ansl2[i]!r:>12} | {q[:90]}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
