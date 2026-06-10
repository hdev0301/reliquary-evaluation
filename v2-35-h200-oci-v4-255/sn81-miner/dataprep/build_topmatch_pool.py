#!/usr/bin/env python3
"""Pure-numeric in-zone pool builder (top-match) — mirrors the B/D numeric winners.

WHY (from the 4-top-miner study, see diagnostics/target_profiles.txt and
build_inzone_v2.py): emission tracks your SHARE of SELECTED *in-zone* prompt
rewards. In-zone needs reward variance (a mix of correct/wrong over 8 rollouts).
The B/D winners' edge is "decimal/rounding ambiguity": the validator's
_normalize_answer only collapses "x.0"->"x", so "17.5"!="17.50", "83.33"!="83.333"
stay distinct strings -> a correct model still scatters in/out of the parser ->
natural in-zone variance. Plain integers are either 8/8 or 0/8 -> bimodal ->
out-of-zone (the A laggard's 72% int trap), but a balanced integer slice adds
reasoning-difficulty in-zone groups and matches the B/D ~46-56% int band.

This pool is SYMBOLIC-FREE BY CONSTRUCTION (no augmented_math, no latex/percent/
fraction/tuple). It is a strict subset of the numeric universe used by B/D:
  - 50% DECIMAL-AMBIGUOUS  (scarce, high-value; all of them, sets the base size)
  - 50% INTEGER            (sampled to an equal count, seed=0)

OUTPUT (under data/):
  inzone_pool_purenum.json   JSON list of int row-indices into env._dataset.data.table,
                             same format as inzone_pool_v2_numeric.json so pregen's
                             uniform _default_sampler can consume it directly.

Run:  cd /root/reliquary && .venv/bin/python /root/sn81-miner/dataprep/build_topmatch_pool.py
Knobs: --max-prompt-len 400 (default)  --seed 0  (int count is pinned to n_dec; no --sym-ratio)
"""
import argparse, json, os, re
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from reliquary.environment import load_environment
from reliquary.environment.openmathinstruct import _normalize_answer

DATA = "/root/sn81-miner/data"
OUT = f"{DATA}/inzone_pool_purenum.json"
# The top-level instruction names inzone_pool_topmatch.json; the detailed SPEC
# names inzone_pool_purenum.json (same content/format). Write both so either
# consumer path works; the canonical OUT used for assertions/summary is purenum.
OUT_ALIASES = [f"{DATA}/inzone_pool_topmatch.json"]


def regex_mask(col, pattern):
    return pc.match_substring_regex(col, pattern)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-prompt-len", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = load_environment("openmathinstruct")
    tbl = env._dataset.data.table
    N = tbl.num_rows
    ans = pc.utf8_trim_whitespace(tbl["expected_answer"].cast(pa.string()))
    src = tbl["problem_source"].cast(pa.string())
    prob = tbl["problem"].cast(pa.string())

    # ---- base masks (numeric sources only; prompt len <= 400; nonempty answer)
    len_ok = pc.less_equal(pc.utf8_length(prob), args.max_prompt_len)
    nonempty = pc.greater(pc.utf8_length(ans), 0)
    src_num = pc.is_in(src, value_set=pa.array(["gsm8k", "augmented_gsm8k"]))
    base = pc.and_(src_num, pc.and_(len_ok, nonempty))

    # ---- shape masks
    is_int = regex_mask(ans, r"^[\-\+]?[0-9]+$")
    is_dec_full = regex_mask(ans, r"^[\-\+]?[0-9]+\.[0-9]+$")
    is_dec_zero = regex_mask(ans, r"^[\-\+]?[0-9]+\.0+$")  # "3.0" -> validator collapses to "3" -> NOT ambiguous
    is_dec_ambig = pc.and_(is_dec_full, pc.invert(is_dec_zero))

    # DECIMAL-AMBIGUOUS slice (the scarce high-value side; sets the base size)
    dec_mask = pc.and_(is_dec_ambig, base)
    # INTEGER slice (balanced down to n_dec)
    int_mask = pc.and_(is_int, base)

    def idxs_of(mask):
        m = mask.combine_chunks().to_numpy(zero_copy_only=False)
        return np.nonzero(m)[0].astype(int).tolist()

    decimals = idxs_of(dec_mask)
    integers = idxs_of(int_mask)

    # ---- balance to ~50/50: keep ALL decimals, sample an equal count of ints (seed=0)
    rng = np.random.default_rng(args.seed)
    n_dec = len(decimals)
    n_int = min(len(integers), n_dec)
    int_sel = list(rng.choice(integers, size=n_int, replace=False)) if n_int else []
    pool = sorted(set(decimals) | set(int(i) for i in int_sel))

    os.makedirs(DATA, exist_ok=True)
    json.dump(pool, open(OUT, "w"))
    for alias in OUT_ALIASES:
        json.dump(pool, open(alias, "w"))

    ans_list = ans.combine_chunks().to_pylist()

    # ---- classifier mirrored from build_inzone_v2.comp (uses _normalize_answer)
    def comp(idxs):
        c = Counter()
        for i in idxs:
            a = ans_list[i]
            norm = _normalize_answer(a)
            if re.fullmatch(r"-?\d+", norm):
                c["int"] += 1
            elif re.fullmatch(r"-?\d+\.\d+", norm):
                c["decimal"] += 1
            elif "\\frac" in a or re.search(r"\b\d+/\d+\b", a):
                c["fraction"] += 1
            elif "\\sqrt" in a:
                c["radical"] += 1
            elif "\\pi" in a or re.search(r"\bpi\b", a):
                c["pi"] += 1
            elif re.search(r"[a-zA-Z]", re.sub(r"\\[a-zA-Z]+", "", a)):
                c["var/text"] += 1
            elif "%" in a:
                c["percent"] += 1
            else:
                c["other"] += 1
        n = len(idxs) or 1
        return {k: (v, v / n) for k, v in c.most_common()}

    composition = comp(pool)
    print(f"dataset rows (shards loaded) = {N}")
    print(f"decimal-ambiguous available (base, ALL kept) : {len(decimals):>7}")
    print(f"integer available (sampled to n_dec)         : {len(integers):>7}  -> took {n_int}")
    print(f"\nOUTPUT {OUT}")
    print(f"  total idxs : {len(pool)}")
    print(f"  composition (via _normalize_answer):")
    for k, (v, frac) in composition.items():
        print(f"    {k:<10} {v:>7}  {frac:.1%}")

    # ---- contamination eyeshot: sample-print 14 answers
    sample = [ans_list[i] for i in pool[:14]]
    print(f"  sample answers (first 14): {sample}")

    # ---- assertions per spec
    pct = {k: frac for k, (v, frac) in composition.items()}
    dec_pct = pct.get("decimal", 0.0)
    int_pct = pct.get("int", 0.0)
    sym_pct = pct.get("fraction", 0.0) + pct.get("radical", 0.0) + pct.get("pi", 0.0) \
        + pct.get("var/text", 0.0) + pct.get("percent", 0.0)
    other_pct = pct.get("other", 0.0)
    assert 0.46 <= dec_pct <= 0.56, f"decimal share {dec_pct:.1%} outside 46-56% band"
    assert 0.46 <= int_pct <= 0.56, f"integer share {int_pct:.1%} outside 46-56% band"
    assert sym_pct == 0.0, f"symbolic/percent/fraction leaked: {sym_pct:.2%}"

    print(f"\nASSERTIONS PASSED: decimal {dec_pct:.1%} (46-56%), integer {int_pct:.1%} (46-56%), "
          f"symbolic+percent+fraction {sym_pct:.1%}, other {other_pct:.1%}")
    print("\nRESULT_SUMMARY "
          f"path={OUT} total={len(pool)} integer={int_pct:.2%} decimal={dec_pct:.2%} other={other_pct:.2%}")


if __name__ == "__main__":
    main()
