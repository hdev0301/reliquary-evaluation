#!/usr/bin/env python3
"""Variance-targeted pool builder (v2) — implements the top-miner analysis.

WHY (from the 4-top-miner study, see diagnostics/target_profiles.txt):
  Emission = 72-window EMA of your SHARE of SELECTED *in-zone* prompt rewards.
  In-zone needs reward variance sigma>=0.43 over 8 rollouts, i.e. a MIX of correct
  and wrong. A converged checkpoint is BIMODAL: it scores 8/8 or 0/8 on most prompts
  -> out-of-zone -> wasted compute. The empirical edge of the top earners is the
  ANSWER-SHAPE of the prompts they mine:

    winners' accepted-group ground-truth shapes (cached API oracle):
      B_5DARq6 : 54% decimal, 46% int            <- "decimal/rounding ambiguity" engine
      D_5HQbAQ4: 55% decimal, 45% int  (cumTAO #1)
      C_5HEAK6 : 23% fraction +9% tuple +6% var +radical/pi/matrix (44% symbolic)  <- "LaTeX-form ambiguity" engine
      A_5F6VZ2 : 72% INT (bimodal trap)          <- laggard, floor-pinned at sigma=0.433

  Mechanism (verified against reliquary/environment/openmathinstruct.py:_normalize_answer):
    the validator's normalizer collapses only "x.0"->"x". So "17.5" != "17.50",
    "83.33" != "83.333", "\\frac{1}{2}" != "0.5" all stay DISTINCT strings. Those are
    exactly the answers where a *correct* model still scatters in/out of the parser
    -> NATURAL in-zone variance with reliable EOS. Plain integers do not: the model
    is either right (8/8) or wrong (0/8) -> bimodal -> out-of-zone.

  pregen's sampler picks UNIFORMLY AT RANDOM over pool MEMBERSHIP (pregen.py
  _default_sampler), so ordering is irrelevant: the only lever is *which idxs are in
  the pool*. A pool hard-filtered to decimal+symbolic answers raises in-zone density
  per screen -> faster discovery of the rare curatable prompts (the real bottleneck).

OUTPUTS (under data/):
  inzone_pool_v2_numeric.json   decimal-ambiguous answers, gsm8k/augmented_gsm8k  (B/D engine)
  inzone_pool_v2_symbolic.json  non-integer answers from augmented_math           (C engine)
  inzone_pool_v2.json           ACTIVE combined pool = all numeric + sym_ratio symbolic
                                (+ optional capped integers via --int-ratio)

Run:  cd /root/reliquary && .venv/bin/python /root/sn81-miner/dataprep/build_inzone_v2.py
Knobs (argv or env): --sym-ratio 0.25  --int-ratio 0.0  --max-prompt-len 400
"""
import argparse, json, os, re, sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from reliquary.environment import load_environment
from reliquary.environment.openmathinstruct import _normalize_answer

DATA = "/root/sn81-miner/data"
DIAG = "/root/sn81-miner/diagnostics"


def regex_mask(col, pattern):
    return pc.match_substring_regex(col, pattern)


def main():
    ap = argparse.ArgumentParser()
    # Defaults target the WINNER-ACCEPT mix (B/D ~55% decimal / ~45% int; C ~44% symbolic):
    # decimal-dominant (the high-yield sigma=0.5 engine), a C-style symbolic slice for
    # diversification, and a modest integer slice for reasoning-difficulty in-zone groups
    # (hedge against decimal-only discovery starvation). Set --int-ratio 0 for the pure
    # "edge" pool, or --int-ratio 0.45 to match the winner accept mix exactly.
    ap.add_argument("--sym-ratio", type=float, default=float(os.environ.get("V2_SYM_RATIO", "0.15")),
                    help="symbolic idxs as a fraction of the combined pool (C-style diversification)")
    ap.add_argument("--int-ratio", type=float, default=float(os.environ.get("V2_INT_RATIO", "0.30")),
                    help="plain-integer idxs as a fraction of the combined pool (bimodal-prone; hedge)")
    ap.add_argument("--basen-ratio", type=float, default=float(os.environ.get("V2_BASEN_RATIO", "0.12")),
                    help="base-N format-ambiguous INTEGER idxs ('express N in base 7' -> answer like 1010100) as a "
                         "fraction of the pool. They LOOK integer but SCATTER (model emits base-10 vs base-N) and are "
                         "5HEAK6's ~22pct 'int' wins -- caught by neither --int-ratio (gsm) nor symbolic (is_int).")
    ap.add_argument("--max-prompt-len", type=int, default=600,
                    help="max problem CHARS. 5HEAK6 mines prompts up to ~1078 (p95=417); 600 captures nearly all "
                         "and lets the longer base-conversion prompts through.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-symbolic", action="store_true",
                    help="include ALL symbolic idxs (no sampling); symbolic becomes the pool base for max coverage")
    ap.add_argument("--seed-file", type=str, default=None,
                    help="JSON list of extra prompt idxs to UNION in (e.g. proven-winner idxs from a top miner)")
    args = ap.parse_args()

    env = load_environment("openmathinstruct")
    tbl = env._dataset.data.table
    N = tbl.num_rows
    ans = pc.utf8_trim_whitespace(tbl["expected_answer"].cast(pa.string()))
    src = tbl["problem_source"].cast(pa.string())
    prob = tbl["problem"].cast(pa.string())

    len_ok = pc.less_equal(pc.utf8_length(prob), args.max_prompt_len)
    nonempty = pc.greater(pc.utf8_length(ans), 0)

    is_int = regex_mask(ans, r"^[\-\+]?[0-9]+$")
    is_dec_full = regex_mask(ans, r"^[\-\+]?[0-9]+\.[0-9]+$")
    is_dec_zero = regex_mask(ans, r"^[\-\+]?[0-9]+\.0+$")  # "3.0" -> validator collapses to "3" -> NOT ambiguous
    is_decimal_ambig = pc.and_(is_dec_full, pc.invert(is_dec_zero))

    src_numeric = pc.is_in(src, value_set=pa.array(["gsm8k", "augmented_gsm8k"]))
    # C engine: hard symbolic sources. Includes the original "math" competition set
    # (5HEAK6 banks ~9% of accepts from it) in addition to augmented_math.
    src_math = pc.is_in(src, value_set=pa.array(["augmented_math", "math"]))

    # B/D engine: decimal-ambiguous answers on the numeric (gsm8k-style) sources
    numeric_mask = pc.and_(pc.and_(is_decimal_ambig, src_numeric), pc.and_(len_ok, nonempty))
    # C engine: any NON-integer answer from the hard-math source (fractions, radicals,
    # pi, tuples, variables, text -> all format-ambiguous in LaTeX)
    symbolic_mask = pc.and_(pc.and_(pc.invert(is_int), src_math), pc.and_(len_ok, nonempty))
    # optional integer slice (reasoning-difficulty in-zone, but bimodal-prone)
    integer_mask = pc.and_(pc.and_(is_int, src_numeric), len_ok)
    # base-N format-ambiguous INTEGERS: integer-valued answers to base-conversion problems
    # ("express N in base 7" -> answer "1010100"). They look like plain ints (so the symbolic
    # NON-int mask skips them) but come from the math source (so the gsm-only integer_mask skips
    # them too) -- the exact 22% "int" slice 5HEAK6 wins. The model scatters base-10 vs base-N
    # -> natural in-zone variance. Gate on a base-conversion KEYWORD in the prompt so we don't
    # pull plain-arithmetic integers (bimodal). "binary" requires a number-word to dodge
    # "binary operation / tree / search".
    prob_lower = pc.utf8_lower(prob)
    basen_kw = regex_mask(
        prob_lower,
        r"base[ \-_]?(\d|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|sixteen)"
        r"|\boctal\b|hexadecimal|\bradix\b"
        r"|binary (number|representation|form|notation|expansion|equivalent|digit|string)",
    )
    basen_mask = pc.and_(pc.and_(is_int, src_math), pc.and_(basen_kw, pc.and_(len_ok, nonempty)))

    def idxs_of(mask):
        m = mask.combine_chunks().to_numpy(zero_copy_only=False)
        return np.nonzero(m)[0].astype(int).tolist()

    numeric = idxs_of(numeric_mask)
    symbolic = idxs_of(symbolic_mask)
    integers = idxs_of(integer_mask)
    basen = idxs_of(basen_mask)

    rng = np.random.default_rng(args.seed)

    # ---- combined ACTIVE pool
    if args.full_symbolic:
        # MAX COVERAGE: take ALL symbolic (no sampling), all numeric (scarce), and an
        # integer slice sized so symbolic == sym_ratio of the pool. Trade-off: the
        # 14837 decimal-ambiguous prompts can't grow, so decimal share shrinks as the
        # pool grows -- coverage over composition-purity.
        n_sym = len(symbolic)
        total = int(round(n_sym / max(1e-6, args.sym_ratio)))
        n_int = min(len(integers), int(round(total * args.int_ratio)))
        n_basen = min(len(basen), int(round(total * args.basen_ratio)))
        sym_sel = list(symbolic)
    else:
        # numeric is the base; mix in symbolic + optional ints to the requested ratios.
        base = len(numeric)
        # solve for total T s.t. numeric/T = 1 - sym_ratio - int_ratio - basen_ratio
        keep_frac = max(1e-6, 1.0 - args.sym_ratio - args.int_ratio - args.basen_ratio)
        total = int(round(base / keep_frac))
        n_sym = min(len(symbolic), int(round(total * args.sym_ratio)))
        n_int = min(len(integers), int(round(total * args.int_ratio)))
        n_basen = min(len(basen), int(round(total * args.basen_ratio)))
        sym_sel = list(rng.choice(symbolic, size=n_sym, replace=False)) if n_sym else []

    int_sel = list(rng.choice(integers, size=n_int, replace=False)) if n_int else []
    basen_sel = list(rng.choice(basen, size=n_basen, replace=False)) if n_basen else []
    combined_set = (set(numeric) | set(int(i) for i in sym_sel)
                    | set(int(i) for i in int_sel) | set(int(i) for i in basen_sel))

    # ---- seed proven-winner idxs (e.g. a top miner's accepted prompts) ----
    n_seed = 0
    if args.seed_file and os.path.exists(args.seed_file):
        seed = [int(i) for i in json.load(open(args.seed_file)) if 0 <= int(i) < N]
        n_seed = len(set(seed) - combined_set)
        combined_set |= set(seed)
    combined = sorted(combined_set)

    os.makedirs(DATA, exist_ok=True)
    json.dump(numeric, open(f"{DATA}/inzone_pool_v2_numeric.json", "w"))
    json.dump(symbolic, open(f"{DATA}/inzone_pool_v2_symbolic.json", "w"))
    json.dump(basen, open(f"{DATA}/inzone_pool_v2_basen.json", "w"))
    json.dump(combined, open(f"{DATA}/inzone_pool_v2.json", "w"))

    ans_list = ans.combine_chunks().to_pylist()

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
            else:
                c["other"] += 1
        n = len(idxs) or 1
        return {k: f"{v} ({v/n:.0%})" for k, v in c.most_common()}

    print(f"dataset rows (shards loaded) = {N}")
    print(f"\nSUB-POOLS:")
    print(f"  numeric (decimal-ambiguous, gsm8k/aug_gsm8k) : {len(numeric):>7}  {comp(numeric[:20000])}")
    print(f"  symbolic (non-int, augmented_math+math)      : {len(symbolic):>7}  {comp(symbolic[:20000])}")
    print(f"  base-N int (format-ambig, aug_math+math)     : {len(basen):>7}  {comp(basen[:20000])}")
    print(f"  integer slice available (gsm8k/aug_gsm8k)    : {len(integers):>7}")
    print(f"\nACTIVE COMBINED POOL  inzone_pool_v2.json : {len(combined)} idxs"
          f"  (sym_ratio={args.sym_ratio} int_ratio={args.int_ratio} basen_ratio={args.basen_ratio} "
          f"full_symbolic={args.full_symbolic} max_prompt_len={args.max_prompt_len})")
    if n_seed:
        print(f"  seeded {n_seed} extra proven-winner idxs from {args.seed_file}")
    print(f"  composition: {comp(combined)}")
    print(f"  sample answers: {[ans_list[i] for i in combined[:14]]}")

    # ---- compare density vs the current broad pool ----
    for name in ("inzone_pool.json", "inzone_pool_qwen35.json"):
        p = f"{DATA}/{name}"
        if os.path.exists(p):
            old = json.load(open(p))
            old = [i for i in old if i < N]
            print(f"\n  [baseline {name}] {len(old)} idxs  composition: {comp(old[:20000])}")

    # ---- winner-oracle target (what actually wins) ----
    print("\nWINNER ORACLE (target distribution, from cached accepted in-zone groups):")
    print("  B/D ~55% decimal / ~45% int ; C ~44% symbolic ; (A laggard 72% int)")
    print("  -> v2 numeric pool is decimal-pure (the scarce high-value side); the screen/curation")
    print("     supplies the wrong rollouts. Dial --int-ratio up toward ~0.45 to match winner mix")
    print("     of accepts if discovery starves on decimals alone.")


if __name__ == "__main__":
    main()
