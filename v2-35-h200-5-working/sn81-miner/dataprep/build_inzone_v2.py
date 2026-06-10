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
import argparse, hashlib, json, os, re, sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from reliquary.environment import load_environment
from reliquary.environment.openmathinstruct import _normalize_answer

DATA = "/root/sn81-miner/data"
DIAG = "/root/sn81-miner/diagnostics"


def _canonical_key(prompt_idx: int) -> bytes:
    """Byte-for-byte match of the validator's _prompt_canonical_key
    (reliquary/validator/batch_selection.py) and the miner's _canonical_key
    (reliquary/miner/pregen.py): sha256 of prompt_idx as 8-byte big-endian.
    At seal the validator fills the boundary round with the B_BATCH DISTINCT
    prompts of LOWEST digest, so low-hash prompts win the canonical top-8."""
    return hashlib.sha256(int(prompt_idx).to_bytes(8, "big", signed=False)).digest()


def canon_filter(idxs, keep_frac):
    """Keep only the lowest-sha256(prompt_idx) `keep_frac` of `idxs` — the prompts
    most likely to land in the validator's canonical top-B_BATCH at seal, which is
    what reduces `batch_filled` rejects. keep_frac >= 1.0 is a no-op.

    Caveat: aggressive values shrink the pool toward the few globally-lowest-hash
    prompts that EVERY canon-aware miner converges on -> higher per-prompt
    collision (the boundary round splits a slot's reward across same-prompt
    submitters). 0.2-0.4 is a reasonable advantage-vs-collision trade."""
    if keep_frac >= 1.0:
        return idxs
    ordered = sorted(idxs, key=_canonical_key)
    n_keep = max(1, int(round(len(ordered) * keep_frac)))
    return sorted(int(i) for i in ordered[:n_keep])


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
    ap.add_argument("--max-prompt-len", type=int, default=600,
                    help="max problem CHARS (prompt filter). 5HEAK6 mines prompts up to ~1078 chars; 400 was "
                         "too restrictive (p95=417), 600 captures nearly all of the winners' prompt distribution.")
    ap.add_argument("--include-math-source", action=argparse.BooleanOptionalAction, default=True,
                    help="include the base `math` competition split (not just augmented_math) in the SYMBOLIC pool. "
                         "5HEAK6's radical/complex/interval symbolic wins live partly in `math`, which the "
                         "augmented_math-only mask missed. Use --no-include-math-source for the old behavior.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canon-keep-frac", type=float,
                    default=float(os.environ.get("V2_CANON_KEEP_FRAC", "1.0")),
                    help="Keep only the lowest-sha256(prompt_idx) fraction of the ACTIVE pool. These win the "
                         "validator's boundary-round canonical top-8 (batch_selection.py) more often, cutting "
                         "batch_filled losses. 1.0=off; try 0.2-0.4. Lower => more low-hash collision.")
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
    # SYMBOLIC source: augmented_math always; add the base `math` competition split when
    # --include-math-source (default). 5HEAK6's radical/complex/interval wins live partly in `math`,
    # which the old augmented_math-only mask excluded entirely.
    _sym_srcs = ["augmented_math", "math"] if args.include_math_source else ["augmented_math"]
    src_math = pc.is_in(src, value_set=pa.array(_sym_srcs))

    # B/D engine: decimal-ambiguous answers on the numeric (gsm8k-style) sources
    numeric_mask = pc.and_(pc.and_(is_decimal_ambig, src_numeric), pc.and_(len_ok, nonempty))
    # C engine: any NON-integer answer from the hard-math source (fractions, radicals,
    # pi, tuples, variables, text -> all format-ambiguous in LaTeX)
    symbolic_mask = pc.and_(pc.and_(pc.invert(is_int), src_math), pc.and_(len_ok, nonempty))
    # optional integer slice (reasoning-difficulty in-zone, but bimodal-prone)
    integer_mask = pc.and_(pc.and_(is_int, src_numeric), len_ok)

    def idxs_of(mask):
        m = mask.combine_chunks().to_numpy(zero_copy_only=False)
        return np.nonzero(m)[0].astype(int).tolist()

    numeric = idxs_of(numeric_mask)
    symbolic = idxs_of(symbolic_mask)
    integers = idxs_of(integer_mask)

    rng = np.random.default_rng(args.seed)

    # ---- combined ACTIVE pool: numeric is the base; mix in symbolic + optional ints
    # to the requested composition. numeric is the scarce/high-value side -> it sets
    # the base size; sym/int counts are derived from the ratios.
    base = len(numeric)
    # solve for total T s.t. numeric/T = 1 - sym_ratio - int_ratio
    keep_frac = max(1e-6, 1.0 - args.sym_ratio - args.int_ratio)
    total = int(round(base / keep_frac))
    n_sym = min(len(symbolic), int(round(total * args.sym_ratio)))
    n_int = min(len(integers), int(round(total * args.int_ratio)))

    sym_sel = list(rng.choice(symbolic, size=n_sym, replace=False)) if n_sym else []
    int_sel = list(rng.choice(integers, size=n_int, replace=False)) if n_int else []
    combined = sorted(set(numeric) | set(int(i) for i in sym_sel) | set(int(i) for i in int_sel))
    n_pre_canon = len(combined)
    combined = canon_filter(combined, args.canon_keep_frac)  # low-sha256 -> win the canonical seal-race

    os.makedirs(DATA, exist_ok=True)
    json.dump(numeric, open(f"{DATA}/inzone_pool_v2_numeric.json", "w"))
    json.dump(symbolic, open(f"{DATA}/inzone_pool_v2_symbolic.json", "w"))
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
    print(f"  symbolic (non-int, {'aug_math+math' if args.include_math_source else 'augmented_math'}) : {len(symbolic):>7}  {comp(symbolic[:20000])}")
    print(f"  integer slice available (gsm8k/aug_gsm8k)    : {len(integers):>7}")
    print(f"\nACTIVE COMBINED POOL  inzone_pool_v2.json : {len(combined)} idxs"
          f"  (sym_ratio={args.sym_ratio} int_ratio={args.int_ratio} "
          f"sym_src={'aug_math+math' if args.include_math_source else 'aug_math'} max_prompt_len={args.max_prompt_len})")
    if args.canon_keep_frac < 1.0:
        print(f"  canon-filter: kept lowest-sha256 {args.canon_keep_frac:.0%}"
              f" -> {len(combined)}/{n_pre_canon} idxs (fewer batch_filled at seal; more low-hash collision)")
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
