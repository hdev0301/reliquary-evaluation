"""Build the HARD-INTEGER pool the standard build_inzone_v2.py misses.

Top int-heavy miners win on integer-ANSWER problems from the HARD math source
(augmented_math/math: cubic factoring, binomial theorem, modular arithmetic),
NOT gsm8k word-problem integers. On hard problems the model's VARIED reasoning
errors produce DISTINCT terminating wrong integers = the scarce k=6 wrong side.
gsm8k integers are either trivial (8/8 -> sigma 0) or ramble-when-wrong (no
terminating wrong) -> not_curatable.

build_inzone_v2.py masks miss this slice entirely:
  symbolic_mask = NOT is_int AND src_math   (excludes integer answers)
  integer_mask  = is_int AND src_numeric    (gsm8k only)
  basen_mask    = is_int AND src_math AND basen_keyword  (tiny subset)

This captures: is_int AND src_math (all hard integer-answer math problems).
Runtime SCREEN_P_LOW/HIGH=[0.20,0.80] then keeps only the ones that actually
scatter on the live checkpoint (the difficulty filter).
"""
import json
import os

import pyarrow as pa
import pyarrow.compute as pc
from reliquary.environment import load_environment

DATA = "/root/sn81-miner/data"
MAX_PROMPT_LEN = int(os.environ.get("HARDINT_MAX_PROMPT_LEN", "600"))
OUT = os.environ.get("HARDINT_OUT", f"{DATA}/inzone_pool_v2_hardint.json")


def main():
    env = load_environment("openmathinstruct")
    tbl = env._dataset.data.table
    ans = pc.utf8_trim_whitespace(tbl["expected_answer"].cast(pa.string()))
    src = tbl["problem_source"].cast(pa.string())
    prob = tbl["problem"].cast(pa.string())

    is_int = pc.match_substring_regex(ans, r"^[\-\+]?[0-9]+$")
    src_math = pc.is_in(src, value_set=pa.array(["augmented_math", "math"]))
    len_ok = pc.less_equal(pc.utf8_length(prob), MAX_PROMPT_LEN)
    nonempty = pc.greater(pc.utf8_length(ans), 0)

    mask = pc.and_(pc.and_(is_int, src_math), pc.and_(len_ok, nonempty))
    idxs = [i for i, v in enumerate(mask.to_pylist()) if v]

    json.dump(idxs, open(OUT, "w"))
    print(f"hard-int pool: {len(idxs)} idxs  (is_int AND src in augmented_math/math, len<={MAX_PROMPT_LEN})")
    print(f"written: {OUT}")

    # source + sample diagnostics
    import random
    random.seed(0)
    ans_l = ans.to_pylist()
    src_l = src.to_pylist()
    samp = random.sample(idxs, min(16, len(idxs)))
    from collections import Counter
    print("source mix:", dict(Counter(src_l[i] for i in idxs)))
    print("sample answers:", [ans_l[i] for i in samp])


if __name__ == "__main__":
    main()
