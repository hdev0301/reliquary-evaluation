#!/usr/bin/env python3
"""Adversarial independent verification of inzone_pool_topmatch.json.

Recomputes everything from the FILE (not the builder printout):
  - load pool via json
  - total size
  - all idxs in [0, len(env))
  - random sample >=3000 (seed=1): call env.get_problem(i), classify ground_truth
    with the EXACT build_inzone_v2.comp logic (_normalize_answer then int/dec/...)
  - %integer, %decimal, %non-numeric over the sample
  - all sampled problem_source in {gsm8k, augmented_gsm8k}
"""
import json, re
from collections import Counter
import numpy as np
from reliquary.environment import load_environment
from reliquary.environment.openmathinstruct import _normalize_answer

POOL = "/root/sn81-miner/data/inzone_pool_topmatch.json"

env = load_environment("openmathinstruct")
N = len(env)
tbl = env._dataset.data.table
src_col = tbl["problem_source"].cast(__import__("pyarrow").string()).combine_chunks().to_pylist()

pool = json.load(open(POOL))
total = len(pool)

# integrity of the index list itself
assert isinstance(pool, list)
all_int = all(isinstance(i, int) for i in pool)
dupes = total - len(set(pool))
mn, mx = (min(pool), max(pool)) if pool else (None, None)
all_in_range = all(0 <= i < N for i in pool)

# ---- classifier: EXACTLY build_inzone_v2.comp (uses _normalize_answer on the
# ground_truth, then raw-string checks for fraction/radical/pi/var/text). We
# additionally bucket int+decimal as numeric; everything else is non-numeric.
def classify(a):
    norm = _normalize_answer(a)
    if re.fullmatch(r"-?\d+", norm):
        return "int"
    elif re.fullmatch(r"-?\d+\.\d+", norm):
        return "decimal"
    elif "\\frac" in a or re.search(r"\b\d+/\d+\b", a):
        return "fraction"
    elif "\\sqrt" in a:
        return "radical"
    elif "\\pi" in a or re.search(r"\bpi\b", a):
        return "pi"
    elif re.search(r"[a-zA-Z]", re.sub(r"\\[a-zA-Z]+", "", a)):
        return "var/text"
    else:
        return "other"

rng = np.random.default_rng(1)
SAMPLE = 3000
samp_n = min(SAMPLE, total)
sample_idx = rng.choice(np.array(pool), size=samp_n, replace=False)

c = Counter()
src_seen = Counter()
gt_mismatch = []  # cases where get_problem ground_truth != table expected_answer
nonnum_examples = []
for i in sample_idx:
    i = int(i)
    p = env.get_problem(i)
    gt = p["ground_truth"]
    # adversarial cross-check: get_problem gt vs raw table value
    raw = tbl["expected_answer"][i].as_py()
    if str(gt) != str(raw):
        gt_mismatch.append((i, raw, gt))
    cls = classify(str(gt))
    c[cls] += 1
    src_seen[src_col[i]] += 1
    if cls not in ("int", "decimal") and len(nonnum_examples) < 25:
        nonnum_examples.append((i, gt, cls, src_col[i]))

n = samp_n
pct_int = 100.0 * c["int"] / n
pct_dec = 100.0 * c["decimal"] / n
pct_nonnum = 100.0 * (n - c["int"] - c["decimal"]) / n

bad_sources = {s for s in src_seen if s not in ("gsm8k", "augmented_gsm8k")}

print("=== INDEPENDENT VERIFICATION (recomputed from file) ===")
print(f"pool file              : {POOL}")
print(f"env len (N)            : {N}")
print(f"total idxs             : {total}")
print(f"all entries int        : {all_int}")
print(f"duplicate idxs         : {dupes}")
print(f"min/max idx            : {mn}/{mx}")
print(f"all idxs in [0,N)       : {all_in_range}")
print(f"sample size (seed=1)   : {samp_n}")
print(f"sample class counts    : {dict(c)}")
print(f"%integer               : {pct_int:.2f}%")
print(f"%decimal               : {pct_dec:.2f}%")
print(f"%non-numeric           : {pct_nonnum:.4f}%")
print(f"sampled sources        : {dict(src_seen)}")
print(f"bad sources (not gsm)  : {bad_sources}")
print(f"get_problem gt mismatch vs table: {len(gt_mismatch)}")
if gt_mismatch[:5]:
    print(f"  mismatch examples    : {gt_mismatch[:5]}")
print(f"non-numeric examples   : {nonnum_examples}")

# full-pool source check (cheap, no get_problem) for extra adversarial coverage
full_src = Counter(src_col[i] for i in pool)
full_bad = {s for s in full_src if s not in ("gsm8k", "augmented_gsm8k")}
print(f"\nFULL-POOL source counts (table, all {total}): {dict(full_src)}")
print(f"FULL-POOL bad sources  : {full_bad}")

import json as _j
print("\nRESULT_JSON " + _j.dumps({
    "total": total,
    "pct_int": round(pct_int, 4),
    "pct_dec": round(pct_dec, 4),
    "pct_nonnum": round(pct_nonnum, 6),
    "all_in_range": bool(all_in_range),
    "bad_sources_sample": sorted(bad_sources),
    "bad_sources_full": sorted(full_bad),
    "dupes": dupes,
    "gt_mismatch": len(gt_mismatch),
    "sample_n": samp_n,
}))
