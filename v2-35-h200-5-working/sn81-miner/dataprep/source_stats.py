#!/usr/bin/env python3
"""Per-source profile of the loaded OpenMathInstruct-2 shards: count, problem
char-length, REAL prompt token-length (chat template + boxed instruction, exactly
how the miner/validator encode), expected_answer length, and answer-format mix.

Run on the box (the dataset + model snapshot live under ~/.cache):
    python source_stats.py [sample_per_source]   # default 40000

Char stats are computed over ALL rows (cheap, via pyarrow). Token stats are
computed over a fixed-seed sample per source (tokenizing 873k prompts is slow).
"""
import os, re, sys, json, statistics as st
from collections import Counter, defaultdict
import pyarrow.compute as pc
from datasets import Dataset, concatenate_datasets

sys.path.insert(0, "/root/reliquary")
from reliquary.environment.openmathinstruct import _ANSWER_FORMAT_INSTRUCTION
from reliquary.protocol.tokens import encode_prompt

# Same 2 shards the env loads by default (RELIQUARY_OMI_SHARDS=2).
BASE = "/root/.cache/huggingface/datasets/nvidia___open_math_instruct-2/default-0f7c920d839784cc/0.0.0/469216e3f46f4dacf476b382e192485ea51a143e"
SNAP = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/7926d852f1d955f44443fac1476681e0e0fdde92/"
SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 40000

ds = concatenate_datasets(
    [Dataset.from_file(f"{BASE}/open_math_instruct-2-train-0000{i}-of-00002.arrow") for i in (0, 1)]
)
tbl = ds.data.table
src = tbl["problem_source"].to_pylist()
prob = tbl["problem"].to_pylist()
ans = [str(a) for a in tbl["expected_answer"].to_pylist()]
plen = pc.utf8_length(tbl["problem"]).to_pylist()
alen = [len(a) for a in ans]
N = len(src)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SNAP)


def fmt(a: str) -> str:
    a = a.strip()
    if re.fullmatch(r"-?\d+", a):            return "plain_int"
    if re.fullmatch(r"-?\d+\.\d+", a):       return "decimal"
    if "\\frac" in a or re.fullmatch(r"-?\d+/\d+", a): return "fraction"
    if "\\sqrt" in a:                        return "radical"
    if "\\pi" in a or re.search(r"\bpi\b", a): return "pi"
    if "matrix" in a:                        return "matrix"
    if re.search(r"[a-zA-Z]", re.sub(r"\\[a-zA-Z]+", "", a)): return "has_var"
    return "other"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


by = defaultdict(list)
for i in range(N):
    by[src[i]].append(i)

import random
random.seed(0)
print(f"loaded {N} rows from 2 shards; tokenizing up to {SAMPLE}/source\n")
hdr = f"{'source':16} {'count':>8} {'share':>6} | {'prob_chars med/p90/max':>24} | {'PROMPT_TOKENS med/p90/max':>26} | {'ans_chars med/max':>14}"
print(hdr); print("-" * len(hdr))
rows_out = {}
for s in ("augmented_math", "math", "augmented_gsm8k", "gsm8k"):
    idxs = by.get(s, [])
    n = len(idxs)
    if not n:
        continue
    pc_med, pc_p90, pc_max = pct([plen[i] for i in idxs], 50), pct([plen[i] for i in idxs], 90), max(plen[i] for i in idxs)
    ac_med, ac_max = pct([alen[i] for i in idxs], 50), max(alen[i] for i in idxs)
    samp = random.sample(idxs, min(SAMPLE, n))
    toklens = [len(encode_prompt(tok, prob[i] + _ANSWER_FORMAT_INSTRUCTION)) for i in samp]
    t_med, t_p90, t_max = pct(toklens, 50), pct(toklens, 90), max(toklens)
    fmts = Counter(fmt(ans[i]) for i in idxs)
    print(f"{s:16} {n:>8} {n/N:>6.1%} | {pc_med:>6}/{pc_p90:>5}/{pc_max:>6}      | {t_med:>7}/{t_p90:>6}/{t_max:>6}         | {ac_med:>5}/{ac_max:>6}")
    rows_out[s] = {"count": n, "share": n / N, "prob_chars": [pc_med, pc_p90, pc_max],
                   "prompt_tokens": [t_med, t_p90, t_max], "ans_chars": [ac_med, ac_max],
                   "answer_formats": {k: round(v / n, 4) for k, v in fmts.most_common()}}

print("\nanswer-format mix per source (fraction of rows):")
for s, d in rows_out.items():
    print(f"  {s:16} " + "  ".join(f"{k}={v:.0%}" for k, v in d["answer_formats"].items()))

json.dump(rows_out, open("/root/source_stats.json", "w"), indent=2)
print("\nwrote /root/source_stats.json")
