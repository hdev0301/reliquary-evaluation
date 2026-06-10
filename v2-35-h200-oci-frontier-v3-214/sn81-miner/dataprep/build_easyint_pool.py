"""Build the 5F6VZ2-profile pool: EASY plain-integer/decimal word problems,
hard topics (matrix/radical/has-variable/fraction/pi) excluded. Requiring the
ground_truth answer to be plain_int or decimal structurally drops the hard
answer types (they can't fullmatch \d+ or \d+\.\d+). Prefer word-problem
sources (augmented_gsm8k/gsm8k); short prompts only. Writes inzone_pool_easyint.json."""
import json, re
from collections import Counter
import pyarrow as pa, pyarrow.compute as pc
import numpy as np
from reliquary.environment import load_environment

env = load_environment("openmathinstruct")
tbl = env._dataset.data.table
cols = tbl.column_names
print("columns:", cols)
ans_field = next((c for c in ("expected_answer","ground_truth","answer","final_answer") if c in cols), None)
print("answer field:", ans_field)
src  = tbl["problem_source"].cast(pa.string()).to_pylist() if "problem_source" in cols else [""]*tbl.num_rows
prob = tbl["problem"].cast(pa.string()).to_pylist() if "problem" in cols else [""]*tbl.num_rows
ans  = tbl[ans_field].cast(pa.string()).to_pylist()
N = len(ans)
print("rows:", N)
print("sources:", Counter(src).most_common(12))

def cls(a):
    a=(a or "").strip()
    core = re.sub(r"\\(text|frac|sqrt|pi|circ|begin|end|pmatrix|bmatrix|cdot|times|left|right|sin|cos|tan|log|ln|theta|alpha|beta)","",a)
    return {
        "plain_int": bool(re.fullmatch(r"-?\d+", a)),
        "decimal":   bool(re.fullmatch(r"-?\d+\.\d+", a)),
        "matrix":    "matrix" in a,
        "radical":   "\\sqrt" in a,
        "frac":      "\\frac" in a or bool(re.fullmatch(r"-?\d+/\d+", a)),
        "pi":        "\\pi" in a or re.search(r"\bpi\b", a) is not None,
        "has_var":   bool(re.search(r"[a-zA-Z]", core)),
    }
ats=[cls(a) for a in ans]
print("answer-type totals: plain_int=%d decimal=%d matrix=%d radical=%d frac=%d pi=%d has_var=%d"%(
    sum(t["plain_int"] for t in ats),sum(t["decimal"] for t in ats),sum(t["matrix"] for t in ats),
    sum(t["radical"] for t in ats),sum(t["frac"] for t in ats),sum(t["pi"] for t in ats),sum(t["has_var"] for t in ats)))

GSM={"augmented_gsm8k","gsm8k"}
easy=[]; easy_gsm=[]
for i in range(N):
    t=ats[i]
    if (t["plain_int"] or t["decimal"]) and not (t["matrix"] or t["radical"] or t["has_var"] or t["frac"] or t["pi"]) and len(prob[i] or "")<=500:
        easy.append(i)
        if src[i] in GSM: easy_gsm.append(i)
# Prefer the word-problem (gsm-family) subset if it's large enough; else all easy plain-int/dec.
chosen = easy_gsm if len(easy_gsm)>=3000 else easy
out="/root/sn81-miner/data/inzone_pool_easyint.json"
json.dump(chosen, open(out,"w"))
print("BUILT %s size=%d  (easy_gsm=%d  easy_all=%d)"%(out,len(chosen),len(easy_gsm),len(easy)))
print("chosen sources:", Counter(src[i] for i in chosen).most_common(6))
print("sample answers:", [ans[i] for i in chosen[:15]])
