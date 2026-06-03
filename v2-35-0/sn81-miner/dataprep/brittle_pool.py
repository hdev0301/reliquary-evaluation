#!/usr/bin/env python3
"""Build a PARSER-BRITTLE candidate pool: augmented_math prompts whose ground_truth
has a natural alternative form that the env normalizer scores WRONG. On these the
model frequently emits a parser-rejected form -> elevated wrong-rate -> >=2 distinct
wrong rollouts -> curatable. This is a PRIOR (deterministic guess); the live screen
does the empirical wrong-rate confirmation. CPU-only, no GPU.
"""
import json, re, sys
import pyarrow.compute as pc
from datasets import Dataset, concatenate_datasets
sys.path.insert(0, "/root/reliquary")
from reliquary.environment.openmathinstruct import _normalize_answer

base = "/root/.cache/huggingface/datasets/nvidia___open_math_instruct-2/default-0f7c920d839784cc/0.0.0/469216e3f46f4dacf476b382e192485ea51a143e"
ds = concatenate_datasets([Dataset.from_file(f"{base}/open_math_instruct-2-train-0000{i}-of-00002.arrow") for i in (0, 1)])
tbl = ds.data.table
ans = tbl["expected_answer"].to_pylist()
src = tbl["problem_source"].to_pylist()
plen = pc.utf8_length(tbl["problem"]).to_pylist()
N = len(ans)


def alt_forms(g: str):
    g = str(g).strip()
    out = set()
    # degree: model drops the degree marker
    if "\\circ" in g or "°" in g or re.search(r"degree", g, re.I):
        out.add(re.sub(r"\^?\{?\\circ\}?|°|\s*degrees?", "", g, flags=re.I).strip())
    # \frac{a}{b}: model writes a/b or its decimal
    m = re.search(r"\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}", g)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b:
            out.add(f"{a}/{b}")
            d = a / b
            out.add(str(int(d)) if d == int(d) else str(round(d, 4)))
    # pi: model writes with a space / 'pi' word / decimal coefficient
    if "\\pi" in g:
        out.add(g.replace("\\pi", " pi"))
        out.add(g.replace("\\pi", "\\pi "))
    # sqrt: model sometimes writes a decimal approximation
    m2 = re.search(r"\\sqrt\{(\d+)\}", g)
    if m2:
        import math
        out.add(str(round(math.sqrt(int(m2.group(1))), 3)))
    # bare slash-fraction a/b: model writes \frac or decimal
    m3 = re.fullmatch(r"(-?\d+)/(\d+)", g)
    if m3:
        a, b = int(m3.group(1)), int(m3.group(2))
        if b:
            out.add(f"\\frac{{{a}}}{{{b}}}")
            d = a / b
            out.add(str(int(d)) if d == int(d) else str(round(d, 4)))
    return {x for x in out if x and x != g}


def is_brittle(g: str) -> bool:
    ng = _normalize_answer(g)
    return any(_normalize_answer(a) != ng for a in alt_forms(g))


from collections import Counter
pool = []
typ = Counter()
for i in range(N):
    if src[i] != "augmented_math":
        continue
    if plen[i] > 500:
        continue
    g = ans[i]
    if is_brittle(g):
        pool.append(i)
        if "\\circ" in str(g) or "°" in str(g) or "degree" in str(g).lower():
            typ["degree"] += 1
        elif "frac" in str(g) or re.fullmatch(r"-?\d+/\d+", str(g).strip()):
            typ["fraction"] += 1
        elif "\\pi" in str(g):
            typ["pi"] += 1
        elif "\\sqrt" in str(g):
            typ["sqrt"] += 1
        else:
            typ["other"] += 1

json.dump(pool, open("/root/brittle_pool.json", "w"))
am = sum(1 for s in src if s == "augmented_math")
print(f"brittle pool: {len(pool)} / {am} augmented_math ({100*len(pool)/am:.0f}%) -> /root/brittle_pool.json")
print(f"type breakdown: {dict(typ)}")
import random
random.seed(0)
print("samples:")
for i in random.sample(pool, min(10, len(pool))):
    print("  ", repr(ans[i]))
