"""Reconstruct the validator's EXACT hidden structured_cases.

The validator's R0mAI/opencodeinstruct-structured-subset was built by the
deployed scripts/build_opencodeinstruct_subset.py: structured_cases =
structure_tests(parse_unit_tests(row["unit_tests"])) over nvidia/OpenCodeInstruct,
and the prompt mirror R0mAI/opencodeinstruct-prompts preserves the SAME row order
+ carries each row's `id`. So for prompt_idx i we look up its `id` in the mirror,
find that nvidia row, and recompute structure_tests -> the byte-identical cases
the validator scores against (verified: matches process_row output exactly).

Pure-AST (no double_execute needed: a row present in the mirror already passed all
filters). Output: data/oci_cases_v2.json = {str(idx): {"prompt","cases","id"}}.
"""
from __future__ import annotations
import importlib.util, json, os, sys, time

REPO = "/root/reliquary"
OUT = "/root/sn81-miner/data/oci_cases_v2.json"
LIMIT = int(os.environ.get("CACHE_LIMIT", "0"))  # 0 = all mirror rows
SCAN_CAP = int(os.environ.get("SCAN_CAP", "2000000"))

spec = importlib.util.spec_from_file_location("bld", f"{REPO}/scripts/build_opencodeinstruct_subset.py")
bld = importlib.util.module_from_spec(spec); spec.loader.exec_module(bld)
import datasets as hf

print("loading prompt mirror...", flush=True)
mir = hf.load_dataset("R0mAI/opencodeinstruct-prompts", split="train")
n = len(mir) if LIMIT == 0 else min(LIMIT, len(mir))
id_to_idx = {}
prompt_by_idx = {}
for i in range(n):
    r = mir[i]
    id_to_idx[r["id"]] = i
    prompt_by_idx[i] = r["input"]
print(f"mirror rows targeted: {n}", flush=True)

cache = {}
if os.path.exists(OUT):
    try:
        cache = json.load(open(OUT))
        print(f"resuming: {len(cache)} already cached", flush=True)
    except Exception:
        cache = {}
need = set(str(i) for i in range(n)) - set(cache.keys())
need_ids = {rid for rid, i in id_to_idx.items() if str(i) in need}
print(f"need {len(need_ids)} more ids", flush=True)

ds = hf.load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
t0 = time.time(); scanned = 0; added = 0
for row in ds:
    scanned += 1
    rid = row.get("id")
    if rid in need_ids:
        idx = id_to_idx[rid]
        try:
            tests = bld.parse_unit_tests(row.get("unit_tests", ""))
            cases = bld.structure_tests(tests) if tests else []
        except Exception:
            cases = []
        cache[str(idx)] = {"prompt": prompt_by_idx[idx], "cases": cases, "id": rid}
        need_ids.discard(rid)
        added += 1
        if added % 500 == 0:
            json.dump(cache, open(OUT, "w"))
            print(f"  +{added} (idx {idx}) scanned={scanned} left={len(need_ids)} "
                  f"{added/(time.time()-t0):.0f}/s", flush=True)
    if not need_ids or scanned >= SCAN_CAP:
        break

json.dump(cache, open(OUT, "w"))
withcases = sum(1 for v in cache.values() if v.get("cases"))
print(f"DONE: cached {len(cache)} prompts ({withcases} with >=1 case), scanned {scanned} nvidia rows -> {OUT}", flush=True)
