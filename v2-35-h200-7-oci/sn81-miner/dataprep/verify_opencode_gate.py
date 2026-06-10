#!/usr/bin/env python3
"""OpenCode go/no-go GATE: prove we can reconstruct the validator's hidden test cases
offline, by reproducing Q's (uid170) observed case_id ground_truths.

Method: Q's live data gives real (prompt_idx, prompt, ground_truth=case_id) triples.
The public mirror (R0mAI/opencodeinstruct-prompts) maps prompt_idx -> the NVIDIA join
`id`. We stream nvidia/OpenCodeInstruct, find those ids, run the IN-REPO build pipeline
(process_row -> structured_cases) and recompute
   case_id = sha256( sha256(prompt)[:16] + canonical_json(cases) )[:16]
If recomputed case_id == Q's observed ground_truth -> reconstruction PROVEN -> GO.
"""
import hashlib, json, sys, time
sys.path.insert(0, "/root/reliquary")
sys.path.insert(0, "/root/reliquary/scripts")
import datasets as hf
from build_opencodeinstruct_subset import process_row

MIRROR, MREV = "R0mAI/opencodeinstruct-prompts", "f50bef12e244f5d51a7ae3f55ee8d31fdf33365f"
MAX_SCAN = 2_000_000

def case_id_of(prompt, structured_cases_str):
    cases = json.loads(structured_cases_str)
    pid = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    return hashlib.sha256((pid + json.dumps(cases, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()[:16]

# 1) Q triples
q = json.load(open("/root/sn81-miner/diagnostics/m_Q_5GxSiK_opencode.json"))
qrows = {}
for w in q.get("window_detail") or []:
    for s in (w.get("samples") or []):
        if s.get("prompt") and s.get("ground_truth") and isinstance(s.get("prompt_idx"), int):
            qrows[s["prompt_idx"]] = (s["prompt"], s["ground_truth"])
print(f"Q: {len(qrows)} distinct (prompt_idx -> case_id) triples", flush=True)

# 2) mirror: prompt_idx -> nvidia join id
m = hf.load_dataset(MIRROR, revision=MREV, split="train")
N = len(m)
targets = {}   # nvidia_id -> (prompt_idx, prompt, q_case_id)
for idx, (prompt, gt) in qrows.items():
    row = m[idx % N]
    targets[row["id"]] = (idx, prompt, gt)
print(f"mirror: {len(targets)} target nvidia ids to find (mirror N={N})", flush=True)

# 3) stream nvidia, reconstruct, compare
found = matched = dropped = 0
results = []
t0 = time.time()
it = hf.load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
for i, row in enumerate(it):
    if i >= MAX_SCAN or not targets:
        break
    rid = row.get("id")
    if rid not in targets:
        continue
    idx, prompt, q_gt = targets.pop(rid)
    found += 1
    proc = process_row(row)
    if proc is None:
        dropped += 1
        results.append((idx, "DROPPED_BY_FILTER", q_gt))
        continue
    rebuilt = case_id_of(proc["input"], proc["structured_cases"])
    ok = (rebuilt == q_gt)
    matched += ok
    results.append((idx, "MATCH" if ok else f"MISMATCH(got {rebuilt})", q_gt))
    if found <= 8 or ok:
        print(f"  idx={idx} id={rid[:12]} -> {'✅ MATCH' if ok else results[-1][1]}  (Q={q_gt})", flush=True)
    if found % 10 == 0:
        print(f"  ...scanned {i} rows, found {found}, matched {matched} ({time.time()-t0:.0f}s)", flush=True)

print("\n===== GATE RESULT =====", flush=True)
print(f"scanned ~{min(i+1, MAX_SCAN)} nvidia rows in {time.time()-t0:.0f}s", flush=True)
print(f"target ids found: {found}/{len(qrows)}  | case_id MATCH: {matched}  | dropped_by_filter: {dropped}  | not_found: {len(targets)}", flush=True)
if matched >= 5:
    print(f">>> VERDICT: GO — reconstruction PROVEN ({matched} validator case_ids reproduced byte-exact).", flush=True)
elif found and matched == 0:
    print(">>> VERDICT: NO-GO/INVESTIGATE — rows found but case_id never matched (build drift or revision mismatch).", flush=True)
else:
    print(">>> VERDICT: INCONCLUSIVE — too few target rows found within scan budget; raise MAX_SCAN or download fully.", flush=True)
