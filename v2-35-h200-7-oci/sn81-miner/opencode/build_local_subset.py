#!/usr/bin/env python3
"""Build a LOCAL opencode structured subset the miner can grade against.

The honest opencode miner needs structured_cases to compute reward (screen/curate
in-zone groups). The validator's structured subset (R0mAI/opencodeinstruct-structured-subset)
is HF-gated (401 without a token), and prompt-only mode has NO cases -> reward is always 0.

This attaches our RECONSTRUCTED cases (data/oci_cases_cache.json, built byte-exact from the
PUBLIC nvidia/OpenCodeInstruct) onto the FULL prompt mirror, preserving row order & length so
prompt_idx i maps to the SAME problem the validator grades for i (alignment confirmed by audit).
Rows without a reconstructed case get an empty cases list (reward 0 -> just never in-zone).

Output: opencode/data/oci_local_subset (HF save_to_disk). Point the miner at it with
  RELIQUARY_OCI_PROMPT_ONLY=0  RELIQUARY_OCI_SUBSET_REPO=<this path>
and run the local grader server (opencode/grader.sh) so compute_reward scores locally.
"""
import json
import os
import sys

sys.path.insert(0, "/root/reliquary")
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment  # noqa: E402

OC = "/root/sn81-miner/opencode"
CACHE = os.path.join(OC, "data", "oci_cases_cache.json")
OUT = os.path.join(OC, "data", "oci_local_subset")


def main():
    import datasets as hf

    repo = OpenCodeInstructEnvironment._DEFAULT_PROMPT_REPO
    rev = OpenCodeInstructEnvironment._DEFAULT_PROMPT_REVISION
    print(f"loading prompt mirror {repo}@{rev[:8]} ...")
    mirror = hf.load_dataset(repo, revision=rev, split="train")
    n = len(mirror)
    print(f"mirror rows = {n}")

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    print(f"reconstructed case-sets available: {sum(1 for v in cache.values() if v and v.get('cases'))}")

    # Build the structured_cases column DETERMINISTICALLY, NOT via datasets.map(): map() caches on
    # the (function + input-dataset) fingerprint and does NOT see changes to the external `cache`
    # dict, so after grow_data.sh adds new cases a map() rebuild would silently serve the STALE
    # subset (and skip the counter -> the misleading "0"). add_column re-runs every time.
    # structured_cases is a JSON string per row (matches the validator subset; _row_cases handles
    # str and list). Empty "[]" where we have no reconstructed cases.
    have = 0
    col = []
    for i in range(len(mirror)):
        ent = cache.get(str(i))
        cases = ent["cases"] if (ent and ent.get("cases")) else []
        if cases:
            have += 1
        col.append(json.dumps(cases))
    if "structured_cases" in mirror.column_names:
        mirror = mirror.remove_columns(["structured_cases"])
    ds = mirror.add_column("structured_cases", col)

    # sanity: the env requires the column and reads row["input"]
    assert "structured_cases" in ds.column_names, "structured_cases column missing"
    assert "input" in ds.column_names, "input column missing (prompt)"

    # Atomic-ish write: save to a temp dir then swap, so a miner restart during a rebuild
    # never load_from_disk's a half-written subset (the RUNNING miner is unaffected — it holds
    # its copy in RAM and only re-reads on boot).
    import shutil
    tmp = OUT + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds.save_to_disk(tmp)
    shutil.rmtree(OUT + ".old", ignore_errors=True)
    if os.path.exists(OUT):
        os.rename(OUT, OUT + ".old")
    os.rename(tmp, OUT)
    shutil.rmtree(OUT + ".old", ignore_errors=True)
    print(f"\nwrote local subset: {OUT}")
    print(f"  rows={len(ds)} (index-aligned with mirror)  rows_with_cases={have}")
    print(f"\nMINE IT:  RELIQUARY_OCI_PROMPT_ONLY=0  RELIQUARY_OCI_SUBSET_REPO={OUT}")
    print(f"          (start the local grader first: bash {OC}/grader.sh start)")


if __name__ == "__main__":
    main()
