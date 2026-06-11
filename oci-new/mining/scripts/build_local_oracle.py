"""Build the local OpenCode frontier oracle (one-time, offline).

Reconstructs the validator's hidden structured cases from the **public**
``nvidia/OpenCodeInstruct`` ``unit_tests`` column using the exact same
``structure_tests`` logic the validator's subset builder uses
(``scripts/build_opencodeinstruct_subset.py``), keyed by ``sha256(prompt)`` so
the miner can look them up at runtime without depending on dataset row order or
revision.

Only the prompts that appear in the public prompt mirror
(``R0mAI/opencodeinstruct-prompts``) are needed, so we restrict to those ``id``s
and skip the rest of nvidia's 5M rows cheaply — the expensive structuring runs
on ~50k rows.

Usage:
    python -m mining.scripts.build_local_oracle \
        --mirror R0mAI/opencodeinstruct-prompts \
        --source nvidia/OpenCodeInstruct \
        --out mining/state/opencode_oracle.json.gz \
        [--verify-determinism]   # slower; double-exec the reference like the validator

Output: gzip JSON ``{"meta": {...}, "by_sha": {sha256(input): [cases...]}}``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Make the repo root importable so we can reuse the validator's own builder.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    from scripts.build_opencodeinstruct_subset import (
        double_execute, extract_reference_code, parse_unit_tests, structure_tests,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default="R0mAI/opencodeinstruct-prompts")
    ap.add_argument("--mirror-revision", default="f50bef12e244f5d51a7ae3f55ee8d31fdf33365f")
    ap.add_argument("--source", default="nvidia/OpenCodeInstruct")
    ap.add_argument("--out", default="mining/state/opencode_oracle.json.gz")
    ap.add_argument("--verify-determinism", action="store_true",
                    help="Double-execute the reference solution against the cases "
                         "(matches the validator subset filter exactly; much slower).")
    ap.add_argument("--max-source-rows", type=int, default=None,
                    help="Cap on source rows scanned (debug).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    import datasets as hf

    logger.info("loading prompt mirror %s ...", args.mirror)
    mirror = hf.load_dataset(args.mirror, split="train", revision=args.mirror_revision)
    needed_ids = {str(r) for r in mirror["id"]}
    # sha by id so we can key output on the *mirror's* prompt text (canonical).
    sha_by_id = {str(i): _sha(inp) for i, inp in zip(mirror["id"], mirror["input"])}
    logger.info("mirror rows: %d (distinct ids: %d)", len(mirror), len(needed_ids))

    by_sha: dict[str, list] = {}
    src = hf.load_dataset(args.source, split="train", streaming=True)
    scanned = kept = 0
    for scanned, row in enumerate(src):
        if args.max_source_rows is not None and scanned >= args.max_source_rows:
            break
        rid = str(row.get("id", ""))
        if rid not in needed_ids or rid in ("",):
            if scanned % 100000 == 0:
                logger.info("scanned=%d kept=%d", scanned, kept)
            continue
        tests = parse_unit_tests(row.get("unit_tests", ""))
        if not tests:
            continue
        cases = structure_tests(tests)
        if not cases:
            continue
        if args.verify_determinism:
            code = extract_reference_code(row.get("output", ""))
            if not double_execute(code, cases):
                continue
        # Key on the MIRROR's input text (what the miner will sha at runtime).
        sha = sha_by_id.get(rid) or _sha(row.get("input", ""))
        by_sha[sha] = cases
        kept += 1
        if kept % 1000 == 0:
            logger.info("scanned=%d kept=%d (coverage=%.1f%%)",
                        scanned, kept, 100.0 * kept / max(1, len(needed_ids)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = {
        "meta": {
            "source": args.source,
            "mirror": args.mirror,
            "mirror_revision": args.mirror_revision,
            "verified_determinism": args.verify_determinism,
            "prompts": len(by_sha),
            "mirror_size": len(mirror),
        },
        "by_sha": by_sha,
    }
    with gzip.open(args.out, "wt") as fh:
        json.dump(blob, fh)
    logger.info(
        "wrote %s — %d/%d mirror prompts with cases (%.1f%% coverage)",
        args.out, len(by_sha), len(needed_ids),
        100.0 * len(by_sha) / max(1, len(needed_ids)),
    )


if __name__ == "__main__":
    main()
