#!/bin/bash
# build_frontier_pool.sh — build the FRONTIER-TARGETED OMI candidate pool.
#
# Method: select HARD OMI sources (augmented_math + math) with SHORT, NUMERIC
# answers — the band where a converged Qwen3.5-4B checkpoint still sits at its
# learning frontier (genuine 20-80% pass mass), so the live two-stage screen +
# curation can build in-zone (sigma>=0.43) groups. The easy gsm8k sources are
# solved ~100% by v23 and yield zero curatable mass. See the python module
# header (dataprep/build_frontier_pool.py) for the full rationale.
#
# Pins RELIQUARY_OMI_SHARDS=4 so written idxs match the validator's index space
# (idx % len(dataset)); building with a different shard count -> PROMPT_MISMATCH.
#
#   bash bin/build_frontier_pool.sh                 # defaults -> data/inzone_pool_frontier.json
#   FRONTIER_MAX_CHARS=300 bash bin/build_frontier_pool.sh        # tighter/shorter
#   FRONTIER_SOURCES=augmented_math bash bin/build_frontier_pool.sh  # one source
#   FRONTIER_NUMERIC_ONLY=0 bash bin/build_frontier_pool.sh       # keep symbolic answers too
set -eu

SN81="${SN81:-/root/sn81-miner}"
REPO="${REPO:-/root/reliquary}"
export RELIQUARY_OMI_SHARDS=4   # MUST match the validator's shard count for index correctness

cd "$REPO" || { echo "FATAL: REPO '$REPO' not found"; exit 1; }
# Reuse the miner's HF_TOKEN / cache settings if present (higher HF rate limits).
[ -f "$REPO/scripts/.env" ] && { set -a; . "$REPO/scripts/.env"; set +a; }

exec "$REPO/.venv/bin/python" "$SN81/dataprep/build_frontier_pool.py" "$@"
