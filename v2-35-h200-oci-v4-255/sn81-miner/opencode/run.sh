#!/bin/bash
# OpenCode miner launcher — thin wrapper over ../bin/run_miner.sh with MODE=opencode.
#
# Every run_miner.sh knob still applies and overrides the opencode preset, e.g.:
#   POOL=$SN81/data/inzone_pool_opencode.json bash opencode/run.sh   # curated pool
#   CURATE=1 USE_GRADER=1 bash opencode/run.sh                       # local-graded curation
#   MAX_NEW_TOKENS=1280 TARGET_K=4 bash opencode/run.sh
#
# Defaults (set in run_miner.sh's `opencode` case): ENVIRONMENT=opencodeinstruct,
# OCI_PROMPT_ONLY=1, CURATE=0 (blind-submit, validator grades), broad sampling.
set -u
SN81_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env MODE=opencode bash "$SN81_HOME/bin/run_miner.sh" "$@"
