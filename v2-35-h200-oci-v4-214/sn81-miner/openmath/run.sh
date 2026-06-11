#!/bin/bash
# OpenMath miner launcher — thin wrapper over ../bin/run_miner.sh.
#
# MODE defaults to `symbolic` (the #1 openmath play); pass MODE=numeric for the
# lower-variance numeric blend. Every run_miner.sh knob still applies and overrides
# the preset (see openmath/presets.sh), e.g.:
#   bash openmath/run.sh                              # symbolic
#   MODE=numeric bash openmath/run.sh                 # numeric blend
#   POOL=$SN81/data/inzone_pool_v2.json bash openmath/run.sh
#   TARGET_K=5 MAX_NEW_TOKENS=1280 bash openmath/run.sh
#   DRY_RUN=1 bash openmath/run.sh                    # print resolved config, don't launch
set -u
SN81_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env MODE="${MODE:-symbolic}" bash "$SN81_HOME/bin/run_miner.sh" "$@"
