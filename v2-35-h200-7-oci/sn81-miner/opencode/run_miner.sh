#!/bin/bash
# Reliquary OPENCODE miner launcher — thin wrapper over run_miner.sh (MODE=opencode).
# ---------------------------------------------------------------------------------
# WHY a separate launcher (vs the math modes in run_miner.sh):
#   * opencode is a DIFFERENT environment: nvidia/OpenCodeInstruct, reward is
#     VALIDATOR-AUTHORITATIVE = passed/total over HIDDEN structured cases. The miner
#     runs RELIQUARY_OCI_PROMPT_ONLY=1 and CANNOT grade locally, so runtime curation
#     against the env reward is a no-op -> we mine HONEST (CURATE=0).
#   * To still land in-zone (k correct + (8-k) wrong, sigma>=0.43) we restrict mining
#     to a SCATTER-SCREENED prompt pool built by dataprep/build_opencode_pool.py:
#     prompts whose pass-fraction is intermediate (not 0/M, not M/M) under the
#     validator's CURRENT checkpoint -> natural in-zone variance across 8 rollouts
#     without any curation. Broad sampling the raw 50k wastes windows on 0/8 & 8/8.
#
# CHECKPOINT NOTE: the miner ignores --checkpoint and loads the validator's published
#   checkpoint from /state (currently R0mAI/reliquary-sn-v23, hot-reloaded as the
#   frontier advances). The scatter pool is therefore CHECKPOINT-SPECIFIC: rebuild it
#   whenever ckpt_n advances:
#     cd /root/reliquary && .venv/bin/python /root/sn81-miner/dataprep/build_opencode_pool.py \
#       --checkpoint <current local snapshot path> --cases-source reconstruct --m 16
#
# Usage:  bash /root/sn81-miner/opencode/run_miner.sh
#         POOL="" bash opencode/run_miner.sh      # force broad sampling
#         PREDICT_BLIND=0 bash opencode/run_miner.sh   # disable predictive seal fire
#
# All opencode state (pool, hot-pool, burned blocklist) lives under opencode/data/ so it
# never mixes with the openmath modes' shared data/. Build the pool: opencode/build_pool.sh
set -u
SN81="${SN81:-/root/sn81-miner}"
OCDATA="$SN81/opencode/data"
mkdir -p "$OCDATA"

# ============================ opencode wiring (overridable) ============================
export MODE=opencode
export ENVIRONMENT="${ENVIRONMENT:-opencodeinstruct}"
export CURATE="${CURATE:-0}"                                  # honest: reward is validator-authoritative

# REAL REWARD PATH: prompt-only mode has no test cases -> compute_reward()==0 -> 0 in-zone groups.
# Instead load a LOCAL structured subset (mirror + reconstructed cases, index-aligned) and route
# compute_reward() through a local grader server (exact validator parity, non-gVisor). Set
# OCI_PROMPT_ONLY=1 to revert to the (non-producing) prompt-only mode.
export OCI_PROMPT_ONLY="${OCI_PROMPT_ONLY:-0}"
LOCAL_SUBSET="${OCI_SUBSET_REPO:-$SN81/opencode/data/oci_local_subset}"
if [ "$OCI_PROMPT_ONLY" = "0" ]; then
  [ -d "$LOCAL_SUBSET" ] || { echo "FATAL: local subset '$LOCAL_SUBSET' missing — build it: .venv/bin/python $SN81/opencode/build_local_subset.py"; exit 1; }
  export RELIQUARY_OCI_SUBSET_REPO="$LOCAL_SUBSET"
  bash "$SN81/opencode/grader.sh" start || { echo "FATAL: local grader failed to start"; exit 1; }
  echo "[opencode] real-reward mode: subset=$LOCAL_SUBSET + local grader"
fi
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1280}"              # code completions: med ~330, max ~1190
export SCREEN_MAX_TOKENS="${SCREEN_MAX_TOKENS:-1280}"
export TARGET_K="${TARGET_K:-4}"
export GPU_MEM="${GPU_MEM:-0.65}"
export POOL_SIZE="${POOL_SIZE:-48}"
export GEN_BATCH="${GEN_BATCH:-24}"

# Keep opencode runtime state OUT of the shared data/ (openmath modes use that).
export HOT_POOL="${HOT_POOL:-$OCDATA/hot_pool.json}"          # opencode-only screen-proven cache
export BURNED_PATH="${BURNED_PATH:-$OCDATA/submitted_idx.json}"  # opencode-only anti-dup blocklist

# Auto-wire the scatter pool: use it once built, else fall back to BROAD sampling so this
# launcher works before the pool exists (and never silently mines gsm8k via the loader fallback).
DEFAULT_POOL="$OCDATA/inzone_pool_opencode.json"
if [ -n "${POOL:-}" ]; then
  export POOL                                                # explicit user override (path or "")
elif [ -s "$DEFAULT_POOL" ]; then
  export POOL="$DEFAULT_POOL"
  N=$(/root/reliquary/.venv/bin/python -c "import json;print(len(json.load(open('$DEFAULT_POOL'))))" 2>/dev/null || echo "?")
  echo "[opencode] scatter-screened pool: $DEFAULT_POOL ($N idxs)"
else
  export POOL=""
  echo "[opencode] scatter pool not built yet -> BROAD sampling over the 50k subset"
  echo "[opencode]   build it: bash $SN81/opencode/build_pool.sh"
fi

# ============================ seal-race timing (env-agnostic) ============================
# THE path to real `verdict ACCEPTED`. Without this we arrive at over=0 and lose batch_filled
# to miners who fire PREDICTIVELY (over<0), filling the 8 shared seal slots before the trigger.
# PREDICT_BLIND=1 pre-builds and fires into the predicted post-boundary window. MIN_WINDOWS=2
# because we only see ~3 windows / 15 min, so the engine default (3) never learns window-length L.
# Toggle off:  PREDICT_BLIND=0 bash run_miner_opencode.sh
export RELIQUARY_PREDICT_BLIND="${PREDICT_BLIND:-1}"
export RELIQUARY_PREDICT_MIN_WINDOWS="${PREDICT_MIN_WINDOWS:-2}"
export RELIQUARY_PREDICT_LEAD_MS="${PREDICT_LEAD_MS:-1000}"
export RELIQUARY_PREDICT_POST_MS="${PREDICT_POST_MS:-300}"

echo "=== OPENCODE launcher | pool=${POOL:-<broad>} | predict_blind=$RELIQUARY_PREDICT_BLIND (min_windows=$RELIQUARY_PREDICT_MIN_WINDOWS lead=${RELIQUARY_PREDICT_LEAD_MS}ms) ==="
exec bash "$SN81/bin/run_miner.sh"
