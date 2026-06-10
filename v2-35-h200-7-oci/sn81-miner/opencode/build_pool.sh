#!/bin/bash
# Build the OPENCODE scatter-screened mining pool — all artifacts under opencode/.
# -----------------------------------------------------------------------------
# Wraps dataprep/build_opencode_pool.py but (a) routes ALL outputs into the opencode
# workspace (so nothing lands in the shared data/ used by the openmath modes) and
# (b) auto-resolves the VALIDATOR'S CURRENT checkpoint (the pool is checkpoint-specific:
# the model must scatter on the SAME weights the validator scores with).
#
# Usage:
#   bash /root/sn81-miner/opencode/build_pool.sh                 # defaults (1000 cand, m=16, seed 7)
#   MAXC=3000 M=16 SEED=7 bash .../build_pool.sh                 # scale up
#   bash .../build_pool.sh --strict-zone                         # pass extra flags through
#   REGRADE=opencode/data/oci_gen_cache_seed7.json bash .../build_pool.sh   # re-grade, no GPU
set -u
SN81="${SN81:-/root/sn81-miner}"
REPO="${REPO:-/root/reliquary}"
VALIDATOR_URL="${VALIDATOR_URL:-http://86.38.238.30:8080}"
OC="$SN81/opencode"

# opencode-only output dirs (build_opencode_pool.py honors these env vars)
export RELIQUARY_DATA_DIR="$OC/data"
export RELIQUARY_DIAG_DIR="$OC/diagnostics"
mkdir -p "$RELIQUARY_DATA_DIR" "$RELIQUARY_DIAG_DIR" "$OC/logs"

MAXC="${MAXC:-1000}"; M="${M:-16}"; SEED="${SEED:-7}"
LOG="$OC/logs/build_opencode.log"

# --- resolve the validator's CURRENT checkpoint to a local snapshot path ---
CKPT="${CHECKPOINT:-}"
if [ -z "$CKPT" ]; then
  STATE="$(curl -s --max-time 15 "$VALIDATOR_URL/state" 2>/dev/null || true)"
  REPO_ID="$(printf '%s' "$STATE" | python3 -c "import sys,json;print(json.load(sys.stdin).get('checkpoint_repo_id',''))" 2>/dev/null || true)"
  REV="$(printf '%s' "$STATE" | python3 -c "import sys,json;print(json.load(sys.stdin).get('checkpoint_revision',''))" 2>/dev/null || true)"
  if [ -n "$REPO_ID" ] && [ -n "$REV" ]; then
    CACHE_DIR="/root/.cache/huggingface/hub/models--${REPO_ID//\//--}/snapshots/$REV"
    [ -d "$CACHE_DIR" ] && CKPT="$CACHE_DIR"
  fi
fi
if [ -z "$CKPT" ]; then   # fallback: most-recent cached reliquary-sn-v23 snapshot
  CKPT="$(ls -dt /root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/*/ 2>/dev/null | head -1)"
fi
[ -n "$CKPT" ] && [ -e "$CKPT" ] || { echo "FATAL: could not resolve a checkpoint snapshot (set CHECKPOINT=...)"; exit 1; }
echo "=== opencode pool build | ckpt=$(basename "$CKPT") | cand=$MAXC m=$M seed=$SEED | out=$RELIQUARY_DATA_DIR ==="

cd "$REPO" || { echo "FATAL: REPO '$REPO' not found"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; source scripts/.env 2>/dev/null; set +a

ARGS=(--checkpoint "$CKPT" --cases-source reconstruct --max-candidates "$MAXC" --m "$M" --seed "$SEED")
[ -n "${REGRADE:-}" ] && ARGS+=(--regrade-from "$REGRADE")     # re-grade persisted completions, skip GPU
[ -n "${CASES_ONLY:-}" ] && ARGS+=(--cases-only)               # reconstruct cases then STOP (no GPU) — safe while mining

.venv/bin/python "$SN81/dataprep/build_opencode_pool.py" "${ARGS[@]}" "$@" 2>&1 | tee "$LOG"
echo "=== done -> pool: $RELIQUARY_DATA_DIR/inzone_pool_opencode.json | log: $LOG ==="
