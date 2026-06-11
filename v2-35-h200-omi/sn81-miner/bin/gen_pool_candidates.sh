#!/usr/bin/env bash
# gen_pool_candidates.sh — generate HARDEST-BAND inzone_pool candidates for the
# rotation watchdog (run_miner_random.sh) to mine. Supports a CONTINUOUS background
# loop (LOOP=1) that keeps producing fresh seeded variants up to a cap.
#
# STRONG-MODEL REGIME: the published checkpoint is strong/converged, so it scores
# 8/8 on easy problems (gsm8k decimals/ints) -> bimodal -> out of zone -> ~no accepts.
# In-zone variance (sigma>=0.43) survives ONLY on problems the model gets right only
# SOMETIMES -> the hardest math. So every candidate is --math-only (the easy-decimal
# numeric base is dropped). We sweep the HARDNESS axes:
#   * math-split-only : `math` competition split only (hardest) vs +augmented_math
#   * max-prompt-len  : longer = harder multi-step problems
#   * canon-keep-frac : seal-race low-sha256 slice (anti-batch_filled) vs off
#
#   tag = m<split>_l<len>_c<canon%>[ _g<seed> ]   e.g. inzone_pool_cand_m1_l700_c30.json
#
# ACCEPT ATTRIBUTION: the watchdog logs every validator-ACCEPTED submission with its
# source pool to logs/pool_accepts.log. This script surfaces that tally each cycle.
#
# Usage:
#   nohup LOOP=1 bash bin/gen_pool_candidates.sh > logs/gen_candidates.log 2>&1 &   # continuous background
#   bash bin/gen_pool_candidates.sh                                                 # one-shot
#   LOOP=1 MAX_LENS="600 900 1200" MATH_SPLITS="1" bash bin/gen_pool_candidates.sh  # custom
set -u
SN81=/root/sn81-miner
REPO=/root/reliquary
DATA="$SN81/data"
PY="$REPO/.venv/bin/python"
BUILD="$SN81/dataprep/build_inzone_v2.py"

# --- HARDEST-band sweep grid (override via env) ---
MATH_SPLITS="${MATH_SPLITS:-0 1}"          # 1 = `math` competition split only (hardest); 0 = +augmented_math
MAX_LENS="${MAX_LENS:-400 700}"            # problem char cap; longer = harder multi-step
CANON_FRACS="${CANON_FRACS:-1.00 0.30}"    # 1.0 = off; 0.30 = low-sha256 seal-race slice (anti-batch_filled)
INT_RATIO="${INT_RATIO:-0.0}"             # keep 0 for pure hard-math (ints are bimodal on a strong model)
FORCE="${FORCE:-0}"

# --- continuous background loop ---
LOOP="${LOOP:-0}"
LOOP_INTERVAL="${LOOP_INTERVAL:-1800}"
MAX_CANDIDATES="${MAX_CANDIDATES:-18}"
SEED_START="${SEED_START:-0}"
ACCEPT_LOG="${ACCEPT_LOG:-$SN81/logs/pool_accepts.log}"

cd "$REPO" || { echo "FATAL: $REPO missing"; exit 1; }
[ -x "$PY" ] || { echo "FATAL: venv python missing ($PY) — run setup.sh first"; exit 1; }
set -a; source "$REPO/scripts/.env"; set +a
export RELIQUARY_OMI_SHARDS="${RELIQUARY_OMI_SHARDS:-4}"   # match the validator's 4-shard space (2x prompts; shards 0-1 idxs unchanged, + shards 2-3)
mkdir -p "$DATA" "$SN81/logs"

pct() { awk -v x="$1" 'BEGIN{printf "%02d", x*100 + 0.5}'; }
cand_count() { ls -1 "$DATA"/inzone_pool_cand_*.json 2>/dev/null | wc -l; }

summarize_accepts() {
  if [ -s "$ACCEPT_LOG" ]; then
    echo "  --- accept->pool tally (from $(basename "$ACCEPT_LOG")) ---"
    grep -oE "pool=[^ ]+" "$ACCEPT_LOG" | sort | uniq -c | sort -rn | sed 's/^/    /'
  else
    echo "  (no validator ACCEPTs logged yet -> $ACCEPT_LOG is empty)"
  fi
}

build_grid() {          # one full HARDEST-band grid pass for a given seed
  local seed="$1" ms ml cf tag out sz st split_arg
  st=""; [ "$seed" != "0" ] && st="_g${seed}"
  for ms in $MATH_SPLITS; do for ml in $MAX_LENS; do for cf in $CANON_FRACS; do
    if [ "$(cand_count)" -ge "$MAX_CANDIDATES" ]; then echo "  reached cap ($MAX_CANDIDATES) — stop this cycle"; return; fi
    split_arg=""; [ "$ms" = "1" ] && split_arg="--math-split-only"
    tag="m${ms}_l${ml}_c$(pct "$cf")${st}"
    out="$DATA/inzone_pool_cand_${tag}.json"
    if [ -s "$out" ] && [ "$FORCE" != "1" ]; then echo "  skip (exists): $(basename "$out")"; continue; fi
    if "$PY" "$BUILD" --math-only $split_arg --int-ratio "$INT_RATIO" --canon-keep-frac "$cf" --seed "$seed" \
          --max-prompt-len "$ml" --out "$out" >/dev/null 2>"$DATA/.cand_build_err"; then
      sz=$("$PY" -c "import json;print(len(json.load(open('$out'))))" 2>/dev/null || echo '?')
      echo "  built $(basename "$out")  size=$sz idxs  (math_split=$ms max_len=$ml canon=$cf)"
    else
      echo "  FAILED $(basename "$out") — last lines:"; tail -3 "$DATA/.cand_build_err" | sed 's/^/    /'
    fi
  done; done; done
}

echo "=== HARDEST-band generator | LOOP=$LOOP interval=${LOOP_INTERVAL}s cap=$MAX_CANDIDATES | math_split={$MATH_SPLITS} max_len={$MAX_LENS} canon={$CANON_FRACS} int=$INT_RATIO OMI_SHARDS=$RELIQUARY_OMI_SHARDS ==="
if [ "$LOOP" = "1" ]; then
  seed="$SEED_START"
  while true; do
    echo "[$(date '+%F %T')] cycle seed=$seed  ($(cand_count)/$MAX_CANDIDATES candidates on disk)"
    if [ "$(cand_count)" -ge "$MAX_CANDIDATES" ]; then
      echo "  at cap — waiting for the watchdog to prune losers before generating more"
    else
      build_grid "$seed"; seed=$((seed + 1))
    fi
    summarize_accepts
    echo "[$(date '+%F %T')] sleeping ${LOOP_INTERVAL}s ..."
    sleep "$LOOP_INTERVAL"
  done
else
  build_grid "$SEED_START"
  echo "=== done (one-shot): $(cand_count) candidate pool(s) under $DATA ==="
  summarize_accepts
fi
