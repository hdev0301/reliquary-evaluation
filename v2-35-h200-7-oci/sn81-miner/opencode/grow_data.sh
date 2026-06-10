#!/bin/bash
# Auto-grow the opencode gradeable universe WHILE the miner mines (CPU/network only — NO GPU).
# Each round reconstructs a NEW random batch of test cases (new SEED) into the ACCUMULATING cache
# (data/oci_cases_cache.json merges, never overwrites), then rebuilds the local subset so the next
# miner restart picks up the bigger, more diverse prompt set -> fewer cooldown/hash_duplicate rejects.
#
# Safe to run in the background next to a live miner; it never loads vLLM / touches the GPU.
# Run:   bash /root/sn81-miner/opencode/grow_data.sh
#        ROUNDS=20 MAXC=5000 RESTART=1 bash /root/sn81-miner/opencode/grow_data.sh
#        setsid bash /root/sn81-miner/opencode/grow_data.sh </dev/null >/dev/null 2>&1 &   # detached
set -u
SN81="${SN81:-/root/sn81-miner}"
REPO="${REPO:-/root/reliquary}"
OC="$SN81/opencode"
LOG="$OC/logs/grow_data.log"
PY="$REPO/.venv/bin/python"

ROUNDS="${ROUNDS:-8}"              # reconstruct rounds
MAXC="${MAXC:-4000}"              # candidates sampled per round (new prompts, minus small overlap)
SEED_BASE="${SEED_BASE:-100}"     # round r uses SEED_BASE+r (distinct samples)
REBUILD_EVERY="${REBUILD_EVERY:-2}"  # rebuild the local subset every K rounds (cheap, but not every round)
TARGET="${TARGET:-0}"             # stop early once the cache reaches this many entries (0 = ignore)
RESTART="${RESTART:-0}"           # 1 = restart the miner at the end to APPLY the new data

cache="$OC/data/oci_cases_cache.json"
mkdir -p "$OC/logs"
count() { [ -s "$cache" ] && "$PY" -c "import json;print(len(json.load(open('$cache'))))" 2>/dev/null || echo 0; }
rebuild_subset() { echo "    [grow] rebuilding local subset (cache=$(count)) ..." | tee -a "$LOG"; ( cd "$REPO" && "$PY" "$OC/build_local_subset.py" >> "$LOG" 2>&1 ); }

echo "=== grow_data START $(date -u +%H:%M:%S) | rounds=$ROUNDS maxc=$MAXC seed_base=$SEED_BASE target=$TARGET | cache=$(count) ===" | tee -a "$LOG"
did_rebuild=0
for r in $(seq 1 "$ROUNDS"); do
  if [ "$TARGET" -gt 0 ] && [ "$(count)" -ge "$TARGET" ]; then
    echo "    [grow] reached target cache=$(count) >= $TARGET -> stopping early" | tee -a "$LOG"; break
  fi
  seed=$((SEED_BASE + r))
  echo "--- round $r/$ROUNDS seed=$seed (cache=$(count)) $(date -u +%H:%M:%S) ---" | tee -a "$LOG"
  CASES_ONLY=1 MAXC="$MAXC" SEED="$seed" bash "$OC/build_pool.sh" >> "$LOG" 2>&1 \
    || echo "    [grow] round $r build_pool failed (see $LOG); continuing" | tee -a "$LOG"
  if [ $((r % REBUILD_EVERY)) -eq 0 ]; then rebuild_subset; did_rebuild=1; fi
done
# final rebuild if the last round(s) didn't land on a REBUILD_EVERY boundary
rebuild_subset
echo "=== grow_data DONE $(date -u +%H:%M:%S) | final cache=$(count) ===" | tee -a "$LOG"

if [ "$RESTART" = "1" ]; then
  echo "[grow] restarting miner to apply the bigger subset ..." | tee -a "$LOG"
  setsid bash "$OC/run_miner.sh" < /dev/null >> "$LOG" 2>&1 &
  echo "[grow] miner restart launched." | tee -a "$LOG"
else
  echo "[grow] NOTE: restart the miner to USE the new data:  bash $OC/run_miner.sh" | tee -a "$LOG"
fi
