#!/bin/bash
# harvest_winners_loop.sh — periodic FRONTIER-SEED re-harvest (detached watchdog).
#
# Every HARVEST_INTERVAL seconds, snapshot the top OMI miner's recent in-zone winners into the
# frontier-seed file (RELIQUARY_WINNERS_PATH) so a (re)started miner seeds from CURRENT-checkpoint
# exemplars rather than a stale post-checkpoint-flip set. Runs on the box independently of the
# miner (like run_miner_random.sh's supervisor). The atomic write means the miner can read the
# file any time without seeing a partial.
#
# NOTE: the live miner loads the seed only at startup (pregen init), so this keeps the file fresh
# for the NEXT restart; between restarts the online frontier model adapts from live outcomes.
#
#   knobs (env):  HARVEST_INTERVAL=1800   # seconds between harvests (default 30 min)
#                 HARVEST_HOTKEYS=<ss58>[,<ss58>...]   # who to harvest (default top OMI miner)
#                 RELIQUARY_WINNERS_PATH=<file>        # seed file (default data/topminer_winners.jsonl)
#   logs:  $SN81/logs/harvest_winners.log     stop:  kill "$(cat $SN81/harvest_winners.pid)"
set -u
SN81="${SN81:-/root/sn81-miner}"
REPO="${REPO:-/root/reliquary}"
INTERVAL="${HARVEST_INTERVAL:-1800}"
OUT="${RELIQUARY_WINNERS_PATH:-$SN81/data/topminer_winners.jsonl}"
LOG="$SN81/logs/harvest_winners.log"
PIDF="$SN81/harvest_winners.pid"
export RELIQUARY_WINNERS_PATH="$OUT"
mkdir -p "$SN81/logs" "$(dirname "$OUT")"

# ----- detached loop body (re-invoked via --loop; its internal sleeps run detached) -----
if [[ "${1:-}" == "--loop" ]]; then
  echo "[$(date '+%F %T')] harvest loop START: every ${INTERVAL}s -> $OUT (hotkeys=${HARVEST_HOTKEYS:-default})"
  while true; do
    if "$REPO/.venv/bin/python" "$SN81/dataprep/harvest_winners.py" 2>&1; then :; else
      echo "[$(date '+%F %T')] harvest attempt failed (kept previous seed)"
    fi
    echo "[$(date '+%F %T')] next harvest in ${INTERVAL}s"
    sleep "$INTERVAL"
  done
  exit 0
fi

# ----- launcher: stop any prior loop (by PID, never by -f so we don't match this shell) -----
[[ -f "$PIDF" ]] && kill -9 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null || true
echo "==== harvest watchdog (re)start $(date '+%F %T') ====" >> "$LOG"
nohup bash "$SN81/bin/harvest_winners_loop.sh" --loop >> "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "harvest-seed watchdog PID=$(cat "$PIDF") | every ${INTERVAL}s -> $OUT | log=$LOG"
echo "  stop: kill \$(cat $PIDF)   |   hotkeys=${HARVEST_HOTKEYS:-<top OMI miner default>}"
