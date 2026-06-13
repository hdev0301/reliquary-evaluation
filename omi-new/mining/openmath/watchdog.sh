#!/bin/bash
# Watchdog for the OpenMath miner (sibling of mining/opencode/watchdog.sh):
#   1. keeps it alive — restarts via run.sh if the process dies OR the log goes
#      stale (hung worker), and
#   2. alerts on every FIRE and every validator verdict, appended to a dedicated
#      notifications file you can tail/open.
#
# Usage (from repo root, after run.sh has launched the miner once):
#     source mining/openmath/.env
#     nohup bash mining/openmath/watchdog.sh > /root/openmath_watchdog.log 2>&1 &
#
# Tunables (env): RELIQUARY_WD_INTERVAL (poll seconds, default 30),
#   RELIQUARY_WD_STALE (restart if log idle this many seconds, default 600),
#   RELIQUARY_NOTIFY_LOG (default /root/openmath_notifications.log).
set -uo pipefail

REPO_DIR="${RELIQUARY_INSTALL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MINER_LOG="${RELIQUARY_MINER_LOG:-/root/openmath_miner.log}"
PID_FILE="${RELIQUARY_MINER_PID:-/root/openmath_miner.pid}"
NOTIFY_LOG="${RELIQUARY_NOTIFY_LOG:-/root/openmath_notifications.log}"
VALIDATOR_URL="${RELIQUARY_VALIDATOR_URL:-http://86.38.238.30:8080}"
INTERVAL="${RELIQUARY_WD_INTERVAL:-30}"
STALE="${RELIQUARY_WD_STALE:-600}"

cd "$REPO_DIR"

alert() {
  local ts; ts="$(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  printf '[%s] %s\n' "$ts" "$*" | tee -a "$NOTIFY_LOG"
}

# Hotkey ss58 for /verdicts — read from the miner log's poll URL, fall back to wallet.
hotkey_of() {
  grep -oE '/verdicts/[1-9A-HJ-NP-Za-km-z]+' "$MINER_LOG" 2>/dev/null | head -1 | sed 's#/verdicts/##'
}

restart_miner() {
  alert "MINER DOWN/STALE — restarting via run.sh"
  ( source mining/openmath/.env 2>/dev/null; bash mining/openmath/run.sh ) >>"$NOTIFY_LOG" 2>&1 || true
  sleep 90  # allow boot + model load before the next liveness check
}

alert "watchdog started (interval=${INTERVAL}s stale=${STALE}s) for miner pid=$(cat "$PID_FILE" 2>/dev/null)"
last_fire=0
last_ts="$(date +%s)"   # only alert on verdicts from now on (skip historical)

while true; do
  # --- 1. liveness: process alive AND log fresh ---
  pid="$(cat "$PID_FILE" 2>/dev/null || echo)"
  alive=1
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then alive=0; fi
  if [ -f "$MINER_LOG" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$MINER_LOG" 2>/dev/null || echo 0) ))
    if [ "$age" -gt "$STALE" ]; then alive=0; alert "log idle ${age}s (> ${STALE}s)"; fi
  fi
  if [ "$alive" -eq 0 ]; then restart_miner; last_fire=0; continue; fi

  # --- 2. new fires in the miner log ---
  fc=$(grep -c "FIRED" "$MINER_LOG" 2>/dev/null); fc=${fc:-0}
  if [ "$fc" -gt "$last_fire" ]; then
    grep "FIRED" "$MINER_LOG" | tail -n "$((fc - last_fire))" | while IFS= read -r line; do
      alert "🔥 FIRE: ${line#*FIRED }"
    done
    last_fire=$fc
  fi

  # --- 3. new validator verdicts (the REAL outcome) ---
  hk="$(hotkey_of)"
  if [ -n "$hk" ]; then
    out="$(RELIQUARY_VAL="$VALIDATOR_URL" RELIQUARY_HK="$hk" RELIQUARY_SINCE="$last_ts" \
      /root/reliquary/.venv/bin/python - <<'PY' 2>/dev/null
import json, os, urllib.request
url=os.environ["RELIQUARY_VAL"]; hk=os.environ["RELIQUARY_HK"]; since=float(os.environ["RELIQUARY_SINCE"])
try:
    with urllib.request.urlopen(f"{url}/verdicts/{hk}?since={since}", timeout=8) as r:
        vs=json.load(r).get("verdicts", [])
except Exception:
    print(f"TS|{since}"); raise SystemExit
mx=since
for v in vs:
    mx=max(mx, v.get("ts", 0))
    tag="ACCEPTED ✅" if v.get("accepted") else f"REJECTED {v.get('reason')}"
    print(f"VERDICT|win={v.get('window_n')}|{tag}")
print(f"TS|{mx}")
PY
)"
    while IFS= read -r row; do
      case "$row" in
        VERDICT\|*) alert "📨 ${row#VERDICT|}" ;;
        TS\|*)      last_ts="${row#TS|}" ;;
      esac
    done <<< "$out"
  fi

  sleep "$INTERVAL"
done
