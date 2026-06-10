#!/bin/bash
# Local opencode grader server — plain (non-gVisor) workers running the REAL validator
# worker.py, so compute_reward() grades opencode code with exact validator parity.
# Unsandboxed exec of model code on this box (acceptable for own model output).
set -u
REPO="${REPO:-/root/reliquary}"
SOCK="${GRADER_SOCKET:-/tmp/reliquary-grader.sock}"
LOG="/root/sn81-miner/opencode/logs/grader.log"
PIDF="/root/sn81-miner/opencode/logs/grader.pid"
cmd="${1:-start}"

running() { [ -S "$SOCK" ] && pgrep -f "reliquary.environment.grader.server" >/dev/null 2>&1; }

case "$cmd" in
  start)
    if running; then echo "grader already running (socket $SOCK)"; exit 0; fi
    rm -f "$SOCK"
    mkdir -p "$(dirname "$LOG")"
    cd "$REPO" || exit 1
    export PYTHONPATH="$REPO"
    setsid "$REPO/.venv/bin/python" -m reliquary.environment.grader.server \
      --socket "$SOCK" --pool-size "${GRADER_POOL_SIZE:-16}" --timeout "${GRADER_TIMEOUT:-5}" --metrics-port 0 \
      < /dev/null > "$LOG" 2>&1 &
    echo $! > "$PIDF"
    for _ in 1 2 3 4 5 6; do running && break; sleep 1; done
    if running; then echo "grader started (socket $SOCK, pool ${GRADER_POOL_SIZE:-16})"; else echo "FAILED to start grader; see $LOG"; exit 1; fi
    ;;
  stop)
    pkill -9 -f "reliquary.environment.grader.server" 2>/dev/null || true
    rm -f "$SOCK"; echo "grader stopped" ;;
  status)
    running && echo "running (socket $SOCK)" || echo "not running" ;;
  *) echo "usage: grader.sh {start|stop|status}"; exit 1 ;;
esac
