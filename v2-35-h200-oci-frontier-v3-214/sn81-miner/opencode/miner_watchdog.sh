#!/bin/bash
# Auto-restart the SN81 opencode miner if it hangs (vLLM EngineCore deadlock:
# log stops advancing while the process is alive, GPU idles at 0%). Keeps the
# miner producing even if the deadlock recurs. Restarts the SAME config (no
# tuning change). Stop with: touch /root/miner_watchdog.stop  (or kill this pid)
SN81=/root/sn81-miner; OC=$SN81/opencode; LOG=$SN81/logs/miner.log
WLOG=/root/miner_watchdog.log
STALE=120            # no log progress for this long (s) while alive = hung (faster recovery = smaller stale_round burst)
MINGAP=150           # min seconds between restarts (anti-storm)
last_restart=0
echo "$(date -u +%H:%M:%S) watchdog START (stale>${STALE}s -> restart)" >> "$WLOG"
while true; do
  [ -f /root/miner_watchdog.stop ] && { echo "$(date -u +%H:%M:%S) watchdog STOP (flag)" >> "$WLOG"; exit 0; }
  sleep 30
  PID=$(cat "$SN81/miner.pid" 2>/dev/null); [ -z "$PID" ] && continue
  reason=""
  if ! kill -0 "$PID" 2>/dev/null; then
    reason="DEAD"
  else
    age=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo "$(date +%s)") ))
    [ "$age" -gt "$STALE" ] && reason="HUNG_${age}s_gpu$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null|tr -d ' ')"
  fi
  [ -z "$reason" ] && continue
  now=$(date +%s); [ $((now - last_restart)) -lt "$MINGAP" ] && continue
  echo "$(date -u +%H:%M:%S) watchdog: $reason -> restarting miner (was pid $PID)" >> "$WLOG"
  kill -9 "$PID" 2>/dev/null
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  HOTKEY=ronnywebdev_hotkey setsid bash "$OC/run_miner.sh" >> "$WLOG" 2>&1
  last_restart=$(date +%s)
  echo "$(date -u +%H:%M:%S) watchdog: relaunched -> new pid $(cat "$SN81/miner.pid" 2>/dev/null)" >> "$WLOG"
  sleep 120          # let it boot before monitoring again
done
