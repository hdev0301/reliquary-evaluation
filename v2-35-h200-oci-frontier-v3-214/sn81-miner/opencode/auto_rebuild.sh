#!/bin/bash
# Auto-rebuild the difficulty-screened opencode pool when the validator checkpoint
# advances AND the pool has actually gone stale — ending the manual treadmill.
# -----------------------------------------------------------------------------
# A single-GPU rebuild requires miner downtime, so this does NOT rebuild on every
# checkpoint bump. It fires only when ALL hold:
#   (1) checkpoint_n advanced since last seen,
#   (2) the pool is STALE — recent screen `allcorr` rate >= ALLCORR_STALE (the model
#       now solves most pool prompts 8/8 = the pool no longer matches the checkpoint),
#   (3) >= MIN_INTERVAL since the last rebuild (anti-thrash),
#   (4) no build already running and the miner is currently alive.
# When it fires it runs the SAME safe sequence we do by hand: download the new
# snapshot -> stand down watchdog -> stop miner -> build_pool.sh (atomic swap, #1
# cooldown filter, MIN_KEEP guard) -> delete hot_pool -> relaunch miner -> re-enable
# watchdog. Everything is logged. Stop with: touch /root/auto_rebuild.stop
set -u
SN81=/root/sn81-miner; OC=$SN81/opencode; REPO=/root/reliquary; PY=$REPO/.venv/bin/python
VURL="${VALIDATOR_URL:-http://86.38.238.30:8080}"
LOG=/root/auto_rebuild.log
STATE_F=/root/.auto_rebuild_ckpt
LOCK=/root/.auto_rebuild.lock
STOP=/root/auto_rebuild.stop
HOTKEY="${HOTKEY:-ronnywebdev_hotkey}"
POLL="${POLL:-600}"                  # check every 10 min (CPU/network only)
MIN_INTERVAL="${MIN_INTERVAL:-5400}"  # >= 90 min between rebuilds — matched to the ~70-90min checkpoint cadence so a FAST build keeps pace (was 3h, too slow)
ALLCORR_STALE="${ALLCORR_STALE:-19}"  # of 24: pool stale once the model solves >= this many 8/8
NCAND="${NCAND:-4000}"                # 4000 (was 2500): strong ckpts have a low in-band keep-rate; more candidates -> the dense overlay clears MIN_KEEP=150 and is worth unioning. The broad base is the floor regardless.
M="${M:-10}"                          # fewer screen rollouts -> faster build (still enough to classify intermediate)
last_rebuild=0
log(){ echo "$(date -u +'%F %T') $*" >> "$LOG"; }

# init last-seen checkpoint ONLY if we have no prior state. The old unconditional seed meant every
# process RESTART (5x today) re-pinned the baseline to the current ckpt and FORGOT any pending
# advance -> rebuilds were suppressed and the pool went 3 generations stale. Seed-if-absent fixes it.
cur0=$(curl -s --max-time 10 "$VURL/state" | $PY -c "import sys,json;print(json.load(sys.stdin).get('checkpoint_n',''))" 2>/dev/null)
[ -n "$cur0" ] && [ ! -f "$STATE_F" ] && echo "$cur0" > "$STATE_F"
log "auto_rebuild START (poll=${POLL}s min_interval=${MIN_INTERVAL}s allcorr_stale=${ALLCORR_STALE} ckpt0=$cur0)"

while true; do
  [ -f "$STOP" ] && { log "STOP flag -> exit"; rm -f "$LOCK"; exit 0; }
  sleep "$POLL"
  STATE=$(curl -s --max-time 12 "$VURL/state" 2>/dev/null)
  CN=$(printf '%s' "$STATE" | $PY -c "import sys,json;print(json.load(sys.stdin).get('checkpoint_n',''))" 2>/dev/null)
  REV=$(printf '%s' "$STATE" | $PY -c "import sys,json;print(json.load(sys.stdin).get('checkpoint_revision',''))" 2>/dev/null)
  RID=$(printf '%s' "$STATE" | $PY -c "import sys,json;print(json.load(sys.stdin).get('checkpoint_repo_id',''))" 2>/dev/null)
  [ -z "$CN" ] && continue
  LAST=$(cat "$STATE_F" 2>/dev/null || echo "")
  [ "$CN" = "$LAST" ] && continue
  log "checkpoint advance: $LAST -> $CN ($REV)"
  echo "$CN" > "$STATE_F"

  # (2) staleness gate — is the pool actually degraded at the new checkpoint?
  ALLCORR=$(grep -E "screen: [0-9]+/24 promising" "$SN81/logs/miner.log" 2>/dev/null | tail -5 \
            | grep -oE "allcorr=[0-9]+" | grep -oE "[0-9]+" | sort -rn | head -1)
  [ -z "$ALLCORR" ] && ALLCORR=0
  if [ "$ALLCORR" -lt "$ALLCORR_STALE" ]; then
    log "pool still healthy (max recent allcorr=$ALLCORR < $ALLCORR_STALE) -> defer rebuild"
    continue
  fi
  # (3) anti-thrash
  now=$(date +%s)
  if [ $((now - last_rebuild)) -lt "$MIN_INTERVAL" ]; then log "within min_interval -> defer"; continue; fi
  # (4) no concurrent build, miner alive
  if pgrep -f "build_opencode_pool.py" >/dev/null; then log "a build is already running -> skip"; continue; fi
  [ -f "$LOCK" ] && { log "lock present -> skip"; continue; }
  MPID=$(cat "$SN81/miner.pid" 2>/dev/null)
  if [ -z "$MPID" ] || ! kill -0 "$MPID" 2>/dev/null; then log "miner not alive (someone else managing) -> skip"; continue; fi

  touch "$LOCK"
  log "REBUILD start (ckpt $CN, allcorr=$ALLCORR)"
  # download the new snapshot so build_pool.sh screens against the RIGHT model
  set -a; source "$REPO/scripts/.env" 2>/dev/null; set +a
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$RID', revision='$REV')" >> "$LOG" 2>&1 \
     && log "snapshot $RID@${REV:0:12} ready" || log "WARN snapshot download failed -> build will fall back to latest-local"
  # stand down watchdog so it doesn't restart the miner mid-build
  touch /root/miner_watchdog.stop; for p in $(pgrep -f '[m]iner_watchdog.sh'); do kill -9 "$p" 2>/dev/null; done
  # stop miner (surgical group-kill; grader survives)
  PGID=$(ps -o pgid= -p "$MPID" 2>/dev/null | tr -d ' ')
  [ -n "$PGID" ] && kill -9 -"$PGID" 2>/dev/null
  sleep 5
  log "miner stopped (was $MPID); GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
  # build (blocks; atomic swap + #1 cooldown filter + MIN_KEEP guard inside)
  NCAND="$NCAND" M="$M" timeout 9000 bash "$OC/build_pool.sh" >> "$LOG" 2>&1
  bexit=$?
  log "build_pool.sh exit=$bexit"
  # HYBRID apply: capture the dense screen as the priority OVERLAY (only if it actually SWAPPED in),
  # then build the standing pool = BROAD gradeable base UNION dense overlay. build_frontier_pool.sh
  # writes the pool (broad floor -> never starves when the dense set goes stale) and RESTART=1
  # relaunches the miner. Replaces the old "relaunch on the bare 551 dense pool" that re-starved us.
  rm -f "$OC/data/hot_pool.json"
  if [ "$bexit" = "0" ]; then cp "$OC/data/inzone_pool_opencode.json" "$OC/data/inzone_pool_screened.json" 2>/dev/null && log "captured dense overlay ($(wc -l < "$OC/data/inzone_pool_screened.json" 2>/dev/null || echo '?') bytes-ish)"; fi
  OVERLAY="$OC/data/inzone_pool_screened.json" KEEP_FRAC="${KEEP_FRAC:-0.7}" HOTKEY="$HOTKEY" RESTART=1 \
    bash "$OC/build_frontier_pool.sh" >> "$LOG" 2>&1
  sleep 10
  NP=$(cat "$SN81/miner.pid" 2>/dev/null)
  log "miner relaunched -> pid $NP $(kill -0 "$NP" 2>/dev/null && echo ALIVE || echo '??')"
  # re-enable watchdog
  rm -f /root/miner_watchdog.stop; setsid bash /root/miner_watchdog.sh </dev/null >/dev/null 2>&1 &
  last_rebuild=$(date +%s)
  rm -f "$LOCK"
  log "REBUILD done"
done
