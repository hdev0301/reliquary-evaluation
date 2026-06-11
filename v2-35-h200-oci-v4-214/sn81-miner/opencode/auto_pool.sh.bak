#!/bin/bash
# auto_pool.sh — keep the OpenCode miner running on a FRESH curated in-zone pool.
#
# WHY: in-zone prompts are a permanently-depleting resource (won prompts cool for
# 1,000,000 windows => forever) AND the checkpoint republishes ~every 10 windows
# (scatter shifts). A static curated pool therefore goes stale two ways: drift +
# depletion. This supervisor rebuilds the pool when needed and (re)launches the miner
# on it, so the miner always mines network-fresh, checkpoint-current in-zone prompts.
#
# SINGLE H200: build and mine CANNOT overlap (two vLLM instances OOM). So each refresh
# PAUSES mining. The nvidia reconstruction is cached one-time (data/oci_cases_cache.json),
# so a refresh is generation+grading only (~30-40 min for 2000 candidates).
#
# REFRESH TRIGGER (whichever first) — tuned to avoid wasteful every-checkpoint rebuilds:
#   * the validator checkpoint advanced >= REFRESH_EVERY_CKPTS times since the last build, OR
#   * the FRESH (non-cooled) pool count drops below MIN_FRESH (~MIN_FRESH/8 windows left).
# A hard MIN_REFRESH_INTERVAL_S floor prevents thrashing.
#
# USAGE:  bash /root/sn81-miner/opencode/auto_pool.sh        # run under tmux/nohup
#   tune via env: REFRESH_EVERY_CKPTS=3 MIN_FRESH=64 POOL_MAX_CANDIDATES=2000 POOL_M=10
set -u

SN81="${SN81:-/root/sn81-miner}"
REPO="${REPO:-/root/reliquary}"
VURL="${VALIDATOR_URL:-http://86.38.238.30:8080}"
POOL="$SN81/data/inzone_pool_opencode.json"
STATE="$SN81/data/auto_pool_state.json"          # sidecar: {built_ckpt_n}
LOG="$SN81/logs/auto_pool.log"
MODEL_CACHE="/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots"

REFRESH_EVERY_CKPTS="${REFRESH_EVERY_CKPTS:-3}"   # rebuild after the checkpoint advances this many times
MIN_FRESH="${MIN_FRESH:-64}"                       # or when fewer than this many non-cooled pool prompts remain
MIN_REFRESH_INTERVAL_S="${MIN_REFRESH_INTERVAL_S:-3600}"   # never refresh more often than this
POLL_S="${POLL_S:-120}"
POOL_MAX_CANDIDATES="${POOL_MAX_CANDIDATES:-2000}"
POOL_M="${POOL_M:-10}"

log(){ echo "$(date -u +%H:%M:%S) | $*" | tee -a "$LOG"; }

# --- helpers (python one-liners against /state and the pool) -------------------
cur_ckpt_n(){ "$REPO/.venv/bin/python" - "$VURL" <<'PY'
import sys,httpx
try: print(httpx.get(sys.argv[1]+"/state",timeout=10).json().get("checkpoint_n",-1))
except Exception: print(-1)
PY
}
fresh_pool_count(){ "$REPO/.venv/bin/python" - "$VURL" "$POOL" <<'PY'
import sys,json,httpx
try:
    pool=set(json.load(open(sys.argv[2])))
    cd=set(httpx.get(sys.argv[1]+"/state",timeout=10).json().get("cooldown_prompts",[]))
    print(len(pool-cd))
except Exception: print(-1)
PY
}
latest_snapshot(){ ls -dt "$MODEL_CACHE"/*/ 2>/dev/null | head -1; }   # the one the miner last loaded (present locally)
built_ckpt(){ [ -f "$STATE" ] && "$REPO/.venv/bin/python" -c "import json;print(json.load(open('$STATE')).get('built_ckpt_n',-1))" 2>/dev/null || echo -1; }

stop_miner(){
  local mpid; mpid=$(pgrep -f "reliquary.cli.main mine" | head -1)
  if [ -n "$mpid" ]; then pkill -9 -P "$mpid" 2>/dev/null; kill -9 "$mpid" 2>/dev/null; fi
  pkill -9 -f "reliquary.cli.main mine" 2>/dev/null; pkill -9 -f "EngineCore" 2>/dev/null
  sleep 6
}

build_pool(){
  local snap; snap="$(latest_snapshot)"
  [ -z "$snap" ] && { log "BUILD SKIP: no local snapshot found"; return 1; }
  log "BUILD start on $(basename "$snap") (candidates=$POOL_MAX_CANDIDATES m=$POOL_M)"
  ( cd "$REPO" && set -a && source scripts/.env 2>/dev/null; set +a
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "$REPO/.venv/bin/python" "$SN81/opencode/build_opencode_pool.py" \
      --checkpoint "$snap" --max-candidates "$POOL_MAX_CANDIDATES" --m "$POOL_M" \
      --strict-zone --gpu-mem-util 0.65 --out "$POOL" ) >> "$SN81/logs/build_pool.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$POOL" ]; then log "BUILD FAILED (rc=$rc) — keeping previous pool"; return 1; fi
  local n; n=$("$REPO/.venv/bin/python" -c "import json;print(len(json.load(open('$POOL'))))" 2>/dev/null)
  "$REPO/.venv/bin/python" -c "import json;json.dump({'built_ckpt_n':$(cur_ckpt_n)},open('$STATE','w'))"
  log "BUILD done: pool=$n idxs"
}

start_miner(){
  log "MINER (re)launch on curated pool"
  POOL="$POOL" MODE=opencode bash "$SN81/bin/run_miner.sh" >> "$LOG" 2>&1
}

# --- main loop ----------------------------------------------------------------
log "=== auto_pool supervisor start (refresh_every=$REFRESH_EVERY_CKPTS ckpts, min_fresh=$MIN_FRESH) ==="
[ -s "$POOL" ] || { log "no pool yet -> initial build"; stop_miner; build_pool; }
start_miner
last_refresh=0
while true; do
  sleep "$POLL_S"
  pgrep -f "reliquary.cli.main mine" >/dev/null || { log "miner died -> relaunch"; start_miner; continue; }
  cn=$(cur_ckpt_n); bn=$(built_ckpt); fresh=$(fresh_pool_count)
  now=$(date +%s); since=$((now - last_refresh))
  drift=$(( cn>=0 && bn>=0 ? cn-bn : 0 ))
  need=0
  [ "$cn" -ge 0 ] && [ "$bn" -ge 0 ] && [ "$drift" -ge "$REFRESH_EVERY_CKPTS" ] && need=1
  [ "$fresh" -ge 0 ] && [ "$fresh" -lt "$MIN_FRESH" ] && need=1
  if [ "$need" = 1 ] && [ "$since" -ge "$MIN_REFRESH_INTERVAL_S" ]; then
    log "REFRESH trigger: ckpt_drift=$drift fresh=$fresh (built@$bn now@$cn) -> pause+rebuild"
    stop_miner; build_pool; start_miner; last_refresh=$(date +%s)
  else
    log "ok: ckpt=$cn built@$bn drift=$drift fresh=$fresh (no refresh)"
  fi
done
