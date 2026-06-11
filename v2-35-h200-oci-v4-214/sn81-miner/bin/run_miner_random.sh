#!/bin/bash
# Reliquary miner — CURATION / validator-replica pipeline (reward-vector candidate
# selection). On a converged BIMODAL checkpoint, natural 8-sample groups score 8/8
# or 0/8 (never in-zone). Curation (pregen.py build_groups) over-generates, rewards
# every candidate against the PUBLIC env reward, SELECTS an in-zone 8-subset
# (k correct + 8-k wrong), places it non-monotonically (passes reward_shape), and
# pre-validates every gate locally (zero integrity rejects, like the rank-1 miner).
# All rollouts are genuine current-checkpoint samples; the validator recomputes the
# reward -> selection, not fabrication.
#
# === POOL-ROTATION WATCHDOG ===
# `bash run_miner.sh` launches the curation miner on ONE inzone_pool_*.json from
# data/, then watches the validator verdicts in the miner log. If NO `verdict
# ACCEPTED` appears for ROTATE_AFTER_WINDOWS (default 10) consecutive windows, it
# rotates to the NEXT inzone_pool_*.json (sorted, wrapping) and relaunches. Any
# `verdict ACCEPTED` resets the counter (we keep mining a pool that is landing).
# The watchdog runs DETACHED, so this command returns immediately.
#
#   knobs (env):  ROTATE_AFTER_WINDOWS=10   # windows w/o an ACCEPT before rotating
#                 WINDOW_SECS=120           # ~seconds/window, used only as a silent-pool
#                                           # safety net (rotate a pool that submits NOTHING)
#                 POLL_SECS=20              # how often the watchdog checks the log
#                 START_POOL=<path>         # which inzone pool to launch first
#   logs:  miner    -> logs/miner.log       (truncated on each (re)launch)
#          watchdog -> logs/supervisor.log  (rotation history; appended)
#   stop:  pkill -f 'run_miner.sh --supervise'   # stop the watchdog
#          pkill -f 'reliquary.cli.main mine'    # then stop the python miner
#   disable rotation: ROTATE_AFTER_WINDOWS=999999 bash run_miner.sh

set -u
SN81=/root/sn81-miner
REPO=/root/reliquary
SELF="$SN81/bin/run_miner.sh"
LOG="$SN81/logs/miner.log"
SUP_LOG="$SN81/logs/supervisor.log"
MINER_PID="$SN81/miner.pid"
SUP_PID="$SN81/supervisor.pid"
ROTATE_AFTER_WINDOWS="${ROTATE_AFTER_WINDOWS:-5}"
WINDOW_SECS="${WINDOW_SECS:-120}"
POLL_SECS="${POLL_SECS:-20}"
START_POOL="${START_POOL:-$SN81/data/inzone_pool_custom.json}"

mkdir -p "$SN81/logs"

# Rotation candidates: every inzone_pool_*.json in data/ (sorted). The glob excludes
# hot_pool*.json (runtime cache) and submitted_idx*.json (blocklist) by construction.
mapfile -t POOLS < <(ls -1 "$SN81"/data/inzone_pool_*.json 2>/dev/null | sort)

sup_log() { echo "[$(date '+%F %T')] $*"; }   # stdout; under --supervise stdout is SUP_LOG

kill_miner() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f "reliquary.cli.main mine" 2>/dev/null || true
  sleep 2
}

launch_miner() {
  local pool="$1"
  cd "$REPO" || exit 1
  kill_miner
  set -a; source "$REPO/scripts/.env"; set +a
  # Qwen3.5 GRAIL proof model is the MULTIMODAL HF model (AutoModelForImageTextToText,
  # ~297 vision tensors) -> heavier than the old text-only proof model. Reduce
  # fragmentation so its forward pass fits in the headroom left by vLLM.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # --- curation knobs (read by reliquary/miner/pregen.py) ---
  export RELIQUARY_CURATE=1
  export RELIQUARY_CURATE_TARGET_K=5      # k=5 (5 correct + 3 distinct-wrong, sigma=0.484); CORRECT is
                                          # abundant again after fix/omi-boxed-instruction, WRONG is scarce.
  export RELIQUARY_CURATE_MARGIN=2        # validate k+2 / (8-k)+2 candidates, keep first passing
  export RELIQUARY_SAFE_P_STOP=0.10       # require a comfortable p_stop margin so validator-side linear-attn
                                          # forward drift can't push a borderline rollout under p_stop>=0.01.
  # --- two-stage screen, retuned FOR curation: drop pure 8/8 & 0/8, keep the mix ---
  export RELIQUARY_SCREEN_OVERSAMPLE=24   # thinking DISABLED (#78) -> short cheap completions -> screen wider
  export RELIQUARY_SCREEN_MAX_TOKENS=512  # no-thinking: model answers + \boxed{} in <~400 tokens
  export RELIQUARY_SCREEN_MIN_TERM=4      # ~100% terminate; small floor drops pathological prompts
  export RELIQUARY_SCREEN_P_LOW=0.03      # keep prompts with >=~2 wrong present
  export RELIQUARY_SCREEN_P_HIGH=0.97     # keep prompts with <100% correct (some wrong)
  # Hot pool: self-built cache of screen-proven fluent+curatable prompts, re-mined to amortize discovery.
  export RELIQUARY_HOT_POOL_PATH=/root/sn81-miner/data/hot_pool.json
  export RELIQUARY_HOT_FRAC=0.5
  export RELIQUARY_HOT_CAP=4000
  export RELIQUARY_BURNED_PATH=/root/sn81-miner/data/submitted_idx.json  # persistent anti-hash_duplicate blocklist
  rm -f "$LOG"
  nohup .venv/bin/python -m reliquary.cli.main mine \
    --network finney --netuid 81 --wallet-name ronnywebdev --hotkey ronnywebdev_hotkey \
    --checkpoint Qwen/Qwen3.5-4B --validator-url http://86.38.238.30:8080 \
    --gpu-memory-utilization 0.65 --pool-size 48 --gen-batch 8 \
    --max-new-tokens 420 --oversample 64 \
    --prompt-idx-file "$pool" --two-stage \
    --log-level INFO > "$LOG" 2>&1 &
  echo $! > "$MINER_PID"
  sup_log "launched miner PID=$(cat "$MINER_PID") pool=$(basename "$pool") | log=$LOG"
}

# max window number seen in the miner log (from submitted/FIRE/verdict 'win='/'window=' lines)
latest_win()   { grep -aoE "win(dow)?=[0-9]+" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1; }
# count of validator-ACCEPTED verdicts in the current run
accept_count() { grep -acE "verdict ACCEPTED" "$LOG" 2>/dev/null; }

supervise() {
  if (( ${#POOLS[@]} == 0 )); then sup_log "FATAL: no inzone_pool_*.json under $SN81/data"; exit 1; fi
  local idx=0 i
  for i in "${!POOLS[@]}"; do [[ "${POOLS[$i]}" == "$START_POOL" ]] && idx=$i; done
  sup_log "watchdog start: ${#POOLS[@]} pools | start=$(basename "${POOLS[$idx]}") | rotate after ${ROTATE_AFTER_WINDOWS} windows w/o ACCEPT"
  while true; do
    local pool="${POOLS[$idx]}"
    launch_miner "$pool"
    local base_win="" base_time prev_acc cur acc reason elapsed now
    base_time=$(date +%s)
    prev_acc=$(accept_count)          # 0 — log was just truncated
    reason=""
    while true; do
      sleep "$POLL_SECS"
      if ! kill -0 "$(cat "$MINER_PID" 2>/dev/null)" 2>/dev/null; then reason="DIED"; break; fi
      acc=$(accept_count)
      if (( acc > prev_acc )); then   # a new ACCEPT landed -> reset the no-accept window counter
        cur=$(latest_win); [[ -n "$cur" ]] && base_win=$cur
        prev_acc=$acc; base_time=$(date +%s)
        sup_log "ACCEPT on $(basename "$pool") (accepts=$acc win=${cur:-?}) — counter reset"
        continue
      fi
      cur=$(latest_win)
      if [[ -n "$cur" ]]; then        # primary trigger: N windows elapsed with no ACCEPT
        [[ -z "$base_win" ]] && base_win=$cur
        if (( cur - base_win >= ROTATE_AFTER_WINDOWS )); then
          reason="no ACCEPT for $((cur - base_win)) windows"; break
        fi
      else                            # silent-pool safety net: nothing submitted at all
        now=$(date +%s); elapsed=$(( now - base_time ))
        if (( elapsed >= ROTATE_AFTER_WINDOWS * WINDOW_SECS )); then
          reason="no submissions for ${elapsed}s (pool produced nothing)"; break
        fi
      fi
    done
    if [[ "$reason" == "DIED" ]]; then
      sup_log "miner process exited on $(basename "$pool") — relaunching SAME pool"   # do not advance idx
    else
      idx=$(( (idx + 1) % ${#POOLS[@]} ))
      sup_log "ROTATE ($reason) on $(basename "$pool") -> next pool $(basename "${POOLS[$idx]}")"
    fi
  done
}

# ----------------------------- entrypoint ----------------------------- #
if [[ "${1:-}" == "--supervise" ]]; then
  supervise
  exit 0
fi

# Top-level: stop any previous watchdog (and its miner), then spawn a fresh detached one.
pkill -9 -f "run_miner.sh --supervise" 2>/dev/null || true
[[ -f "$SUP_PID" ]] && kill -9 "$(cat "$SUP_PID")" 2>/dev/null || true
sleep 1
echo "==== watchdog (re)start $(date '+%F %T') ====" >> "$SUP_LOG"
nohup bash "$SELF" --supervise >> "$SUP_LOG" 2>&1 &
echo $! > "$SUP_PID"
echo "watchdog PID=$(cat "$SUP_PID") | rotates inzone pools after ${ROTATE_AFTER_WINDOWS} windows w/o 'verdict ACCEPTED'"
echo "  ${#POOLS[@]} pools: $(for p in "${POOLS[@]}"; do basename "$p"; done | tr '\n' ' ')"
echo "  start=$(basename "$START_POOL") | miner-log=$LOG | watchdog-log=$SUP_LOG"
