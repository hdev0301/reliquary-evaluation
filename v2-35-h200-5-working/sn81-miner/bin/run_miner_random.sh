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
# CYCLE / forced round-robin mode:  `bash run_miner.sh --cycle [N]`  (default N=3)
# rotates to the NEXT inzone pool EVERY N windows UNCONDITIONALLY — whether or not
# accepts are landing and whether or not the pool fails. Use it to sweep every pool
# evenly. The crash-rotate + silent-pool safety nets still apply. In cycle mode an
# ACCEPT does NOT reset the counter. Env equivalent: CYCLE_EVERY_WINDOWS=N.
#
# RANDOM selection:  add `--random` (env RANDOM_POOL=1) to pick the next pool at
# RANDOM (a different pool than the current one) instead of the sequential next.
# This controls WHICH pool is chosen on each rotation; --cycle controls WHEN to
# rotate. They compose: `--cycle 3 --random` => every 3 windows, jump to a random
# pool. In random mode the START pool is also chosen randomly (ignores START_POOL).
#
#   knobs (env):  ROTATE_AFTER_WINDOWS=10   # windows w/o an ACCEPT before rotating
#                 WINDOW_SECS=120           # ~seconds/window, used only as a silent-pool
#                                           # safety net (rotate a pool that submits NOTHING)
#                 POLL_SECS=20              # how often the watchdog checks the log
#                 START_POOL=<path>         # which inzone pool to launch first
#                 CRASH_MAX=3               # consecutive miner crashes before rotating OFF a pool
#                 CRASH_WINDOW=180          # seconds: deaths within this window = a "crash loop"
#                 CYCLE_EVERY_WINDOWS=0     # >0 => rotate every N windows regardless (same as --cycle N)
#                 RANDOM_POOL=0             # 1 => pick next pool randomly, not sequential (same as --random)
#   logs:  miner    -> logs/miner.log       (truncated on each (re)launch)
#          watchdog -> logs/supervisor.log  (rotation history; appended)
#   stop:  pkill -f 'run_miner.sh --supervise'   # stop the watchdog
#          pkill -f 'reliquary.cli.main mine'    # then stop the python miner
#   disable rotation: ROTATE_AFTER_WINDOWS=999999 bash run_miner.sh

set -u
SN81=/root/sn81-miner
REPO=/root/reliquary
SELF="$SN81/bin/run_miner_random.sh"
LOG="$SN81/logs/miner.log"
SUP_LOG="$SN81/logs/supervisor.log"
MINER_PID="$SN81/miner.pid"
SUP_PID="$SN81/supervisor.pid"
ROTATE_AFTER_WINDOWS="${ROTATE_AFTER_WINDOWS:-5}"
WINDOW_SECS="${WINDOW_SECS:-120}"
POLL_SECS="${POLL_SECS:-20}"
START_POOL="${START_POOL:-$SN81/data/inzone_pool_custom.json}"
CRASH_MAX="${CRASH_MAX:-3}"          # consecutive miner crashes before rotating off a pool
CRASH_WINDOW="${CRASH_WINDOW:-180}"  # seconds: crashes within this window count as a "crash loop"
CYCLE_EVERY_WINDOWS="${CYCLE_EVERY_WINDOWS:-5}"  # >0 => FORCED round-robin: rotate every N windows regardless of accepts/fails (--cycle)
RANDOM_POOL="${RANDOM_POOL:-0}"      # 1 => pick the next pool RANDOMLY (!= current) instead of sequential (--random)

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
  # Pool banner as the FIRST lines of the (truncated) miner log, so anyone tailing
  # miner.log immediately sees WHICH inzone json this run is mining — refreshed on
  # every (re)launch and every rotation to a new pool.
  { echo "################################################################"
    echo "### MINING POOL: $(basename "$pool")"
    echo "###        file: $pool"
    echo "###     launched: $(date '+%F %T')"
    echo "################################################################"
  } > "$LOG"
  nohup .venv/bin/python -m reliquary.cli.main mine \
    --network finney --netuid 81 --wallet-name ronnywebdev --hotkey ronnywebdev_hotkey \
    --checkpoint Qwen/Qwen3.5-4B --validator-url http://86.38.238.30:8080 \
    --gpu-memory-utilization 0.65 --pool-size 48 --gen-batch 8 \
    --max-new-tokens 420 --oversample 64 \
    --prompt-idx-file "$pool" --two-stage \
    --log-level INFO >> "$LOG" 2>&1 &
  echo $! > "$MINER_PID"
  sup_log "launched miner PID=$(cat "$MINER_PID") pool=$(basename "$pool") | file=$pool | log=$LOG"
}

# max window number seen in the miner log (from submitted/FIRE/verdict 'win='/'window=' lines)
latest_win()   { grep -aoE "win(dow)?=[0-9]+" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1; }
# count of validator-ACCEPTED verdicts in the current run
accept_count() { grep -acE "verdict ACCEPTED" "$LOG" 2>/dev/null; }
# pick the next pool index: RANDOM (!= current) when RANDOM_POOL=1, else sequential next (wrapping)
next_idx() {
  local cur="$1" n=${#POOLS[@]} r
  (( n <= 1 )) && { echo 0; return; }
  if (( RANDOM_POOL )); then
    r=$cur; while (( r == cur )); do r=$(( RANDOM % n )); done; echo "$r"
  else
    echo $(( (cur + 1) % n ))
  fi
}

supervise() {
  if (( ${#POOLS[@]} == 0 )); then sup_log "FATAL: no inzone_pool_*.json under $SN81/data"; exit 1; fi
  local idx=0 i
  for i in "${!POOLS[@]}"; do [[ "${POOLS[$i]}" == "$START_POOL" ]] && idx=$i; done
  local cycle=0 rotate_n=$ROTATE_AFTER_WINDOWS sel="sequential"
  (( CYCLE_EVERY_WINDOWS > 0 )) && { cycle=1; rotate_n=$CYCLE_EVERY_WINDOWS; }
  if (( RANDOM_POOL )); then sel="RANDOM"; (( ${#POOLS[@]} > 1 )) && idx=$(( RANDOM % ${#POOLS[@]} )); fi  # random start too
  if (( cycle )); then
    sup_log "watchdog start: ${#POOLS[@]} pools | start=$(basename "${POOLS[$idx]}") | CYCLE mode: rotate EVERY ${rotate_n} windows (ignore accepts/fails), select=${sel}, or ${CRASH_MAX} crashes/${CRASH_WINDOW}s"
  else
    sup_log "watchdog start: ${#POOLS[@]} pools | start=$(basename "${POOLS[$idx]}") | rotate after ${rotate_n} windows w/o ACCEPT, select=${sel}, or ${CRASH_MAX} crashes/${CRASH_WINDOW}s"
  fi
  local crashes=0 crash_t0=0   # consecutive-crash tracker for the CURRENT pool (rotate off a crash-looping pool)
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
      if (( acc > prev_acc )); then   # a new ACCEPT landed
        cur=$(latest_win); prev_acc=$acc
        if (( cycle )); then          # CYCLE mode: accepts do NOT reset the rotation counter
          sup_log "ACCEPT on $(basename "$pool") (accepts=$acc win=${cur:-?}) — cycle mode, counter NOT reset"
        else                          # normal mode: an accept resets the no-accept window counter
          [[ -n "$cur" ]] && base_win=$cur; base_time=$(date +%s)
          sup_log "ACCEPT on $(basename "$pool") (accepts=$acc win=${cur:-?}) — counter reset"
        fi
        continue
      fi
      cur=$(latest_win)
      if [[ -n "$cur" ]]; then        # primary trigger: rotate_n windows elapsed
        [[ -z "$base_win" ]] && base_win=$cur
        if (( cur - base_win >= rotate_n )); then
          if (( cycle )); then reason="cycled $((cur - base_win)) windows"; else reason="no ACCEPT for $((cur - base_win)) windows"; fi
          break
        fi
      else                            # silent-pool safety net: nothing submitted at all
        now=$(date +%s); elapsed=$(( now - base_time ))
        if (( elapsed >= rotate_n * WINDOW_SECS )); then
          reason="no submissions for ${elapsed}s (pool produced nothing)"; break
        fi
      fi
    done
    if [[ "$reason" == "DIED" ]]; then
      now=$(date +%s)
      # start a fresh crash streak if this is the first death (crashes==0) or the last one was long ago
      if (( crashes == 0 || now - crash_t0 > CRASH_WINDOW )); then crash_t0=$now; crashes=1; else crashes=$(( crashes + 1 )); fi
      if (( crashes >= CRASH_MAX )); then   # crash-looping pool -> rotate OFF it instead of relaunching forever
        sup_log "miner crash-looped ${crashes}x within $(( now - crash_t0 ))s on $(basename "$pool") — ROTATING away (bad pool)"
        idx=$(next_idx "$idx"); crashes=0; crash_t0=0
        sup_log "${sel} pick -> $(basename "${POOLS[$idx]}")"
      else
        sup_log "miner exited on $(basename "$pool") (crash ${crashes}/${CRASH_MAX} in ${CRASH_WINDOW}s) — relaunching SAME pool"
      fi
    else
      idx=$(next_idx "$idx"); crashes=0; crash_t0=0   # healthy rotation -> reset crash streak
      sup_log "ROTATE ($reason, ${sel} pick) on $(basename "$pool") -> $(basename "${POOLS[$idx]}")"
    fi
  done
}

# ----------------------------- entrypoint ----------------------------- #
if [[ "${1:-}" == "--supervise" ]]; then
  supervise
  exit 0
fi

# Top-level flag parsing. --cycle [N] enables FORCED round-robin (rotate every N
# windows, default 3, regardless of accepts/fails); passed to the detached child via env.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cycle) if [[ "${2:-}" =~ ^[0-9]+$ ]]; then CYCLE_EVERY_WINDOWS="$2"; shift 2; else CYCLE_EVERY_WINDOWS=3; shift; fi ;;
    --random) RANDOM_POOL=1; shift ;;
    -h|--help) sed -n '11,40p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "WARN: unknown arg '$1' (ignored)" >&2; shift ;;
  esac
done
# Export every knob so the detached --supervise child inherits flag- and env-set values.
export ROTATE_AFTER_WINDOWS WINDOW_SECS POLL_SECS START_POOL CRASH_MAX CRASH_WINDOW CYCLE_EVERY_WINDOWS RANDOM_POOL

# Stop any previous watchdog (and its miner), then spawn a fresh detached one.
pkill -9 -f "run_miner.sh --supervise" 2>/dev/null || true
[[ -f "$SUP_PID" ]] && kill -9 "$(cat "$SUP_PID")" 2>/dev/null || true
sleep 1
echo "==== watchdog (re)start $(date '+%F %T') ====" >> "$SUP_LOG"
nohup bash "$SELF" --supervise >> "$SUP_LOG" 2>&1 &
echo $! > "$SUP_PID"
SEL_DESC=$([ "${RANDOM_POOL}" = "1" ] && echo "RANDOM pool each rotation" || echo "sequential round-robin")
if (( CYCLE_EVERY_WINDOWS > 0 )); then
  echo "watchdog PID=$(cat "$SUP_PID") | CYCLE mode: rotates EVERY ${CYCLE_EVERY_WINDOWS} windows (regardless of accepts/fails) | ${SEL_DESC}"
else
  echo "watchdog PID=$(cat "$SUP_PID") | rotates after ${ROTATE_AFTER_WINDOWS} windows w/o 'verdict ACCEPTED' | ${SEL_DESC}"
fi
echo "  ${#POOLS[@]} pools: $(for p in "${POOLS[@]}"; do basename "$p"; done | tr '\n' ' ')"
echo "  start=$(basename "$START_POOL") | miner-log=$LOG | watchdog-log=$SUP_LOG"
