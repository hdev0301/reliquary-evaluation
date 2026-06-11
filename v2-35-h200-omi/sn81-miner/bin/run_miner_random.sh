#!/bin/bash
# Reliquary miner — CURATION pipeline + POOL-ROTATION/PRUNE WATCHDOG.
#
# Curation (pregen.py build_groups): on a converged BIMODAL checkpoint, natural
# 8-sample groups score 8/8 or 0/8 (never in-zone). Curation over-generates,
# rewards every candidate against the PUBLIC env reward, SELECTS an in-zone
# 8-subset (k correct + 8-k wrong), and pre-validates every gate locally. All
# rollouts are genuine current-checkpoint samples; the validator recomputes the
# reward -> selection, not fabrication.
#
# === POOL-ROTATION + AUTO-PRUNE WATCHDOG ===
# Launches the curation miner on ONE inzone_pool_*.json from data/, watches the
# validator verdicts in the miner log, and:
#   * ROTATES to the next inzone_pool_*.json (sorted, wrapping) if NO `verdict
#     ACCEPTED` appears for ROTATE_AFTER_WINDOWS consecutive windows. Any ACCEPT
#     resets the counter (keep mining a pool that is landing).
#   * PRUNES (rm) a CANDIDATE pool — basename starting with CAND_PREFIX — that
#     earned ZERO accepts over its whole run, so the candidate search keeps only
#     winners. Hand-built BASE pools (everything NOT matching CAND_PREFIX) are
#     NEVER pruned; they remain as permanent rotation members / fallback.
# Re-globs data/ each cycle, so candidates built in the background by
# gen_pool_candidates.sh are discovered live and pruned ones drop out.
# The watchdog runs DETACHED, so this command returns immediately.
#
#   knobs (env):  ROTATE_AFTER_WINDOWS=5    # windows w/o an ACCEPT before rotating
#                 WINDOW_SECS=120           # ~secs/window; silent-pool safety net
#                 POLL_SECS=20              # how often the watchdog checks the log
#                 START_POOL=<path>         # which inzone pool to launch first
#                 PRUNE_CANDIDATES=1        # 0 to disable deletion (rotate-only)
#                 CAND_PREFIX=inzone_pool_cand_   # only these are ever deleted
#                 CLEAR_HOT_ON_POOL_CHANGE=1      # fresh hot pool per pool (clean accept attribution)
#   logs:  miner    -> logs/miner.log       (truncated on each (re)launch)
#          watchdog -> logs/supervisor.log  (rotation/prune history; appended)
#   stop:  pkill -f 'run_miner_random.sh --supervise'   # stop the watchdog
#          pkill -f 'reliquary.cli.main mine'           # then stop the python miner
#   disable rotation: ROTATE_AFTER_WINDOWS=999999 bash run_miner_random.sh

set -u
SN81=/root/sn81-miner
REPO=/root/reliquary
SELF="$SN81/bin/run_miner_random.sh"
LOG="$SN81/logs/miner.log"
SUP_LOG="$SN81/logs/supervisor.log"
MINER_PID="$SN81/miner.pid"
SUP_PID="$SN81/supervisor.pid"
ROTATE_AFTER_WINDOWS="${ROTATE_AFTER_WINDOWS:-5}"
ACCEPT_KEEP_WINDOWS="${ACCEPT_KEEP_WINDOWS:-3}"   # a pool that has landed >=1 ACCEPT stays while it keeps landing within this many windows; else rotate
WINDOW_SECS="${WINDOW_SECS:-120}"
POLL_SECS="${POLL_SECS:-20}"
START_POOL="${START_POOL:-$SN81/data/inzone_pool_custom.json}"
# --- candidate-search pruning ---
PRUNE_CANDIDATES="${PRUNE_CANDIDATES:-1}"          # rm a CANDIDATE pool that earns 0 ACCEPTs over its run
CAND_PREFIX="${CAND_PREFIX:-inzone_pool_cand_}"    # ONLY files whose basename starts with this are ever pruned
CLEAR_HOT_ON_POOL_CHANGE="${CLEAR_HOT_ON_POOL_CHANGE:-1}"  # fresh hot pool when the active pool changes
ACCEPT_LOG="${ACCEPT_LOG:-$SN81/logs/pool_accepts.log}"    # durable accept->pool attribution (append-only, survives relaunch)

mkdir -p "$SN81/logs"

# Initial rotation set (re-globbed every cycle inside supervise). The glob excludes
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
  export RELIQUARY_OMI_SHARDS="${RELIQUARY_OMI_SHARDS:-4}"   # validator runs 4 shards (cooldown idxs up to ~1.74M). 4-shard load is prefix-compatible with old 2-shard pools (idx<873k unchanged) AND adds shards 2-3 (2x prompts)
  # Qwen3.5 GRAIL proof model is the MULTIMODAL HF model (AutoModelForImageTextToText,
  # ~297 vision tensors) -> heavier than the old text-only proof model. Reduce
  # fragmentation so its forward pass fits in the headroom left by vLLM.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # --- curation knobs (read by reliquary/miner/pregen.py) ---
  export RELIQUARY_CURATE=1
  export RELIQUARY_CURATE_TARGET_K="${TARGET_K:-6}"   # STRONG model: CORRECT abundant, WRONG scarce -> k=6 (6 correct
                                          # + only 2 distinct-wrong, sigma=0.433) needs the FEWEST scarce wrongs, so the
                                          # most prompts become curatable. Safe now that miner reward == validator reward
                                          # (reconciled): k is recomputed identically -> tight sigma=0.433 won't flip out_of_zone.
  export RELIQUARY_CURATE_MARGIN=2        # validate k+2 / (8-k)+2 candidates, keep first passing
  export RELIQUARY_SAFE_P_STOP=0.10       # require a comfortable p_stop margin so validator-side linear-attn
                                          # forward drift can't push a borderline rollout under p_stop>=0.01.
  # --- seal-race / anti-batch_filled (original run_miner.sh set these; random launcher was missing them) ---
  export RELIQUARY_SUBMIT_ORDER="${SUBMIT_ORDER:-short_then_canonical}"   # fire lowest-sha256/shortest-verify first -> win the canonical seal tiebreak
  export RELIQUARY_VCOST_BUCKET="${VCOST_BUCKET:-1024}"                   # bucket same-cost completions so the sha256 canonical tiebreak engages
  export RELIQUARY_SUBMIT_DELAY_S="${SUBMIT_DELAY_S:-12}"                 # hold ~12s after window opens: winners land 8-42s in (p50 ~20s); t=0 races the prior seal -> batch_filled
  export RELIQUARY_FIRE_PER_BURST="${FIRE_PER_BURST:-2}"                  # spread the 8 across drand rounds instead of one t=0 burst
  export RELIQUARY_FIRE_PACE_S="${FIRE_PACE_S:-4}"                        # >= one 3s drand round between bursts
  # --- two-stage screen, retuned FOR curation: drop pure 8/8 & 0/8, keep the mix ---
  export RELIQUARY_SCREEN_OVERSAMPLE="${SCREEN_OVERSAMPLE:-48}"   # STRONG model: wrongs are rare -> sample deeper in the cheap screen to DETECT prompts that have ANY wrong (24 misses 1-in-N rare wrongs)
  export RELIQUARY_SCREEN_MAX_TOKENS=1024  # no-thinking: model answers + \boxed{} in <~400 tokens
  export RELIQUARY_SCREEN_MIN_TERM=4      # ~100% terminate; small floor drops pathological prompts
  export RELIQUARY_SCREEN_P_LOW=0.03      # keep prompts with >=~2 wrong present
  export RELIQUARY_SCREEN_P_HIGH="${SCREEN_P_HIGH:-0.99}"   # STRONG model: keep prompts up to 99% correct (rare wrongs) -> curatable at k=6; 0.97 wrongly dropped these high-correct-but-curatable prompts
  # Hot pool: self-built cache of screen-proven fluent+curatable prompts, re-mined to amortize discovery.
  export RELIQUARY_HOT_POOL_PATH=/root/sn81-miner/data/hot_pool.json
  export RELIQUARY_HOT_FRAC=0.5
  export RELIQUARY_HOT_CAP=4000
  export RELIQUARY_BURNED_PATH=/root/sn81-miner/data/submitted_idx.json  # persistent anti-hash_duplicate blocklist
  # Clean per-candidate attribution: clear the hot pool when the ACTIVE pool changes,
  # so a candidate's ACCEPTs come from ITS idxs, not prompts discovered under a prior pool.
  if [[ "$CLEAR_HOT_ON_POOL_CHANGE" == "1" && "$pool" != "${_LAST_POOL:-}" ]]; then
    rm -f "$RELIQUARY_HOT_POOL_PATH"
    sup_log "cleared hot pool (active pool changed -> $(basename "$pool"))"
  fi
  _LAST_POOL="$pool"
  rm -f "$LOG"
  nohup .venv/bin/python -m reliquary.cli.main mine \
    --network finney --netuid 81 --wallet-name ronnywebdev --hotkey stardev \
    --checkpoint Qwen/Qwen3.5-4B --validator-url http://86.38.238.30:8080 \
    --gpu-memory-utilization 0.65 --pool-size 48 --gen-batch 8 \
    --max-new-tokens 1024 --oversample "${OVERSAMPLE:-128}" \
    --prompt-idx-file "$pool" --two-stage \
    --log-level INFO > "$LOG" 2>&1 &
  echo $! > "$MINER_PID"
  sup_log "launched miner PID=$(cat "$MINER_PID") pool=$(basename "$pool") | log=$LOG"
}

# max window number seen in the miner log (from submitted/FIRE/verdict 'win='/'window=' lines)
latest_win()   { grep -aoE "win(dow)?=[0-9]+" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1; }
# count of validator-ACCEPTED verdicts in the current run (log truncated each launch)
accept_count() { grep -acE "verdict ACCEPTED" "$LOG" 2>/dev/null; }

# Re-glob the rotation set: picks up candidates built in the background, drops pruned ones.
reglob_pools() { mapfile -t POOLS < <(ls -1 "$SN81"/data/inzone_pool_*.json 2>/dev/null | sort); }

# Next pool (full path) whose basename sorts AFTER cur_base; wraps to the first.
# Robust to cur_base having just been deleted (it simply won't be in POOLS).
next_pool_after() {
  local cur_base="$1" p
  for p in "${POOLS[@]}"; do [[ "$(basename "$p")" > "$cur_base" ]] && { echo "$p"; return; }; done
  echo "${POOLS[0]}"
}

supervise() {
  reglob_pools
  if (( ${#POOLS[@]} == 0 )); then sup_log "FATAL: no inzone_pool_*.json under $SN81/data"; exit 1; fi
  local cur_pool="" p
  for p in "${POOLS[@]}"; do [[ "$p" == "$START_POOL" ]] && cur_pool="$p"; done
  [[ -z "$cur_pool" ]] && cur_pool="${POOLS[0]}"
  sup_log "watchdog start: ${#POOLS[@]} pools | start=$(basename "$cur_pool") | rotate after ${ROTATE_AFTER_WINDOWS} win w/o ACCEPT | prune=${PRUNE_CANDIDATES} (prefix ${CAND_PREFIX})"
  while true; do
    reglob_pools                                   # discover new candidates / drop pruned ones each cycle
    if (( ${#POOLS[@]} == 0 )); then sup_log "FATAL: all pools gone under $SN81/data"; exit 1; fi
    [[ -e "$cur_pool" ]] || cur_pool="$(next_pool_after "$(basename "$cur_pool")")"   # current gone -> advance
    local pool="$cur_pool"
    launch_miner "$pool"
    local base_win="" base_time prev_acc cur acc reason elapsed now newn i dry
    base_time=$(date +%s); prev_acc=$(accept_count); reason=""   # prev_acc=0 (log just truncated)
    while true; do
      sleep "$POLL_SECS"
      if ! kill -0 "$(cat "$MINER_PID" 2>/dev/null)" 2>/dev/null; then reason="DIED"; break; fi
      acc=$(accept_count)
      if (( acc > prev_acc )); then                # a new ACCEPT landed -> reset the no-accept counter
        cur=$(latest_win); [[ -n "$cur" ]] && base_win=$cur
        newn=$(( acc - prev_acc ))                 # accept->pool attribution: one line per new ACCEPT
        for ((i=0; i<newn; i++)); do echo "[$(date '+%F %T')] ACCEPT pool=$(basename "$pool") win=${cur:-?} max_new=${MAX_NEW_TOKENS:-1024}" >> "$ACCEPT_LOG"; done
        prev_acc=$acc; base_time=$(date +%s)
        sup_log "ACCEPT on $(basename "$pool") (accepts=$acc win=${cur:-?}) -> logged to $(basename "$ACCEPT_LOG")"
        continue
      fi
      cur=$(latest_win)
      if [[ -n "$cur" ]]; then                     # window-based rotation
        [[ -z "$base_win" ]] && base_win=$cur
        # STAY while the pool lands >=1 ACCEPT within ACCEPT_KEEP_WINDOWS (default 3): base_win is the
        # last-ACCEPT window (reset on each accept above), so cur-base_win = windows since the last accept.
        # A pool that HAS accepted uses that lenient 3-window leash; one that has NEVER accepted uses the
        # ROTATE_AFTER_WINDOWS exploration leash.
        if (( prev_acc > 0 )); then dry="$ACCEPT_KEEP_WINDOWS"; else dry="$ROTATE_AFTER_WINDOWS"; fi
        (( cur - base_win >= dry )) && { reason="no ACCEPT for $((cur - base_win)) windows (leash=$dry)"; break; }
      else                                         # silent-pool safety net: nothing submitted at all
        now=$(date +%s); elapsed=$(( now - base_time ))
        (( elapsed >= ROTATE_AFTER_WINDOWS * WINDOW_SECS )) && { reason="no submissions for ${elapsed}s (pool produced nothing)"; break; }
      fi
    done
    if [[ "$reason" == "DIED" ]]; then
      sup_log "miner process exited on $(basename "$pool") — relaunching SAME pool"
      continue                                     # cur_pool unchanged
    fi
    # ---- rotate; PRUNE only a CANDIDATE pool that earned ZERO accepts over its run ----
    acc=$(accept_count); local base; base="$(basename "$pool")"
    if [[ "$PRUNE_CANDIDATES" == "1" && "$base" == ${CAND_PREFIX}* && "$acc" -eq 0 ]]; then
      rm -f "$pool" && sup_log "PRUNED candidate $base (0 accepts over run; $reason)"
      reglob_pools                                 # exclude the just-deleted file before choosing next
      (( ${#POOLS[@]} == 0 )) && { sup_log "FATAL: pruned the last remaining pool"; exit 1; }
    fi
    cur_pool="$(next_pool_after "$base")"
    sup_log "ROTATE ($reason) on $base -> next $(basename "$cur_pool")"
  done
}

# ----------------------------- entrypoint ----------------------------- #
if [[ "${1:-}" == "--supervise" ]]; then
  supervise
  exit 0
fi

# Top-level: stop any previous watchdog (and its miner), then spawn a fresh detached one.
pkill -9 -f "run_miner_random.sh --supervise" 2>/dev/null || true
[[ -f "$SUP_PID" ]] && kill -9 "$(cat "$SUP_PID")" 2>/dev/null || true
sleep 1
echo "==== watchdog (re)start $(date '+%F %T') ====" >> "$SUP_LOG"
nohup bash "$SELF" --supervise >> "$SUP_LOG" 2>&1 &
echo $! > "$SUP_PID"
echo "watchdog PID=$(cat "$SUP_PID") | rotates inzone pools after ${ROTATE_AFTER_WINDOWS} windows w/o 'verdict ACCEPTED'"
echo "  prune candidates = ${PRUNE_CANDIDATES} (only ${CAND_PREFIX}*.json with 0 accepts are deleted; base pools kept)"
echo "  ${#POOLS[@]} pools: $(for p in "${POOLS[@]}"; do basename "$p"; done | tr '\n' ' ')"
echo "  start=$(basename "$START_POOL") | miner-log=$LOG | watchdog-log=$SUP_LOG"
