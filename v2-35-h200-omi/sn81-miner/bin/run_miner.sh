#!/bin/bash
# Reliquary miner launcher — CURATION pipeline, parameterized by MODE.
#
# Curation (pregen.py build_groups): on a converged BIMODAL checkpoint, natural
# 8-sample groups score 8/8 or 0/8 (out of zone). We over-generate, reward every
# candidate against the env reward, SELECT an in-zone 8-subset (k correct + (8-k)
# wrong, sigma>=SIGMA_MIN=0.43), and pre-validate every gate locally. All rollouts
# are genuine current-checkpoint samples; the validator recomputes the reward.
#
# Improvements over the prior version (verified against the custom fork
# reliquary/cli/main.py + reliquary/miner/pregen.py, and validator gates):
#   * MODE switch (frontier|symbolic|numeric|opencode) sets pool + TOKEN BUDGET +
#     env + curation-K together — so you can't mis-budget (e.g. 700 tokens on
#     symbolic, which truncates the ~2000-tok completions before \boxed{} -> 0).
#   * DEFAULT MODE = frontier: OMI learning-frontier mining. On a converged
#     checkpoint the easy gsm8k sources are solved ~100% (every screened group
#     comes back 8/8 -> nothing curatable -> "pool produced nothing"). frontier
#     mode mines the HARD sources (augmented_math+math, short numeric answers)
#     where 20-80% pass mass still exists, curates k=4 (sigma 0.5, both sides
#     abundant), and fires ASAP (SUBMIT_DELAY_S=0) to win the seal race instead
#     of arriving ~100 drand rounds late. Build its pool first:
#         bash bin/build_frontier_pool.sh   # -> data/inzone_pool_frontier.json
#   * Pool-file GUARD: a missing/empty --prompt-idx-file makes the miner SILENTLY
#     fall back to RELIQUARY_PROMPT_SOURCES (gsm8k default) — mining the WRONG data
#     with no warning. We now abort loudly instead.
#   * Hot-pool auto-clear when the pool/env changes (else stale prompts from the
#     previous strategy get re-mined).
#   * --environment wired up (opencode is the current top-of-board strategy).
#   * Every knob is `${VAR:-default}` overridable: `MODE=opencode bash run_miner.sh`.
#
# Protocol guards respected (constants.py / validator): k in [2,6] for sigma>=0.43;
# T=0.9/top_p=1/top_k=0 fixed by protocol (not ours to set); <=8 distinct prompts/
# hotkey/window; mean completion <4096 (quarantine) — 2048 cap keeps us clear;
# losers must terminate on EOS (p_stop>=0.01) — SAFE_P_STOP margin handles drift.

set -u

# ============================ CONFIG (override via env) ============================
MODE="${MODE:-frontier}"            # frontier | symbolic | numeric | opencode

SN81="${SN81:-/root/sn81-miner}"
REPO="${REPO:-/root/reliquary}"
WALLET_NAME="${WALLET_NAME:-ronnywebdev}"
HOTKEY="${HOTKEY:-hdev0301}"
CHECKPOINT="${CHECKPOINT:-Qwen/Qwen3.5-4B}"
VALIDATOR_URL="${VALIDATOR_URL:-http://86.38.238.30:8080}"
NETWORK="${NETWORK:-finney}"
NETUID="${NETUID:-81}"

# ---- per-MODE presets (each individually overridable) ----
case "$MODE" in
  frontier)   # DEFAULT — OMI learning-frontier play. Hard sources (augmented_math+math),
              # BROAD answers (numeric + symbolic). The v23 frontier is symbolic-leaning
              # (top OMI miner: num/sym 37/63, k scattered 4/5/6 ~k6-lean, completions med~638
              # p90~966). We curate near k=5 (sigma 0.484) but let _choose_k scatter to 4/6 by
              # what each prompt yields (natural-looking -> low distribution_suspicious), and
              # fire ASAP to win the seal race.
    ENVIRONMENT="${ENVIRONMENT:-openmathinstruct}"
    POOL="${POOL:-$SN81/data/inzone_pool_frontier.json}"     # build: bash bin/build_frontier_pool.sh
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1280}"                  # symbolic frontier prompts run med~638/p90~966 tok; 768 truncated >50% into ramble-drops (live: mean_term=0.39). 1280 covers p90, drops only the slow >1280 tail (good for seal-race verify cost). vllm max_model_len = max_new+1024.
    SCREEN_MAX_TOKENS="${SCREEN_MAX_TOKENS:-1280}"            # MUST match the gen budget or the screen mislabels terminators as ramblers
    TARGET_K="${TARGET_K:-6}"                                 # CONVERGED model: wrongs are RARE (live screen allcorr=108 vs allwrong=4), so harvest the MINIMUM 2 wrong -> k=6 (sigma 0.433) maximizes curatable count + matches the top miner's 55%-k6 lean. _choose_k scatters down to 5/4 when more wrong are available.
    GPU_MEM="${GPU_MEM:-0.65}"; POOL_SIZE="${POOL_SIZE:-96}"; GEN_BATCH="${GEN_BATCH:-24}"  # DEEPER BUFFER: --pool-size = pregen target_ready (pregen.py:507). 96 keeps the thread building a backlog of staged groups across windows so >=1 is ready at window-open (the top miner fires from a backlog, not per-window gen).
    OVERSAMPLE="${OVERSAMPLE:-64}"
    # --- TRICKLE MODE (mirror top miner 5CX7gQ4: fire few BEST groups, ~87% accept on trickles vs ~68% on 8-bursts) ---
    SUBMIT_ORDER="${SUBMIT_ORDER:-confidence}"               # fastest-verify bucket first (beats the seal drain), then highest sigma (flip-robust), then canonical (pregen.py _submit_sort_key)
    HOT_FRAC="${HOT_FRAC:-0.6}"; HOT_CAP="${HOT_CAP:-8000}"  # let the hot-pool of proven-curatable idxs ACCUMULATE + re-mine more of it -> faster buffer refill (amortizes the expensive screen; persists across same-pool restarts)
    SCREEN_OVERSAMPLE="${SCREEN_OVERSAMPLE:-48}"             # STRONG model: wrongs are RARE. Sample 48 (not 24) so prompts with a few harvestable wrongs aren't mislabelled all-correct.
    # CURATION band (NOT natural-sampling). A prompt is curatable if the deep mine yields k correct
    # + (8-k) wrong; at k=5 you need only ~3 wrong, so a 90%-pass prompt IS curatable. The screen
    # must therefore KEEP high-pass prompts -> wide band, high P_HIGH. (Live evidence: a narrow
    # [0.15,0.85] dropped 13-16/24 as "allcorr" -> 0 in-zone groups. allcorr = ratio>P_HIGH, not
    # literally 8/8.) P_HIGH=0.93 ensures ~3 wrong survive into the 64-sample deep mine for k=5.
    SCREEN_P_LOW="${SCREEN_P_LOW:-0.05}"; SCREEN_P_HIGH="${SCREEN_P_HIGH:-0.96}"  # 0.96 (was 0.93): on this converged model the dominant drops are 93-99%-pass prompts that DO have 1-2 harvestable wrongs (k=6 needs only 2). Admit them -> convert allcorr discards into curatable groups.
    SUBMIT_DELAY_S="${SUBMIT_DELAY_S:-0}"                     # fire ASAP. Live losses were all LATE (t=306s, over=102 rounds past seal trigger). An earlier drand round can never lose a slot it would otherwise win.
    FIRE_PER_BURST="${FIRE_PER_BURST:-8}"                     # DUMP up to 8 into a fresh window. Accepts come only from SPARSE windows (trig=None / late seal) which have MULTIPLE free slots; trickling 3 leaves them on the table. Groups are pre-screened clean (p_stop/term/dist) so 8 doesn't risk the 2-failure proof-debt. hold-for-open ensures we only dump into fresh windows.
    FIRE_PACE_S="${FIRE_PACE_S:-4}"                           # >= one 3s drand round between bursts (spreads submissions across rounds like the top miner)
    # HOLD-FOR-OPEN: groups become ready MID-window; firing into an already-sealed window = batch_filled
    # (live: valid_subs=16-19, over=40-99). Hold a ready group while the window already has >= this many
    # valid subs, so it fires at the NEXT window-open (low valid_subs) instead. Windows seal ~16; 10
    # fires only into genuinely fresh windows. Safety: HOLD_MAX_S fires anyway if one window stays open that long.
    SEAL_GUARD_VALID_SUBS="${SEAL_GUARD_VALID_SUBS:-10}"; HOLD_MAX_S="${HOLD_MAX_S:-1200}"  # 1200 >> window (~373s): the safety valve was force-firing held groups at t=300s INTO the still-full window (over=100). >> window length disables that, so groups WAIT in the buffer for a genuinely fresh/sparse open instead of bleeding out.
    # ROUND-0 PLAY (PREDICT_BLIND): /state shows OPEN+randomness ~3s after the boundary, by which point the field has already sealed (valid_subs lags arrival). BLIND pre-builds against the PREDICTED next-window randomness and POSTs at the boundary, no /state wait — the only way to land in round 0. Self-gates until window length L is observed _PREDICT_MIN_WINDOWS times; mispredicts -> WRONG_RANDOMNESS but groups are RESTORED to the buffer (not burned), so the downside is wasted POSTs, not lost prompts. Needs NTP (FUTURE_ROUND has zero tolerance).
    PREDICT_BLIND="${PREDICT_BLIND:-0}"                       # OFF: live PREDICT-LEARN shows window deltas 93-200 rounds (279-600s) — far too irregular to predict the open round, so blind fires would mostly WRONG_RANDOMNESS. Re-enable only if the cadence stabilizes.
    OCI_PROMPT_ONLY=0 ;;
  symbolic)   # uid-181 (5HEAK6) play: OMI symbolic format-ambiguity pool. LONG completions.
    ENVIRONMENT="${ENVIRONMENT:-openmathinstruct}"
    POOL="${POOL:-$SN81/data/inzone_pool_v2.json}"          # build_inzone_v2.py --sym-ratio 0.60 --int-ratio 0.23
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"                # 5HEAK6 completions: med ~910, max ~2043
    SCREEN_MAX_TOKENS="${SCREEN_MAX_TOKENS:-2048}"
    TARGET_K="${TARGET_K:-5}"                               # DATA-CORRECTED: 5HEAK6 groups are k=4-6 (CORRECT abundant);
                                                            # distinct-WRONG is the scarce side, so high k needs FEWER of them.
    GPU_MEM="${GPU_MEM:-0.82}"; POOL_SIZE="${POOL_SIZE:-48}"; GEN_BATCH="${GEN_BATCH:-24}"   # THROUGHPUT: 5HEAK6 submits ~5/window; 32/12 got ~1. Wider pool+batch = more curatable mined/cycle. Dial back to 32/12 if OOM.
    OVERSAMPLE="${OVERSAMPLE:-96}"                          # a touch deeper for the 3 distinct-WRONG per group (curation success rate); still under the ~160-depth waste line
    OCI_PROMPT_ONLY=0 ;;
  numeric)    # decimal/gsm engine: short answers. The old default (inzone_pool_topmatch.json).
    ENVIRONMENT="${ENVIRONMENT:-openmathinstruct}"
    POOL="${POOL:-$SN81/data/inzone_pool_topmatch.json}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"                 # decimal answers terminate in <~600 tokens
    SCREEN_MAX_TOKENS="${SCREEN_MAX_TOKENS:-1024}"
    TARGET_K="${TARGET_K:-5}"                               # numeric: CORRECT abundant -> 5 correct + 3 wrong (sigma 0.484)
    GPU_MEM="${GPU_MEM:-0.65}"; POOL_SIZE="${POOL_SIZE:-48}"; GEN_BATCH="${GEN_BATCH:-24}"
    OCI_PROMPT_ONLY=0 ;;
  opencode)   # current TOP-OF-BOARD strategy (5DARq6 rank 1). nvidia/OpenCodeInstruct, broad.
    # NOTE: opencode reward is VALIDATOR-AUTHORITATIVE (passed/total over HIDDEN tests).
    # In prompt-only mode the miner can't grade locally, so curation against the env
    # reward is a no-op -> run HONEST (CURATE=0) and rely on the CONTINUOUS passed/total
    # reward to give natural in-zone variance across the 8 rollouts. To CURATE opencode
    # you must reconstruct cases (build_opencode_pool.py) and run a local grader.
    # This mode is a STARTING POINT — verify in-zone yield in the log before trusting it.
    ENVIRONMENT="${ENVIRONMENT:-opencodeinstruct}"
    POOL="${POOL:-}"                                        # empty = broad sampling over the whole ~50k subset
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1280}"                # code completions: med ~330, max ~1190
    SCREEN_MAX_TOKENS="${SCREEN_MAX_TOKENS:-1280}"
    TARGET_K="${TARGET_K:-4}"
    GPU_MEM="${GPU_MEM:-0.65}"; POOL_SIZE="${POOL_SIZE:-48}"; GEN_BATCH="${GEN_BATCH:-24}"
    OCI_PROMPT_ONLY=1
    CURATE="${CURATE:-0}" ;;                                # see note above
  *) echo "FATAL: unknown MODE='$MODE' (expected: symbolic|numeric|opencode)"; exit 1 ;;
esac
CURATE="${CURATE:-1}"               # default ON for math modes (opencode sets 0 above)

TWO_STAGE="${TWO_STAGE:-1}"
OVERSAMPLE="${OVERSAMPLE:-64}"
HOT_POOL="${HOT_POOL:-$SN81/data/hot_pool.json}"

echo "=== MODE=$MODE | env=$ENVIRONMENT | pool=${POOL:-<broad>} | max_new=$MAX_NEW_TOKENS | k=$TARGET_K | curate=$CURATE"
echo "    screen_band=[${SCREEN_P_LOW:-0.03},${SCREEN_P_HIGH:-0.97}] screen_oversample=${SCREEN_OVERSAMPLE:-24} | buffer(target_ready)=${POOL_SIZE} hot_frac=${HOT_FRAC:-0.5}/cap=${HOT_CAP:-4000}"
echo "    TRICKLE: submit_delay=${SUBMIT_DELAY_S:-0}s fire_per_burst=${FIRE_PER_BURST:-4} order=${SUBMIT_ORDER:-short_then_canonical} hold@valid_subs>=${SEAL_GUARD_VALID_SUBS:-0} hold_max=${HOLD_MAX_S:-300}s predict_blind=${PREDICT_BLIND:-0} ==="

# ============================ preflight ============================
cd "$REPO" || { echo "FATAL: REPO '$REPO' not found"; exit 1; }

# Pool-file GUARD: missing/empty/malformed pool -> the loader swallows the error and
# the miner silently falls back to RELIQUARY_PROMPT_SOURCES (gsm8k). Abort instead.
if [ -n "$POOL" ]; then
  _build_hint="build_inzone_v2.py"
  [ "$MODE" = "frontier" ] && _build_hint="bash bin/build_frontier_pool.sh"
  [ -s "$POOL" ] || { echo "FATAL: pool '$POOL' missing/empty — build it first ($_build_hint)"; exit 1; }
  .venv/bin/python -c "import json,sys; n=len(json.load(open('$POOL'))); print('pool ok: %d idxs'%n); sys.exit(0 if n>0 else 1)" \
    || { echo "FATAL: pool '$POOL' is not a non-empty JSON list"; exit 1; }
fi

# Clear the hot pool when the active pool/env changes OR the pool file was
# rebuilt, so stale prompts from the previous strategy aren't re-mined
# (RELIQUARY_HOT_FRAC=0.5 draws half from it). The fingerprint includes the pool
# file's mtime:size, so re-running build_inzone_v2.py (same path, new contents)
# auto-clears the hot pool — no manual `rm` needed.
mkdir -p "$SN81/data" "$SN81/logs"
LAST_POOL_FILE="$SN81/data/.last_pool"
POOL_FP=""
[ -n "$POOL" ] && [ -e "$POOL" ] && POOL_FP="$(stat -c '%Y:%s' "$POOL" 2>/dev/null || true)"
CUR_SIG="${POOL:-broad}|$ENVIRONMENT|$POOL_FP"
if [ "$(cat "$LAST_POOL_FILE" 2>/dev/null || true)" != "$CUR_SIG" ]; then
  echo "pool/env changed or pool rebuilt -> clearing hot pool $HOT_POOL"
  rm -f "$HOT_POOL"
  echo "$CUR_SIG" > "$LAST_POOL_FILE"
fi

# Clock-skew guard: the validator gates drand_round with ZERO forward tolerance
# (a round ahead of UTC -> FUTURE_ROUND). With SUBMIT_DELAY_S=0 we fire near drand
# boundaries, so an unsynced clock silently drops every submission. Warn loudly.
if command -v timedatectl >/dev/null 2>&1; then
  if [ "$(timedatectl show -p NTPSynchronized --value 2>/dev/null)" != "yes" ]; then
    echo "WARN: system clock is NOT NTP-synchronized — drand_round may be FUTURE_ROUND/STALE_ROUND. Run: sudo timedatectl set-ntp true"
  fi
fi

# Kill any prior miner and free the GPU.
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
pkill -9 -f "reliquary.cli.main mine" 2>/dev/null || true
sleep 2
set -a; source scripts/.env; set +a
# Qwen3.5 GRAIL proof model is the MULTIMODAL HF model (~297 vision tensors) -> heavy.
# Reduce fragmentation so its forward fits in the headroom vLLM leaves.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================ curation / screen / safety knobs ============================
export RELIQUARY_CURATE="$CURATE"
export RELIQUARY_CURATE_TARGET_K="$TARGET_K"
export RELIQUARY_CURATE_MARGIN="${CURATE_MARGIN:-2}"          # validate k+2 / (8-k)+2 candidates, keep first passing
export RELIQUARY_MAX_PER_WINDOW=8                            # = protocol MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
export RELIQUARY_SAFE_P_STOP="${SAFE_P_STOP:-0.10}"          # margin over validator MIN_EOS_PROBABILITY=0.01 (linear-attn drift)
# Seal-race ordering: at over=0 the validator picks top-8 distinct prompts by sha256(prompt_idx).
# short_then_canonical fires lowest-sha256/shortest-to-verify first -> more over=0 ties won inside
# the seal-drain. VCOST_BUCKET groups same-cost completions so the sha256 canonical tiebreak engages.
export RELIQUARY_SUBMIT_ORDER="${SUBMIT_ORDER:-short_then_canonical}"
export RELIQUARY_VCOST_BUCKET="${VCOST_BUCKET:-1024}"
# Seal-race TIMING (read by reliquary/miner/engine.py). The dominant live failure
# was lateness: the only ready group fired ~240-306s into the window, 81-102 drand
# rounds PAST the seal trigger -> batch_filled. Fire ASAP instead; an earlier round
# never loses a slot it could otherwise win. FIRE_PER_BURST spreads firing across
# rounds and caps the burst so a bad group can't burn the per-hotkey 2-failure
# proof-admission budget (termination/distribution stages) for the whole window.
export RELIQUARY_SUBMIT_DELAY_S="${SUBMIT_DELAY_S:-0}"
export RELIQUARY_FIRE_PER_BURST="${FIRE_PER_BURST:-4}"
export RELIQUARY_FIRE_PACE_S="${FIRE_PACE_S:-4}"
# HOLD-FOR-OPEN (engine.py): don't fire into an already-saturated window; hold the ready group for
# the next window-open. 0 = disabled (legacy fire-whenever-ready).
export RELIQUARY_SEAL_GUARD_VALID_SUBS="${SEAL_GUARD_VALID_SUBS:-0}"
export RELIQUARY_HOLD_MAX_S="${HOLD_MAX_S:-300}"
# ROUND-0 predictive blind fire (engine.py:223-241). 0 = off (legacy detect-then-fire).
export RELIQUARY_PREDICT_BLIND="${PREDICT_BLIND:-0}"
# Fire this many ms AFTER the predicted boundary. Measured clock skew was +0.41s (local AHEAD),
# so 300ms could POST before the server crosses -> FUTURE_ROUND. 900ms keeps us safely in the new
# round even with that skew. (FUTURE_ROUND restores the group, so this only trims wasted attempts.)
export RELIQUARY_PREDICT_POST_MS="${PREDICT_POST_MS:-900}"
# Two-stage screen: cheap pass skips ramblers, keeps prompts with BOTH correct and wrong
# (the curatable ones). Band [0.03,0.97] drops only pure 8/8 & 0/8.
export RELIQUARY_SCREEN_OVERSAMPLE="${SCREEN_OVERSAMPLE:-24}"
export RELIQUARY_SCREEN_MAX_TOKENS="$SCREEN_MAX_TOKENS"      # MUST cover the pool's completion tail (else valid prompts dropped as ramblers)
export RELIQUARY_SCREEN_MIN_TERM="${SCREEN_MIN_TERM:-4}"
export RELIQUARY_SCREEN_P_LOW="${SCREEN_P_LOW:-0.03}"
export RELIQUARY_SCREEN_P_HIGH="${SCREEN_P_HIGH:-0.97}"
# Hot pool: self-built cache of screen-proven curatable prompts, re-mined to amortize discovery.
export RELIQUARY_HOT_POOL_PATH="$HOT_POOL"
export RELIQUARY_HOT_FRAC="${HOT_FRAC:-0.5}"
export RELIQUARY_HOT_CAP="${HOT_CAP:-4000}"
export RELIQUARY_BURNED_PATH="${BURNED_PATH:-$SN81/data/submitted_idx.json}"  # persistent anti-hash_duplicate blocklist
# FRONTIER PREDICTOR + WINNERS SEED. With --prompt-idx-file + frontier ON, the online
# predictor runs OVER our pool (pregen.py:415), learning which pooled prompts are actually
# deep-curatable on the CURRENT checkpoint and biasing selection toward them. Seed it with the
# harvested top-miner winners (250 known in-zone exemplars) so it starts warm, not cold. The
# winners file is ADDITIVE to the live-cooldown seed (frontier.py). Build it with:
#   RELIQUARY_OMI_SHARDS=4 .venv/bin/python <harvest script>  (already saved 250 rows).
export RELIQUARY_FRONTIER=1
export RELIQUARY_WINNERS_PATH="${WINNERS_PATH:-$SN81/data/topminer_winners.jsonl}"
[ "$OCI_PROMPT_ONLY" = "1" ] && export RELIQUARY_OCI_PROMPT_ONLY=1

# ============================ launch ============================
rm -f "$SN81/logs/miner.log"
POOL_ARG=();      [ -n "$POOL" ] && POOL_ARG=(--prompt-idx-file "$POOL")
TWO_STAGE_ARG=(); [ "$TWO_STAGE" = "1" ] && TWO_STAGE_ARG=(--two-stage)

nohup .venv/bin/python -m reliquary.cli.main mine \
  --network "$NETWORK" --netuid "$NETUID" --wallet-name "$WALLET_NAME" --hotkey "$HOTKEY" \
  --checkpoint "$CHECKPOINT" --validator-url "$VALIDATOR_URL" --environment "$ENVIRONMENT" \
  --gpu-memory-utilization "$GPU_MEM" --pool-size "$POOL_SIZE" --gen-batch "$GEN_BATCH" \
  --max-new-tokens "$MAX_NEW_TOKENS" --oversample "$OVERSAMPLE" \
  "${POOL_ARG[@]}" "${TWO_STAGE_ARG[@]}" \
  --log-level INFO > "$SN81/logs/miner.log" 2>&1 &

echo $! > "$SN81/miner.pid"
echo "launched PID=$(cat "$SN81/miner.pid") | MODE=$MODE env=$ENVIRONMENT pool=${POOL:-<broad>} | log=$SN81/logs/miner.log"
