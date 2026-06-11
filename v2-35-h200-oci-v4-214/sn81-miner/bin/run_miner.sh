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
#   * MODE switch (symbolic|numeric|opencode) sets pool + TOKEN BUDGET + env +
#     curation-K together — so you can't mis-budget (e.g. 700 tokens on symbolic,
#     which truncates the ~2000-tok completions before \boxed{} -> reward 0).
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
MODE="${MODE:-symbolic}"            # symbolic | numeric | opencode

SN81="${SN81:-/root/sn81-miner}"
REPO="${REPO:-/root/reliquary}"
WALLET_NAME="${WALLET_NAME:-ronnywebdev}"
HOTKEY="${HOTKEY:-hdev0301}"
CHECKPOINT="${CHECKPOINT:-Qwen/Qwen3.5-4B}"
VALIDATOR_URL="${VALIDATOR_URL:-http://86.38.238.30:8080}"
NETWORK="${NETWORK:-finney}"
NETUID="${NETUID:-81}"

# ---- per-MODE presets: delegate to the domain module (openmath | opencode) ----
# Each module owns its strategy presets (<module>/presets.sh); this launcher is the
# shared engine. Add a strategy = drop a module dir with run.sh + presets.sh.
case "$MODE" in
  symbolic|numeric) _MODULE=openmath ;;
  opencode)         _MODULE=opencode ;;
  *) echo "FATAL: unknown MODE='$MODE' (expected: symbolic|numeric|opencode)"; exit 1 ;;
esac
_PRESETS="$SN81/$_MODULE/presets.sh"
[ -f "$_PRESETS" ] || { echo "FATAL: presets not found: $_PRESETS (is the '$_MODULE' module present?)"; exit 1; }
# shellcheck disable=SC1090
source "$_PRESETS"   # sets ENVIRONMENT/POOL/MAX_NEW_TOKENS/TARGET_K/GPU_MEM/... for $MODE
CURATE="${CURATE:-1}"               # default ON for math modes (opencode sets 0 above)
FRONTIER="${FRONTIER:-1}"           # default ON for math modes (opencode sets 0 above)
DECOOL_SNIPE="${DECOOL_SNIPE:-0}"   # default OFF for math modes (opencode sets 1 above)

TWO_STAGE="${TWO_STAGE:-1}"
OVERSAMPLE="${OVERSAMPLE:-64}"
HOT_POOL="${HOT_POOL:-$SN81/data/hot_pool.json}"

echo "=== MODE=$MODE | env=$ENVIRONMENT | pool=${POOL:-<broad>} | max_new=$MAX_NEW_TOKENS | k=$TARGET_K | curate=$CURATE ==="

# DRY_RUN=1: print the fully-resolved config and exit BEFORE clearing the hot pool,
# killing GPU procs, or launching. Safe to run while another GPU job is in flight.
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN module=$_MODULE hotkey=$HOTKEY wallet=$WALLET_NAME netuid=$NETUID"
  echo "DRY_RUN env=$ENVIRONMENT pool=${POOL:-<broad>} k=$TARGET_K curate=$CURATE frontier=$FRONTIER two_stage=$TWO_STAGE"
  echo "DRY_RUN max_new=$MAX_NEW_TOKENS oversample=$OVERSAMPLE gpu_mem=$GPU_MEM pool_size=$POOL_SIZE gen_batch=$GEN_BATCH"
  echo "DRY_RUN consensus=${OCI_CONSENSUS:-0} consensus_strict=${OCI_CONSENSUS_STRICT:-0} grader_curate=${OCI_GRADER_CURATE:-0}"
  echo "DRY_RUN (no hot-pool clear, no GPU kill, no launch)"
  exit 0
fi

# ============================ preflight ============================
cd "$REPO" || { echo "FATAL: REPO '$REPO' not found"; exit 1; }

# Pool-file GUARD: missing/empty/malformed pool -> the loader swallows the error and
# the miner silently falls back to RELIQUARY_PROMPT_SOURCES (gsm8k). Abort instead.
if [ -n "$POOL" ]; then
  [ -s "$POOL" ] || { echo "FATAL: pool '$POOL' missing/empty — build it first (build_inzone_v2.py)"; exit 1; }
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

# Kill any prior miner and free the GPU.
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
pkill -9 -f "reliquary.cli.main mine" 2>/dev/null || true
sleep 2
set -a; source scripts/.env; set +a
# Qwen3.5 GRAIL proof model is the MULTIMODAL HF model (~297 vision tensors) -> heavy.
# Reduce fragmentation so its forward fits in the headroom vLLM leaves.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vLLM concurrency cap. Default 192 capped util at ~50% while the KV cache supports
# ~649 concurrent @2048 tok — raise it to saturate continuous batching (higher GPU util
# + faster pool fill). Stays within KV (384*2048=786k < 1.33M tokens). Drop if init logs
# "max seq len larger than KV cache" or OOM (then 256).
export RELIQUARY_MAX_NUM_SEQS="${MAX_NUM_SEQS:-384}"

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
# FIRING STRATEGY (corrected 2026-06-08 after deep validator review):
# Predictive BLIND firing is STRUCTURALLY DEFEATED and is now OFF by default.
# The validator opens in TWO phases: _open_window() builds the next batcher
# INACTIVE, and /submit returns WINDOW_NOT_ACTIVE (server.py:750) until
# _activate_window() flips OPEN — which only happens AFTER the ~2.8s post-boundary
# drand fetch (service.py:1336-1338). So a POST into the [boundary, boundary+2.8s]
# slot NEVER lands; blind-fire just wastes submissions (and the seed PREDICT_L=375
# vs measured ~211 made it never fire anyway). The ONLY real lever is REACTIVE:
#   1. CO-LOCATE near the validator (sub-10ms RTT, per HELSINKI_MIGRATION.md) so we
#      detect the OPEN flip and land at the FRONT of the window-open arrival flood
#      (queue position = who finishes GRAIL among the first-8-distinct = the seal).
#   2. RELIQUARY_PRESTAGE=1: during the ~3-6s READY->OPEN gap, derive the next
#      window's randomness directly from drand and PRE-BUILD the signed submissions,
#      so the reactive fire at OPEN is a pure POST (zero project/sign/merkle on the
#      critical path). Fallback-safe: only POSTs when /state OPEN confirms matching
#      window_n + randomness + checkpoint.
#   3. SHORT completions (MAX_NEW_TOKENS) clear GRAIL faster -> our early arrivals
#      finish among the first-8-distinct sooner.
export RELIQUARY_PRESTAGE="${PRESTAGE:-1}"                       # READY-anchored reactive pre-build (the real edge). 0 to A/B against plain reactive.
export RELIQUARY_PREDICT_BLIND="${PREDICT_BLIND:-0}"            # DEFEATED by two-phase open (WINDOW_NOT_ACTIVE). Leave 0.
export RELIQUARY_PREDICT_FIRE="${PREDICT_FIRE:-0}"             # SAFE-confirm predictive: also ineffective. Leave 0.
# DECOOL-SNIPE: mine prompts as they EXIT the validator cooldown (= network-proven
# in-zone on the live checkpoint). main.py reads RELIQUARY_DECOOL_SNIPE (NOT the old
# RELIQUARY_FRONTIER_DECOOL_SNIPE, which main.py never read -> that export was dead).
# Activates pregen's _decool_sampler (works with --no-frontier; broad fallback when dry).
export RELIQUARY_DECOOL_SNIPE="$DECOOL_SNIPE"
# Submit-delay / fire-pacing: proven counterproductive (delaying past t=0 pushes us
# to a LATER drand round than the seal-trigger round -> over>0 -> batch_filled, win
# 12527: delay=20 gave over 7-11). Keep both OFF — fire all 8 reactively at OPEN.
export RELIQUARY_SUBMIT_DELAY_S="${SUBMIT_DELAY_S:-0}"   # do NOT raise.
export RELIQUARY_FIRE_PER_BURST="${FIRE_PER_BURST:-0}"   # 0: fire all at once on OPEN detect.
export RELIQUARY_FIRE_PACE_S="${FIRE_PACE_S:-4}"
# Diagnostic: log valid_submissions the instant we detect each window — tells us whether
# windows are ALREADY sealed when we see them (arrival/speed race -> need predictive) vs
# fresh (verify/canonical race -> shorter completions help). Pure logging, no behavior change.
export RELIQUARY_DETECT_PROBE="${DETECT_PROBE:-1}"
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
[ "$OCI_PROMPT_ONLY" = "1" ] && export RELIQUARY_OCI_PROMPT_ONLY=1
# OCI grader-curate (curated SUBMISSION via the real grader): pregen grades our own
# oversample through GRADER_SOCKET_PATH using the reconstructed cases cache, then curates.
[ "${OCI_GRADER_CURATE:-0}" = "1" ] && export RELIQUARY_OCI_GRADER_CURATE=1 \
  && export RELIQUARY_OCI_CASES_PATH="${OCI_CASES_PATH:-$SN81/data/oci_cases_cache.json}"
# OCI consensus (case-INDEPENDENT curation): pregen spawns its own pool of the
# DEPLOYED grader worker (no server, no cases) to bucket winners/guaranteed-0 losers.
[ "${OCI_CONSENSUS:-0}" = "1" ] && export RELIQUARY_OCI_CONSENSUS=1 \
  && export RELIQUARY_CONSENSUS_DIR="${CONSENSUS_DIR:-$SN81/opencode}" \
  && export RELIQUARY_CONSENSUS_WORKERS="${CONSENSUS_WORKERS:-16}" \
  && export RELIQUARY_CONSENSUS_TIMEOUT="${CONSENSUS_TIMEOUT:-5}"
[ "${OCI_CONSENSUS_STRICT:-0}" = "1" ] && export RELIQUARY_OCI_CONSENSUS_STRICT=1
[ "${BURN_COOLDOWN:-0}" = "1" ] && export RELIQUARY_BURN_COOLDOWN=1
[ -n "${CONSENSUS_CASES:-}" ] && export RELIQUARY_CONSENSUS_CASES="$CONSENSUS_CASES"

# ============================ launch ============================
[ -s "$SN81/logs/miner.log" ] && mv "$SN81/logs/miner.log" "$SN81/logs/miner.log.$(date -u +%Y%m%dT%H%M%SZ)"
ls -t "$SN81/logs/miner.log."[0-9]* 2>/dev/null | tail -n +6 | xargs -r rm -f --
POOL_ARG=();      [ -n "$POOL" ] && POOL_ARG=(--prompt-idx-file "$POOL")
TWO_STAGE_ARG=(); [ "$TWO_STAGE" = "1" ] && TWO_STAGE_ARG=(--two-stage)
FRONTIER_ARG=(--frontier); [ "$FRONTIER" = "0" ] && FRONTIER_ARG=(--no-frontier)

nohup .venv/bin/python -m reliquary.cli.main mine \
  --network "$NETWORK" --netuid "$NETUID" --wallet-name "$WALLET_NAME" --hotkey "$HOTKEY" \
  --checkpoint "$CHECKPOINT" --validator-url "$VALIDATOR_URL" --environment "$ENVIRONMENT" \
  --gpu-memory-utilization "$GPU_MEM" --pool-size "$POOL_SIZE" --gen-batch "$GEN_BATCH" \
  --max-new-tokens "$MAX_NEW_TOKENS" --oversample "$OVERSAMPLE" \
  "${POOL_ARG[@]}" "${TWO_STAGE_ARG[@]}" "${FRONTIER_ARG[@]}" \
  --log-level INFO >> "$SN81/logs/miner.log" 2>&1 &

echo $! > "$SN81/miner.pid"
echo "launched PID=$(cat "$SN81/miner.pid") | MODE=$MODE env=$ENVIRONMENT pool=${POOL:-<broad>} | log=$SN81/logs/miner.log"
