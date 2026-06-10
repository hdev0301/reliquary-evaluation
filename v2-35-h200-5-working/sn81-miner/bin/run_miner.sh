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

# ---- per-MODE presets (each individually overridable) ----
case "$MODE" in
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
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-700}"                 # decimal answers terminate in <~600 tokens
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

echo "=== MODE=$MODE | env=$ENVIRONMENT | pool=${POOL:-<broad>} | max_new=$MAX_NEW_TOKENS | k=$TARGET_K | curate=$CURATE ==="

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
