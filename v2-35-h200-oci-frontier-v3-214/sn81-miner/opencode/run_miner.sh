#!/bin/bash
# Reliquary OPENCODE miner launcher — thin wrapper over run_miner.sh (MODE=opencode).
# ---------------------------------------------------------------------------------
# WHY a separate launcher (vs the math modes in run_miner.sh):
#   * opencode is a DIFFERENT environment: nvidia/OpenCodeInstruct, reward is
#     VALIDATOR-AUTHORITATIVE = passed/total over HIDDEN structured cases. The miner
#     runs RELIQUARY_OCI_PROMPT_ONLY=1 and CANNOT grade locally, so runtime curation
#     against the env reward is a no-op -> we mine HONEST (CURATE=0).
#   * To still land in-zone (k correct + (8-k) wrong, sigma>=0.43) we restrict mining
#     to a SCATTER-SCREENED prompt pool built by dataprep/build_opencode_pool.py:
#     prompts whose pass-fraction is intermediate (not 0/M, not M/M) under the
#     validator's CURRENT checkpoint -> natural in-zone variance across 8 rollouts
#     without any curation. Broad sampling the raw 50k wastes windows on 0/8 & 8/8.
#
# CHECKPOINT NOTE: the miner ignores --checkpoint and loads the validator's published
#   checkpoint from /state (currently R0mAI/reliquary-sn-v23, hot-reloaded as the
#   frontier advances). The scatter pool is therefore CHECKPOINT-SPECIFIC: rebuild it
#   whenever ckpt_n advances:
#     cd /root/reliquary && .venv/bin/python /root/sn81-miner/dataprep/build_opencode_pool.py \
#       --checkpoint <current local snapshot path> --cases-source reconstruct --m 16
#
# Usage:  bash /root/sn81-miner/opencode/run_miner.sh
#         POOL="" bash opencode/run_miner.sh      # force broad sampling
#         PREDICT_BLIND=0 bash opencode/run_miner.sh   # disable predictive seal fire
#
# All opencode state (pool, hot-pool, burned blocklist) lives under opencode/data/ so it
# never mixes with the openmath modes' shared data/. Build the pool: opencode/build_pool.sh
set -u
SN81="${SN81:-/root/sn81-miner}"
OCDATA="$SN81/opencode/data"
mkdir -p "$OCDATA"

# ============================ opencode wiring (overridable) ============================
export MODE=opencode
# Disable vLLM prefix caching: on Qwen3.5 GDN/Mamba it uses the experimental
# Mamba-cache 'align' mode that deadlocks EngineCore mid-gen (GPU 0%, miner hangs).
export RELIQUARY_PREFIX_CACHING="${RELIQUARY_PREFIX_CACHING:-0}"
# Lower vLLM concurrency (default 192): the GDN EngineCore deadlock hits at the start
# of large prefill batches; fewer concurrent seqs reduces scheduler pressure/hang risk.
export RELIQUARY_MAX_NUM_SEQS="${RELIQUARY_MAX_NUM_SEQS:-512}"   # was 64 (~35% util). KV holds ~718 concurrent; 512 -> 100% PEAK gen util. Deadlock-free now that prefix-caching+async are OFF (old 192/384 deadlock was WITH prefix-caching). Watchdog = backup if a hang recurs.
# Permanent per-prompt cooldown (validator BATCH_PROMPT_COOLDOWN_WINDOWS=1e6) makes any
# WON prompt dead forever, but the hot pool re-fires won prompts (stale /state cooldown
# race) -> prompt_in_cooldown (proven live: win=12542 re-fired cooled idx 27814). On a
# difficulty-screened pool there are enough FRESH intermediate prompts that the hot-pool
# re-mine is pure liability, so disable it (0) and delete the poisoned cache on restart:
#   rm -f opencode/data/hot_pool.json
export HOT_FRAC="${HOT_FRAC:-0.0}"
export ENVIRONMENT="${ENVIRONMENT:-opencodeinstruct}"
# CURATE=1 is now viable (we have a local grader): it selects an in-zone 8-subset targeting k=4
# -> sigma=0.5, a built-in MARGIN over the 0.43 floor. The validator grades on its OWN pinned cases,
# so our local sigma is only a prediction; the margin defends against that structural divergence
# (the #2 reject = out_of_zone). Honest mode (CURATE=0) gated at the bare floor -> no resilience.
export CURATE="${CURATE:-1}"

# REAL REWARD PATH: prompt-only mode has no test cases -> compute_reward()==0 -> 0 in-zone groups.
# Instead load a LOCAL structured subset (mirror + reconstructed cases, index-aligned) and route
# compute_reward() through a local grader server (exact validator parity, non-gVisor). Set
# OCI_PROMPT_ONLY=1 to revert to the (non-producing) prompt-only mode.
export OCI_PROMPT_ONLY="${OCI_PROMPT_ONLY:-0}"
LOCAL_SUBSET="${OCI_SUBSET_REPO:-$SN81/opencode/data/oci_local_subset}"
if [ "$OCI_PROMPT_ONLY" = "0" ]; then
  [ -d "$LOCAL_SUBSET" ] || { echo "FATAL: local subset '$LOCAL_SUBSET' missing — build it: .venv/bin/python $SN81/opencode/build_local_subset.py"; exit 1; }
  export RELIQUARY_OCI_SUBSET_REPO="$LOCAL_SUBSET"
  bash "$SN81/opencode/grader.sh" start || { echo "FATAL: local grader failed to start"; exit 1; }
  echo "[opencode] real-reward mode: subset=$LOCAL_SUBSET + local grader"
fi
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"            # top-miner pattern: code p90~490-867; 1024 covers the tail, faster gen -> more pregen throughput -> more groups/window.
export SCREEN_MAX_TOKENS="${SCREEN_MAX_TOKENS:-768}"       # cheaper cheap-screen -> faster screening throughput (top-miner pattern)
# Tighten the cheap-screen pass-fraction band (default 0.03/0.97 in bin/run_miner.sh).
# At ckpt 873 the unscreened full-gradeable pool is ~85-95% allcorr (8/8 too-easy): the
# screen dropped 14-22 of every 24 prompts and only 1-4 deep-curated -> store starved ->
# late fires -> batch_filled=57%. Narrowing to [0.10,0.80] culls the borderline-easy band
# at the CHEAP 24-rollout screen so deep-mine budget goes to genuinely ~50/50 prompts ->
# more groups ready at window-open (the single ACCEPT won by firing at over=0/t=0).
export SCREEN_P_LOW="${SCREEN_P_LOW:-0.10}"
export SCREEN_P_HIGH="${SCREEN_P_HIGH:-0.80}"
export TARGET_K="${TARGET_K:-4}"                           # k=4 = validator zone-CENTER (in-zone=2..6/8): 4 pass + 4 fail -> sigma=0.5, fat margin over SIGMA_MIN=0.43. Requires the DIFFICULTY-SCREENED pool (build_pool.sh) where wrong-side is abundant; on a SATURATED pool prefer k=6 (only 2 wrong needed).
# Pure-binary curation (match uid116 "100% binary, zero partials"): keep ONLY full-pass
# (r>=1.0) and full-fail (r<=0.0) rollouts; discard partials (their hidden-case regrade
# drifts sigma off 0.433). Pairs with the continuous-reward curation in pregen.py.
export RELIQUARY_CORRECT_BAND="${RELIQUARY_CORRECT_BAND:-1.0}"
export RELIQUARY_WRONG_BAND="${RELIQUARY_WRONG_BAND:-0.0}"
export GPU_MEM="${GPU_MEM:-0.65}"
export POOL_SIZE="${POOL_SIZE:-48}"
export GEN_BATCH="${GEN_BATCH:-24}"
export OVERSAMPLE="${OVERSAMPLE:-64}"                      # 64 (was 48): on a STRONG live ckpt the wrong side is scarce; current avg ~26 completions/prompt (healthy ~57). More rollouts/prompt = more chances to land the scarce side -> more curatable groups (_choose_k satisfiable). Revert to 48 if groups/window does NOT rise (slower deep-mine per prompt).

# Keep opencode runtime state OUT of the shared data/ (openmath modes use that).
export HOT_POOL="${HOT_POOL:-$OCDATA/hot_pool.json}"          # opencode-only screen-proven cache
export BURNED_PATH="${BURNED_PATH:-$OCDATA/submitted_idx.json}"  # opencode-only anti-dup blocklist
# #2 persistent COOLED blocklist: engine.py loads this and excludes it from selection+firing,
# and appends every newly-observed prompt_in_cooldown idx (covers prompts /state omits). Shared
# with build_pool.sh's #1 cooldown filter so the pool and the live miner exclude the same set.
export RELIQUARY_COOLED_IDX_PATH="${RELIQUARY_COOLED_IDX_PATH:-$OCDATA/cooled_idx.json}"
# Persisted online-learning frontier model. Default in cli/main.py is /root/frontier_model.npz;
# relocate it under the workspace so all OCI state lives in opencode/data/ (takes effect on relaunch).
export RELIQUARY_FRONTIER_MODEL="${RELIQUARY_FRONTIER_MODEL:-$OCDATA/frontier_model.npz}"
# Adaptive allcorr-burn (treadmill fix): pregen burns pool prompts that screen 8/8 at
# the LIVE checkpoint so the screened pool self-stays-intermediate as the checkpoint
# drifts between full rebuilds (in-memory, cleared on ckpt advance, never over-prunes).
export RELIQUARY_ALLCORR_BURN="${RELIQUARY_ALLCORR_BURN:-1}"

# Auto-wire the scatter pool: use it once built, else fall back to BROAD sampling so this
# launcher works before the pool exists (and never silently mines gsm8k via the loader fallback).
DEFAULT_POOL="$OCDATA/inzone_pool_opencode.json"
if [ -n "${POOL:-}" ]; then
  export POOL                                                # explicit user override (path or "")
elif [ -s "$DEFAULT_POOL" ]; then
  export POOL="$DEFAULT_POOL"
  N=$(/root/reliquary/.venv/bin/python -c "import json;print(len(json.load(open('$DEFAULT_POOL'))))" 2>/dev/null || echo "?")
  echo "[opencode] scatter-screened pool: $DEFAULT_POOL ($N idxs)"
else
  export POOL=""
  echo "[opencode] scatter pool not built yet -> BROAD sampling over the 50k subset"
  echo "[opencode]   build it: bash $SN81/opencode/build_pool.sh"
fi

# ============================ seal-race timing (env-agnostic) ============================
# THE path to real `verdict ACCEPTED`. Without this we arrive at over=0 and lose batch_filled
# to miners who fire PREDICTIVELY (over<0), filling the 8 shared seal slots before the trigger.
# PREDICT_BLIND=1 pre-builds and fires into the predicted post-boundary window. MIN_WINDOWS=2
# because we only see ~3 windows / 15 min, so the engine default (3) never learns window-length L.
# Toggle off:  PREDICT_BLIND=0 bash run_miner_opencode.sh
export RELIQUARY_PREDICT_BLIND="${PREDICT_BLIND:-1}"
export RELIQUARY_PREDICT_MIN_WINDOWS="${PREDICT_MIN_WINDOWS:-2}"   # 2 (was 1): MIN_WINDOWS=1 trusted a SINGLE window-length sample -> L learned 327 from one 16-min outage delta (true cadence ~300 rounds/~14min); fire never engaged, every sub arrived over>=0 -> batch_filled. The engine medians a 16-deep deque and self-converges; 2 just refuses to trust the first noisy sample. Do NOT seed a fixed L=19 (that is ~15x too small -> mis-fire every window).
export RELIQUARY_PREDICT_LEAD_MS="${PREDICT_LEAD_MS:-1800}"   # 1800 = documented max (stay clear of future_round): stage the prebuilt group earlier so a ready group lands at over<=0 at the boundary. If WINDOW_MISMATCH appears, revert to 1500.
export RELIQUARY_PREDICT_POST_MS="${PREDICT_POST_MS:-300}"
# Decool-snipe: freshly-decooled prompts are known-in-zone and newly submittable (low collision).
# Free stream of low-risk accepts; falls back to the default sampler when the decool queue is empty.
export RELIQUARY_FRONTIER_DECOOL_SNIPE="${DECOOL_SNIPE:-0}"   # 0 (was 1): cooldown is PERMANENT (BATCH_PROMPT_COOLDOWN_WINDOWS=1e6), so /state "decooled" signals are stale-race noise -> snipe re-fires permanently-cooled prompts -> prompt_in_cooldown rejects that waste the 8/window submit cap. With a broad fresh pool there are plenty of never-cooled prompts; the snipe is pure liability.

echo "=== OPENCODE launcher | pool=${POOL:-<broad>} | predict_blind=$RELIQUARY_PREDICT_BLIND (min_windows=$RELIQUARY_PREDICT_MIN_WINDOWS lead=${RELIQUARY_PREDICT_LEAD_MS}ms) ==="
exec bash "$SN81/bin/run_miner.sh"
