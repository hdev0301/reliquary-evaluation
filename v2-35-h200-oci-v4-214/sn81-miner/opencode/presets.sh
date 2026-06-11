#!/usr/bin/env bash
# OpenCode preset (MODE=opencode) for the shared launcher.
#
# Sourced by ../bin/run_miner.sh AFTER it resolves $MODE and $SN81; this file sets
# ENVIRONMENT / POOL / token budget plus the GRADER_CURATE / CONSENSUS / CONSENSUS_STRICT
# opt-in logic. Every value stays ${VAR:-default} so each knob remains env-overridable.
#
# OpenCode tooling lives alongside this file in ../opencode/: build_opencode_pool.py
# (pool builder), consensus.py + _grader_worker_head.py (consensus curation), run.sh
# (wrapper), auto_pool.sh (pool supervisor), README.md.

# current TOP-OF-BOARD strategy (5DARq6 rank 1). nvidia/OpenCodeInstruct, broad.
# NOTE: opencode reward is VALIDATOR-AUTHORITATIVE (passed/total over HIDDEN tests).
# In prompt-only mode the miner can't grade locally, so curation against the env
# reward is a no-op -> run HONEST (CURATE=0) and rely on the CONTINUOUS passed/total
# reward to give natural in-zone variance across the 8 rollouts. To CURATE opencode
# you must reconstruct cases (opencode/build_opencode_pool.py) and run a local grader.
# This mode is a STARTING POINT — verify in-zone yield in the log before trusting it.
ENVIRONMENT="${ENVIRONMENT:-opencodeinstruct}"
POOL="${POOL:-}"                                        # empty = broad sampling over the whole ~50k subset
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1280}"                # code completions: med ~330, max ~1190
SCREEN_MAX_TOKENS="${SCREEN_MAX_TOKENS:-1280}"
TARGET_K="${TARGET_K:-4}"
GPU_MEM="${GPU_MEM:-0.65}"; POOL_SIZE="${POOL_SIZE:-48}"; GEN_BATCH="${GEN_BATCH:-24}"
OCI_PROMPT_ONLY=1
CURATE="${CURATE:-0}"                                   # see note above
# FRONTIER OFF: the online frontier learns OMI-correctness labels (>=8 term &
# 2..6 correct of 8). opencode reward is authoritative, so the miner's local
# n_correct is ALWAYS 0 -> the frontier would learn "every prompt is bad" and
# degenerate. Uniform sampling over the broad subset is correct for blind-submit.
FRONTIER="${FRONTIER:-0}"
# SCREEN OFF: the two-stage screen filters on LOCAL correctness, which is
# unknowable for opencode -> it can only pre-filter non-terminators (which the
# deep <8-completion drop already handles). Skipping it saves SCREEN_OVERSAMPLE
# gens/prompt -> higher group throughput.
TWO_STAGE="${TWO_STAGE:-0}"
# OVERSAMPLE LOW: blind-submit ships the first 8 DISTINCT terminating samples;
# we don't need a deep oversample to manufacture a k-mix (that's curation, N/A
# here). 16 leaves margin for safety-gate drops while maximizing prompts/sec.
OVERSAMPLE="${OVERSAMPLE:-16}"
# GRADER-CURATE (opt-in: GRADER_CURATE=1) — curate the SUBMISSION, not the pool.
# Grade our OWN oversample via the real grader server (exact validator parity) and
# select a scattered k-correct/(8-k)-wrong 8 => in-zone BY CONSTRUCTION (the curated
# POOL failed because pre-screening can't predict the miner's safety-filtered shipped
# 8; this grades the actual shipped completions). Needs a RUNNING grader server
# (opencode/run_grader_curate.sh starts it) + reconstructed cases cache. Overrides:
# CURATE on, deeper oversample (need BOTH correct+wrong buckets), pool = cached idxs.
if [ "${GRADER_CURATE:-0}" = "1" ]; then
  CURATE=1
  OVERSAMPLE=24
  POOL="${POOL:-$SN81/data/inzone_pool_opencode.json}"
  OCI_GRADER_CURATE=1
  OCI_CASES_PATH="${OCI_CASES_PATH:-$SN81/data/oci_cases_cache.json}"
fi
# CONSENSUS (opt-in: CONSENSUS=1) — case-INDEPENDENT in-zone curation. NO
# grader server, NO reconstructed cases. The validator RECOMPUTES every
# reward and gates on std>=0.43, and BOTH reward extremes are
# case-independent: a completion the sandbox can't run (forbidden top-level
# import like numpy/random/json/sys, syntax error, no entry function) scores
# EXACTLY 0.0 on every hidden case; a genuinely-correct one (confirmed vs the
# prompt's public Sample I/O + output-consensus) scores ~1.0. Shipping 4
# winners + 4 guaranteed-0 losers => recomputed std==0.5 => in-zone BY
# CONSTRUCTION. consensus.py spawns its own pool of the DEPLOYED (HEAD)
# grader worker for exact parity. See opencode/consensus.py.
if [ "${CONSENSUS:-0}" = "1" ]; then
  CURATE=1
  TARGET_K="${TARGET_K:-4}"          # 4 winners + 4 losers => std 0.5 (max margin)
  OVERSAMPLE="${CONSENSUS_OVERSAMPLE:-32}"  # OVERRIDE the opencode :-16 default (set above) — need >=4 winners AND >=4 tier-A losers
  TWO_STAGE="${TWO_STAGE:-0}"        # single deep pass: consensus classifies in build_groups
  OCI_CONSENSUS=1
  # Curatable prompts that already WON are in cooldown ~forever (1M windows),
  # so re-mining proven-curatable (hot pool) just re-fires dead prompts ->
  # prompt_in_cooldown. Mine FRESH instead, and BURN cooldown rejects so the
  # pool stops re-firing them.
  HOT_FRAC="${HOT_FRAC:-0}"
  BURN_COOLDOWN=1
  # EXACT hidden cases (reconstructed from nvidia unit_tests by id). When set,
  # consensus GRADES each completion to its exact validator reward (reliable
  # winners pass all cases / losers fail all) instead of guessing from samples.
  CONSENSUS_CASES="${CONSENSUS_CASES:-$SN81/data/oci_cases_v2.json}"
  # Restrict mining to the cached (exact-graded) prompts so every group is
  # graded against real cases. Proactive cooldown-exclusion drops won ones.
  [ -s "$SN81/data/cases_pool.json" ] && POOL="${POOL:-$SN81/data/cases_pool.json}"
  # STRICT (CONSENSUS_STRICT=1): only fire reliable consensus 4+4, no broad
  # fallback — isolates consensus accept rate / pure high-precision strategy.
  [ "${CONSENSUS_STRICT:-0}" = "1" ] && OCI_CONSENSUS_STRICT=1
fi
# DECOOL-SNIPE DEAD HERE (measured 2026-06-08): BATCH_PROMPT_COOLDOWN_WINDOWS=1_000_000
# (constants.py:283) => won prompts cool for ~19yr and NEVER exit. The cooldown list
# is a permanent "already-won" blocklist that only GROWS (measured: +15 entries/window,
# 0 exits), not a recycling pool. So there are no decool targets; _decool_sampler just
# falls back to broad. In-zone prompts are a PERMANENTLY-DEPLETING resource; the only
# real lift is PRE-SCREENING fresh prompts (opencode/build_opencode_pool.py). Leave OFF.
DECOOL_SNIPE="${DECOOL_SNIPE:-0}"
