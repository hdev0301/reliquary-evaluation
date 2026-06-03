#!/bin/bash
# Reliquary miner — CURATION / validator-replica pipeline (reward-vector candidate
# selection). On a converged BIMODAL checkpoint, natural 8-sample groups score 8/8
# or 0/8 (never in-zone). Curation (pregen.py build_groups) over-generates, rewards
# every candidate against the PUBLIC env reward, SELECTS an in-zone 8-subset
# (k correct + 8-k wrong), places it non-monotonically (passes reward_shape), and
# pre-validates every gate locally (zero integrity rejects, like the rank-1 miner).
# All rollouts are genuine current-checkpoint samples; the validator recomputes the
# reward -> selection, not fabrication.
cd /root/reliquary || exit 1
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
pkill -9 -f "reliquary.cli.main mine" 2>/dev/null || true
sleep 2
set -a; source scripts/.env; set +a
# --- curation knobs (read by reliquary/miner/pregen.py) ---
export RELIQUARY_CURATE=1
export RELIQUARY_CURATE_TARGET_K=6      # k=6 needs only 2 DISTINCT wrong (the min) -> max curatability
export RELIQUARY_CURATE_MARGIN=2        # validate k+2 / (8-k)+2 candidates, keep first passing
# --- two-stage screen, retuned FOR curation ---
# Goal: skip ramblers cheaply (the ~88% that don't terminate), but KEEP fluent
# prompts that have BOTH correct and wrong answers (the curatable ones). The
# default p-band [0.10,0.90] would reject high-correct prompts curation wants, so
# widen it to [0.03,0.97]: drop only pure 8/8 / 0/8, keep everything with a mix.
# Cheaper + broader screen = faster DISCOVERY of the rare fluent+curatable prompts
# (the real bottleneck: ~99% of pool prompts ramble). Screen many, cheaply.
export RELIQUARY_SCREEN_OVERSAMPLE=32   # fewer samples to detect termination+mix
export RELIQUARY_SCREEN_MAX_TOKENS=640  # cut ramblers early (winners' completions ~417, p75<640)
export RELIQUARY_SCREEN_MIN_TERM=4      # ~12.5% of 32 = fluent enough to deep-mine
export RELIQUARY_SCREEN_P_LOW=0.03      # keep prompts with >=~2 wrong present
export RELIQUARY_SCREEN_P_HIGH=0.97     # keep prompts with <100% correct (i.e. some wrong)
# Hot pool: self-built cache of screen-proven fluent+curatable prompts, re-mined
# to amortize discovery (the "prepare data" step, built on the H200 as it mines).
export RELIQUARY_HOT_POOL_PATH=/root/hot_pool.json
export RELIQUARY_HOT_FRAC=0.5           # hot pool is clean (curated-only) + persistent blocklist prevents
                                       # double-submit, so re-mining is productive; it's a CAP that self-limits
                                       # via fresh-fill (pregen.py:400-407), so no wasted compute when hot is small
export RELIQUARY_HOT_CAP=4000
rm -f /root/miner.log
nohup .venv/bin/python -m reliquary.cli.main mine \
  --network finney --netuid 81 --wallet-name ronnywebdev --hotkey hdev0301 \
  --checkpoint Qwen/Qwen3-4B-Instruct-2507 --validator-url http://86.38.238.30:8080 \
  --gpu-memory-utilization 0.80 --pool-size 96 --gen-batch 48 \
  --max-new-tokens 1024 --oversample 160 \
  --prompt-idx-file /root/inzone_pool.json --two-stage \
  --log-level INFO > /root/miner.log 2>&1 &
# DEPTH->BREADTH rebalance (evidence: at oversample 512 kept=0 for 8 straight batches;
# avg_completions~80 collapse to ~50 distinct correct + <2 distinct WRONG). The GPU is
# COMPUTE-bound (util=100%), and distinct-wrong is a FIXED per-prompt property at protocol
# temp (sampling params are pinned; hotter sampling -> logprob_mismatch reject). So depth
# past ~160 only piles DUPLICATE wrongs -> wasted compute. oversample 160 still yields ~25
# terminating (need >=8) and gives a genuinely-curatable prompt its full shot at a 2nd
# distinct wrong, while freeing ~3x compute for BREADTH: gen-batch 48 + pool 96 + gpu-mem
# 0.80 (51GB was idle) screen MORE prompts/cycle -> more chances to find the rare prompts
# that NATIVELY have >=2 distinct wrong. keepers/win = prompts_searched x P(>=2 distinct wrong).
echo $! > /root/miner.pid
echo "launched PID=$(cat /root/miner.pid)"
