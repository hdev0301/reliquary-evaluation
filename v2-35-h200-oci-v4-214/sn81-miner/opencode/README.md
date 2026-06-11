# OpenCode (`opencodeinstruct`) mining — all tooling in one place

The SN81 OpenCode strategy. Reward is **continuous** (fraction of HIDDEN unit tests
passed), so the 8 rollouts naturally scatter in-zone — no converged-checkpoint
discovery wall like openmath. This dir holds everything OpenCode-specific; the
shared launcher (`../bin/run_miner.sh`) supplies the `opencode` MODE preset.

## Why a curated pool (the structural reason)
In-zone prompts (the model SCATTERS on them → σ≥0.43) are a **permanently-depleting
resource**: any prompt won by any miner cools for `BATCH_PROMPT_COOLDOWN_WINDOWS=1_000_000`
windows (≈forever). The network steadily consumes the ~12% in-zone prompts, so blind
broad mining decays toward all-`out_of_zone`. The only lift is to **pre-screen fresh
prompts** for scatter before submitting — that's this pool. The seal race is already
won from the US box (warm keep-alive + GROUPDIAG off the fire path), so in-zone rate is
the binding constraint.

## Files
- **`run.sh`** — thin wrapper: `MODE=opencode ../bin/run_miner.sh "$@"`. All
  run_miner.sh env knobs still apply.
- **`auto_pool.sh`** — supervisor that keeps the miner on a FRESH pool: rebuilds when
  the checkpoint drifts (≥`REFRESH_EVERY_CKPTS`) or the non-cooled pool falls below
  `MIN_FRESH`, then relaunches the miner. Single H200 ⇒ each refresh pauses mining
  ~30-40 min (reconstruction is cached, so refreshes are generation+grading only).
  Run under tmux/nohup: `bash opencode/auto_pool.sh`.
- **`build_opencode_pool.py`** — empirical in-zone pool builder. Reconstructs each
  prompt's hidden test cases (validator-replica, cached to `../data/oci_cases_cache.json`),
  generates M completions, grades locally, keeps prompts the model SCATTERS on.
  Use `--strict-zone` (band [0.25,0.75] = k∈[2,6]/8) and `--checkpoint <live snapshot>`.
  Output: `../data/inzone_pool_opencode.json`. NOTE: pass the multimodal kwarg
  (already wired) — Qwen3.5 needs `limit_mm_per_prompt={image:0,video:0}` or vLLM
  errors on the missing image processor.
- **`verify_opencode_gate.py`** — go/no-go gate (its capture file is currently corrupt;
  reconstruction is instead trusted because it uses the validator's own `process_row`
  and the prompt_idx→case_id mapping is proven by live accepts, validated empirically).

## Two ways to mine

### 1. Blind-submit (default, MINER-FILES-ONLY, no grader)
Generate 8, submit; the validator grades against hidden tests. Relies on natural
pass/fail variance for in-zone scatter. No local test reconstruction needed.
```
bash opencode/run.sh
```

### 2. Curated (higher in-zone yield, needs local grading)
Prove reconstruction, build a scatter-filtered pool, then mine it with curation on.
```
# a) prove we can reconstruct hidden cases offline
cd /root/reliquary && .venv/bin/python /root/sn81-miner/opencode/verify_opencode_gate.py

# b) build the in-zone pool (one-time; caches cases to ../data/oci_cases_cache.json)
cd /root/reliquary && .venv/bin/python /root/sn81-miner/opencode/build_opencode_pool.py \
    --max-candidates 3000 --m 16

# c) mine the curated pool with local grading on
POOL=$SN81/data/inzone_pool_opencode.json CURATE=1 bash /root/sn81-miner/opencode/run.sh
```

## Notes
- Both python scripts use absolute `sys.path` inserts (`/root/reliquary`,
  `/root/reliquary/scripts`) — they run from anywhere, no `cd` needed for imports,
  but `build_opencodeinstruct_subset` lives in `/root/reliquary/scripts/`.
- `build_opencode_pool.py` local grading exec's model code in a SIGALRM-timeout
  subprocess pool (not gVisor). Use `--use-grader` to route through a running
  grader server for sandboxed, exact-parity grading.
- Prompt mirror: `R0mAI/opencodeinstruct-prompts`; source: `nvidia/OpenCodeInstruct`.
