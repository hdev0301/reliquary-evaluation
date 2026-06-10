# OpenCode workspace

Self-contained home for the **opencode** mining strategy (nvidia/OpenCodeInstruct env),
kept separate from the **openmath** modes (symbolic/numeric) that use the shared
`../data/` + `../bin/run_miner.sh`. Nothing here writes into the shared `data/`.

## Why opencode is different
The opencode reward is **validator-authoritative** = passed/total over HIDDEN structured
tests. The miner runs `RELIQUARY_OCI_PROMPT_ONLY=1` and can't grade locally, so it mines
**HONEST** (`CURATE=0`). To still land in-zone (k correct + (8−k) wrong, σ≥0.43) we mine a
**scatter-screened pool**: prompts the *current validator checkpoint* solves intermediately
(not 0/M, not M/M), so the 8 rollouts naturally vary.

## Layout
```
opencode/
  build_pool.sh     # build the scatter pool (auto-resolves validator checkpoint; outputs here)
  run_miner.sh      # launch the opencode miner (wraps ../bin/run_miner.sh, MODE=opencode)
  data/             # inzone_pool_opencode.json, oci_cases_cache.json, oci_gen_cache_seed*.json,
                    # hot_pool.json, submitted_idx.json   (all opencode-only)
  diagnostics/      # opencode_pool_meta.json (per-prompt scatter stats)
  logs/             # build_opencode.log
```
The heavy builder lib stays at `../dataprep/build_opencode_pool.py` (shared infra); it honors
`RELIQUARY_DATA_DIR`/`RELIQUARY_DIAG_DIR`, which `build_pool.sh` points here.

## Build the pool
```bash
bash /root/sn81-miner/opencode/build_pool.sh                 # 1000 candidates, m=16, seed 7
MAXC=3000 bash /root/sn81-miner/opencode/build_pool.sh       # scale up (reuses case cache)
REGRADE=opencode/data/oci_gen_cache_seed7.json bash .../build_pool.sh   # re-grade, no GPU
```
The pool is **checkpoint-specific** — rebuild when the validator advances `ckpt_n`
(`build_pool.sh` re-queries `/state` each run).

## Mine
```bash
bash /root/sn81-miner/opencode/run_miner.sh                  # uses the pool once built, else broad
PREDICT_BLIND=0 bash /root/sn81-miner/opencode/run_miner.sh  # disable predictive seal fire
```
