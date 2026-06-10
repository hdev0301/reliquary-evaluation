# OpenCode Mining Runbook (SN81 / Reliquary)

End-to-end pipeline to mine the **opencode** environment (`nvidia/OpenCodeInstruct`) and land
real `verdict ACCEPTED` submissions. Three phases: two are data prep (one GPU, one CPU), then mining.

All paths are under `/root/sn81-miner/opencode/` unless noted. Commands assume the repo venv at
`/root/reliquary/.venv`.

---

## Why opencode needs data prep (the key fact)

The opencode reward is **validator-authoritative** = passed/total over HIDDEN tests. The miner's
default `RELIQUARY_OCI_PROMPT_ONLY=1` loads a prompt mirror with **no test cases**, so
`compute_reward()` is always `0` → every rollout looks "all-wrong" → **0 in-zone groups → 0 submits**.

The fix is to give the miner a **local reward signal**:
1. **reconstruct** the hidden test cases from the PUBLIC `nvidia/OpenCodeInstruct`,
2. attach them onto the prompt mirror as a **local structured subset** (index-aligned),
3. run a **local grader server** (real validator `worker.py`, exact parity) so `compute_reward()`
   actually scores code.

Then the miner produces in-zone groups (σ ≥ 0.43) and submits.

---

## Phase 0 — Prerequisites (one-time, fresh box)

```bash
bash /root/sn81-miner/bin/setup.sh
```
- Installs the miner env + cu13 toolchain + GDN kernel.
- Needs **public** HF access only (the gated structured-subset is NOT used; we reconstruct cases
  from the public nvidia dataset). The checkpoint auto-resolves from the validator.

---

## Phase 1 — DATA PREP

### Step 1: reconstruct test cases + build the scatter pool  *(GPU + network, ~25 min / 1000 cand)*

```bash
bash /root/sn81-miner/opencode/build_pool.sh                  # defaults: MAXC=1000 M=16 SEED=7
# scale up for a bigger / fresher pool (raises accept rate, fewer cooldown/dup collisions):
MAXC=5000 bash /root/sn81-miner/opencode/build_pool.sh
# re-grade persisted completions WITHOUT re-running the GPU:
REGRADE=/root/sn81-miner/opencode/data/oci_gen_cache_seed7.json bash /root/sn81-miner/opencode/build_pool.sh
```

Streams the public `nvidia/OpenCodeInstruct`, reconstructs each prompt's cases byte-exact, generates
rollouts on the **validator's current checkpoint**, grades them (crash-proof sandbox), selects
scattering prompts.

**Produces** (under `opencode/`):
| file | role |
|---|---|
| `data/oci_cases_cache.json` | reconstructed test cases — **required by Step 2** |
| `data/inzone_pool_opencode.json` | the scatter pool (mining prompt-idx list) |
| `data/oci_gen_cache_seed*.json` | persisted completions (for `REGRADE=`) |
| `diagnostics/opencode_pool_meta.json` | per-prompt scatter stats |

> The case cache makes re-runs cheap (no re-streaming). For exact-parity grading during the build,
> add `--use-grader` (requires the grader from Phase 2 running).

### Step 2: build the local gradeable subset  *(CPU, ~30s)*

```bash
cd /root/reliquary && .venv/bin/python /root/sn81-miner/opencode/build_local_subset.py
```

Attaches the reconstructed cases onto the full 50k prompt mirror, **index-aligned** with the
validator → `opencode/data/oci_local_subset/`. Rows without a reconstructed case score 0 (just never
in-zone). This is what gives the miner its reward signal.

---

## Phase 2 — MINE  *(GPU)*

```bash
bash /root/sn81-miner/opencode/run_miner.sh
```

One command does everything:
- starts the local grader (`grader.sh start`) if not already up,
- sets `OCI_PROMPT_ONLY=0` + points the env at `oci_local_subset` (real reward),
- wires the scatter pool, honest `curate=0`, predictive seal-fire,
- kills any prior miner, frees the GPU, launches → logs to `/root/sn81-miner/logs/miner.log`.

Useful overrides:
```bash
POOL="" bash opencode/run_miner.sh            # broad sampling (only the ~cases-having prompts score)
PREDICT_BLIND=0 bash opencode/run_miner.sh    # disable predictive seal fire
OCI_PROMPT_ONLY=1 bash opencode/run_miner.sh  # revert to the (non-producing) prompt-only mode
```

---

## Preparing data WHILE the miner is running

The constraint is the **GPU** (a second vLLM would OOM the H200). Split the work:

**Safe to run alongside a live miner (CPU/network only — no GPU):**
```bash
# 1. reconstruct MORE cases (new SEED = new prompts). CASES_ONLY stops before the GPU step.
CASES_ONLY=1 MAXC=5000 SEED=11 bash /root/sn81-miner/opencode/build_pool.sh
# 2. fold the bigger case cache into the local subset (CPU, ~30s)
cd /root/reliquary && .venv/bin/python /root/sn81-miner/opencode/build_local_subset.py
# 3. restart the miner to pick up the bigger subset (brief GPU reload)
bash /root/sn81-miner/opencode/run_miner.sh
```
The case cache **accrues** across runs, so repeated `CASES_ONLY` runs with different `SEED`s keep
growing the gradeable universe (more diverse prompts → fewer `cooldown`/`hash_duplicate` rejects).

**Must wait for a GPU window (stop the miner first):**
```bash
# pool SCREENING loads vLLM -> needs the GPU. Stop miner, screen, restart.
kill -9 $(cat /root/sn81-miner/miner.pid); for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
MAXC=5000 bash /root/sn81-miner/opencode/build_pool.sh     # cases + pool (GPU)
bash /root/sn81-miner/opencode/run_miner.sh                # back to mining
```
> Note: a fresh **pool** is optional — the miner's runtime σ-gate keeps submissions correct and its
> HOT_POOL self-discovers good prompts. The high-value while-mining prep is **more cases + subset**.

---

## Dependency chain

```
build_pool.sh ─► oci_cases_cache.json ─► build_local_subset.py ─► oci_local_subset ┐
       └────────► inzone_pool_opencode.json ───────────────────────────────────────┤
                                          grader.sh (auto) ─► local grader ──────────┴─► run_miner.sh
```

---

## Monitor / manage

```bash
bash /root/sn81-miner/opencode/grader.sh status            # grader up? (reward is 0 without it)
tail -f /root/sn81-miner/logs/miner.log                    # live mining
grep -c "verdict ACCEPTED" /root/sn81-miner/logs/miner.log # accept count
grep -E "pregen: \+[0-9]+/[0-9]+ in-zone" /root/sn81-miner/logs/miner.log | tail   # in-zone yield
grep -E "verdict (ACCEPTED|REJECTED)" /root/sn81-miner/logs/miner.log | tail       # real verdicts

# stop everything
kill -9 $(cat /root/sn81-miner/miner.pid); for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done
bash /root/sn81-miner/opencode/grader.sh stop
```

Health checklist: `grader.sh status` = running, log shows `EXPLICIT prompt-idx pool` + `vLLM generator ready`,
and `screen: N/24 promising` with **N > 0** (N=0 means no reward signal → grader/subset misconfigured).

---

## Maintenance (important)

The cases AND the pool are **checkpoint-specific** — the model must scatter on the SAME weights the
validator scores with. When the validator advances `ckpt_n` (`curl -s http://86.38.238.30:8080/state`),
refresh: **Phase 1 Step 1 → Step 2 → restart Phase 2**. `build_pool.sh` re-queries the live checkpoint
automatically each run.

---

## Troubleshooting

| Symptom (in `miner.log`) | Cause | Fix |
|---|---|---|
| `screen: 0/24 promising`, `allwrong=24` | no reward signal | grader not running, or `OCI_PROMPT_ONLY=1`, or subset missing → check `grader.sh status`, rebuild subset |
| `verdict REJECTED reason=prompt_in_cooldown` | prompt recently won | mine fresher prompts → bigger pool (higher `MAXC`) |
| `verdict REJECTED reason=hash_duplicate` | resubmitted identical group | bigger/more-diverse pool; burned blocklist at `data/submitted_idx.json` |
| `verdict REJECTED reason=batch_filled` | lost the seal race | keep `PREDICT_BLIND=1` (predictive fire); more in-zone groups ready per window |
| boot `OverflowError ... frontier.py` | infinite ground-truth | patched in `reliquary/miner/frontier.py` (re-apply if repo reinstalled) |
| boot `Column 'problem_source'` traceback | opencode env has no such column | patched in `reliquary/miner/pregen.py` (handled fallback) |

---

## Caveats

- The local grader executes model-generated code **UNSANDBOXED** (no gVisor) — acceptable for your
  own model output on your own box; do not point it at untrusted code.
- Two **reliquary-repo patches** are required (`miner/frontier.py`, `miner/pregen.py`) and would be
  reverted by a `git pull`/reinstall — re-apply them.
- `build_pool.sh`'s prompt **selection** still uses the pre-audit binary-`frac` screen; the miner's
  runtime σ-gate keeps submissions correct regardless, but applying the audit's σ-screen fix makes
  the *pool* higher-yield (fewer wasted windows).

---

## TL;DR

```bash
# fresh build:
bash   /root/sn81-miner/opencode/build_pool.sh                                  # cases + pool   (GPU)
.venv/bin/python /root/sn81-miner/opencode/build_local_subset.py               # local subset   (CPU)
bash   /root/sn81-miner/opencode/run_miner.sh                                   # mine           (GPU)

# already prepped on this box -> just:
bash   /root/sn81-miner/opencode/run_miner.sh
```
