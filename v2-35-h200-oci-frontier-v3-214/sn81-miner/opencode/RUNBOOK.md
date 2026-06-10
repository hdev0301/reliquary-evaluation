# OpenCode Mining Runbook (SN81 / Reliquary)

End-to-end pipeline to mine the **opencode** environment (`nvidia/OpenCodeInstruct`) and land
real `verdict ACCEPTED` submissions.

> **The current flow is _broad-frontier + canon + burn_ — CPU-only, no GPU-per-checkpoint treadmill.**
> See [Current flow](#current-flow) below. The legacy GPU scatter-screen (Phase 1 Step 1 /
> `build_pool.sh`) is kept only as an optional cold-start warmer.

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

## Current flow

### Broad-frontier + canon + burn (no GPU per checkpoint)

The validator advances its checkpoint **~hourly**, and an empirically scatter-screened pool is
**checkpoint-specific** → it goes stale on every bump (a GPU-rebuild treadmill). We avoid that by
leaning on the miner's **frontier predictor**: an online logistic-regression that scores prompts by
*features* (keywords, answer-shape, stems — `extract_features`, **not** a memorized idx list) and
updates from every mined outcome, persisting to `/root/frontier_model.npz`. Feature-based prediction
**generalizes across checkpoint bumps**, so we hand the frontier the whole gradeable universe and let
it discover + adapt online — no GPU screen needed.

Three CPU/network steps + always-on mining:

```bash
# 1. grow the gradeable universe (reconstruct more cases; CPU/network, SAFE while mining)
ROUNDS=8 MAXC=4000 bash /root/sn81-miner/opencode/grow_data.sh

# 2. build the pool: divergence-refresh + canon-filter the gradeable universe (CPU, ~4s)
HOTKEY=hdev0301 KEEP_FRAC=0.4 bash /root/sn81-miner/opencode/build_frontier_pool.sh

# 3. mine: real-reward + grader + frontier over the canon pool
HOTKEY=hdev0301 bash /root/sn81-miner/opencode/run_miner.sh
#    (or fold 2+3 in one shot: HOTKEY=hdev0301 RESTART=1 bash .../build_frontier_pool.sh)
```

What each piece does:
- **`grow_data.sh`** — reconstructs new cases into `oci_cases_cache.json` (accrues across runs) and
  rebuilds `oci_local_subset`. The only thing that grows the gradeable universe.
- **`build_frontier_pool.sh`** — (a) pulls `/verdicts/<hotkey>`, recovers prompts the validator
  proof-rejected (`out_of_zone`/`reward_mismatch` = our reconstructed cases diverge from its pinned
  HIDDEN cases) → `divergent_idx.json`; (b) gradeable list = cache entries that have cases; (c) drops
  divergent, keeps the lowest-`sha256(prompt_idx)` `KEEP_FRAC` (these win the validator's canonical
  top-8 at seal → fewer `batch_filled`) → `inzone_pool_opencode.json`. **No GPU.**
- **`run_miner.sh`** — real-reward mode (local subset + grader) with the **frontier over the canon
  pool**; it discovers which prompts scatter on the *current* checkpoint, re-mines proven ones
  (`hot_pool`) and front-loads freshly-decooled ones (`decool-snipe=1`).

Composes: **canon-filter** (batch_filled) + **divergence-burn** (case parity) + **broad-gradeable**
(no treadmill) + **frontier online learning** (checkpoint adaptation).

> Trade-off: broad (~18k) has lower in-zone *density* than a fresh screen, so the frontier's ε-explore
> wastes some early screens — but the persistent model self-warms and **never goes stale**.

---

## Phase 0 — Prerequisites (one-time, fresh box)

```bash
bash /root/sn81-miner/bin/setup.sh
```
- Installs the miner env + cu13 toolchain + GDN kernel.
- Needs **public** HF access only (the gated structured-subset is NOT used; we reconstruct cases
  from the public nvidia dataset). The checkpoint auto-resolves from the validator.

---

## Phase 1 — DATA PREP *(LEGACY — GPU scatter-screen; superseded by [Current flow](#current-flow))*

> Kept as an optional **cold-start warmer** (seeds the case cache + a high-density starter pool). For
> steady-state, use `build_frontier_pool.sh` instead — it needs no GPU and doesn't go stale per checkpoint.

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
# CURRENT (broad-frontier):
grow_data.sh ──────────► oci_cases_cache.json ─► build_local_subset.py ─► oci_local_subset ┐
build_frontier_pool.sh ─► divergent_idx.json + inzone_pool_opencode.json (canon) ──────────┤
                          grader.sh (auto) ─► local grader ──────────────────────────────── ┤
                          /root/frontier_model.npz (persistent, online-learning) ───────────┴─► run_miner.sh

# LEGACY (GPU scatter-screen): build_pool.sh ─► oci_cases_cache.json + inzone_pool_opencode.json
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

**Broad-frontier flow (current):** the pool is **not** checkpoint-specific — the frontier adapts
online as the validator advances `ckpt_n` (`curl -s http://86.38.238.30:8080/checkpoint`). **No GPU
rebuild per bump.** The miner auto-resolves + downloads the validator's current checkpoint on each
(re)start. Just periodically re-run the CPU steps — to grow the universe and refresh divergent prompts:
```bash
ROUNDS=8 bash /root/sn81-miner/opencode/grow_data.sh                              # optional: grow universe
HOTKEY=hdev0301 RESTART=1 bash /root/sn81-miner/opencode/build_frontier_pool.sh   # refresh pool + apply
```

**Legacy scatter-screen flow:** a `build_pool.sh` pool IS checkpoint-specific (the model must scatter
on the same weights the validator scores with) — refresh **Phase 1 Step 1 → Step 2 → restart** on every
`ckpt_n` advance. This is the treadmill the broad-frontier flow exists to avoid.

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
# CURRENT flow — broad-frontier + canon + burn (no GPU treadmill):
bash /root/sn81-miner/opencode/grow_data.sh                                       # grow universe (CPU)
HOTKEY=hdev0301 RESTART=1 bash /root/sn81-miner/opencode/build_frontier_pool.sh   # canon+burn pool + mine

# already prepped on this box -> just mine:
HOTKEY=hdev0301 bash /root/sn81-miner/opencode/run_miner.sh

# LEGACY GPU scatter-screen (optional cold-start warmer only):
bash /root/sn81-miner/opencode/build_pool.sh && .venv/bin/python /root/sn81-miner/opencode/build_local_subset.py
```
