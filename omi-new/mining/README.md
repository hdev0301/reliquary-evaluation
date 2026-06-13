# Reliquary optimized mining toolkit

A self-contained, organized mining subsystem that sits **next to** the
reference `reliquary.miner` engine (it does not modify it) and reuses
`reliquary.protocol` / `reliquary.constants` so the wire format stays
byte-for-byte compatible with the live validator.

```
mining/
├── common/        env-agnostic engine: vLLM gen, GRAIL pregen, pool, fire, state
├── opencode/      OpenCodeInstruct miner  (this is the env that ships today)
├── openmath/      OpenMathInstruct miner  (sibling, added later — see below)
├── scripts/       offline prep (build the local frontier oracle)
└── tests/         fast, dependency-light correctness tests
```

One directory per environment, exactly as requested: `opencode/` is fully
self-contained, and `openmath/` will be its sibling reusing everything in
`common/`. Nothing in `opencode/` is OpenMath-specific and nothing in `common/`
is OpenCode-specific.

---

## Why the reference miner leaves money on the table

Reliquary is a **prediction market on prompt selection**, not a throughput race
(see `docs/concepts.md`). A window seals when each env collects `B_BATCH = 8`
distinct valid prompts; each filled slot pays `pool/8`, split `K_p` ways among
the miners who hit that prompt; submissions are ordered by **drand round**
(3 s buckets), and once 8 distinct prompts land the rest reject `BATCH_FILLED`.

So earning well means three things, and the reference engine does none of them
optimally:

| Problem | Reference engine | This toolkit |
|---|---|---|
| **The model is well-trained** → most prompts score 0/8 or 8/8 (σ≈0 → `OUT_OF_ZONE`). Finding σ≥0.43 prompts is the whole game. | uniform-random pick, blind | **frontier prediction**: screen prompts with a local σ proxy *before* spending a slot |
| **Fire-time latency** = ~60–100 s of gen + GRAIL after the window opens → you miss the early drand round and the batch fills without you | serial pick→gen→prove→submit *inside* the window | **pregeneration**: do all the heavy, window-independent work ahead of time; fire in **milliseconds** |
| **Slow generation** | `transformers.generate` | **vLLM** paged-attention batched sampling |

---

## The three wins

### 1. vLLM generation (`common/vllm_generator.py`)
Batched, continuous-batching sampling at the protocol-fixed params
(`T_PROTO=0.9`, `top_p=1.0`, top-k off, cap `8192`, stop at EOS). vLLM is used
**only for sampling** — the GRAIL proof is still built from our own HF forward
pass, so submitted log-probs/sketches come from the same kernel the validator
recomputes with (keeps us inside `LOGPROB_IS_EPS=0.10` and the distribution /
token-authenticity guards). Prompt tokens come from the shared `encode_prompt`
(chat template, `enable_thinking=False`) and are fed to vLLM as `prompt_token_ids`
so we never trip `PROMPT_MISMATCH`.

### 2. Pregeneration (`common/grail_proof.py`, `common/pregen.py`, `common/fire.py`)
The validator's GRAIL sketch is `buckets · r_vec`, where `buckets` are the
log-magnitude buckets of the proof model's hidden state (**depend only on
model+tokens**) and `r_vec` is derived from the window's `randomness` (**the only
window-dependent input**). So:

* **During idle/training/publishing windows** we sample candidates with vLLM,
  screen them, run the HF forward pass, and cache the tiny int8 `buckets`
  matrix + fp32 token log-probs — keyed to the current `checkpoint_n`.
* **The instant a window opens** we only do `buckets @ r_vec`, sign, build the
  Merkle root, stamp the drand round, and POST.

`mining/tests/test_pregen_sketch.py` proves the split is **bit-identical** to
calling `GRAILVerifier.create_commitments_batch` on the spot, so a pregen proof
is indistinguishable on the wire. The pool is flushed the moment `checkpoint_n`
advances (stale model would fail GRAIL) — safe because checkpoints publish only
every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10` trained windows, so one pool
serves many windows.

### 3. Frontier prediction for OpenCode (`opencode/oracle.py`, `opencode/frontier.py`)
OpenCode is validator-authoritative — miners get prompts only. But the
validator's hidden cases are *derived from the public* `nvidia/OpenCodeInstruct`
`unit_tests` by exactly `scripts/build_opencodeinstruct_subset.py`. So we
reconstruct the same structured cases offline, key them by `sha256(prompt)`, and
grade our 8 vLLM completions with the **validator's own grader** (`GraderClient`
→ identical sandbox) to predict the reward vector and its σ. We keep only groups
the proxy scores in the high-σ core (`k∈[3,5]/8` by default) with a σ margin.

This is **frontier prediction, not reward gaming**: OpenCode submissions carry
the `0.0` placeholder reward (the validator recomputes the real one), so this
can never cause `REWARD_MISMATCH` — it only decides *which prompts deserve a
slot*. It is the "cheap proxy to predict frontier likelihood" `docs/mining.md`
explicitly endorses. If a future task makes cases private/generated, swap the
oracle for the self-consistency proxy; the rest of the pipeline is unchanged.

---

## Setup (uv, two venvs)

All package management goes through **uv**. The catch: the GRAIL proof is
recomputed by the validator, so the proof stack must match the validator's
Dockerfile **exactly** — but a vLLM new enough to accept `transformers==5.9.0`
needs torch 2.11, while the validator runs **torch 2.7.0**. They can't share one
venv. They don't need to: the validator never runs vLLM, and only token ids
cross from generation into the proof (which the validator-matched HF model
recomputes). So `mining/setup.sh` builds **two venvs**:

| venv | role | stack |
|---|---|---|
| `.venv` | miner + **GRAIL proof** (consensus-critical) | torch **2.7.0+cu128**, `transformers==5.9.0`, **flash-attn 2.8.3**, **flash-linear-attention 0.5.0**, reliquary — *exact validator match* |
| `.venv-vllm` | generation worker only | vLLM (its own newer torch) — decoupled from consensus |

Validator-compat invariants (all enforced by `setup.sh`, none safe to relax on
mainnet): `transformers==5.9.0` exact (`encode_prompt` byte-identity →
`PROMPT_MISMATCH`); **flash_attention_2** (sketch kernel bit-sensitivity →
`GRAIL_FAIL`); **flash-linear-attention 0.5.0** (Qwen3.5 GatedDeltaNet kernel);
bf16 + **same GPU class as the validator (H200)**.

```bash
cd /root/reliquary

# 0. Build both venvs (validator-matched proof venv + vLLM venv). One command.
bash mining/setup.sh                # add --with-oracle to also build the oracle

# 1. Build the local frontier oracle ONCE (if not done in step 0). Reconstructs
#    ~50k case sets from the public unit tests (scans nvidia/OpenCodeInstruct).
.venv/bin/python -m mining.scripts.build_local_oracle \
    --out mining/state/opencode_oracle.json.gz
#    add --verify-determinism to match the validator's subset filter exactly (slower).

# 2. Configure and launch (run.sh launches the miner in .venv; it spawns the
#    vLLM worker in .venv-vllm itself).
cp mining/opencode/.env.example mining/opencode/.env   # edit wallet + validator url
source mining/opencode/.env
bash mining/opencode/run.sh
```

Hardware: single H200 is enough — the small HF proof model + vLLM's KV cache
share GPU 0 (the two processes hold independent CUDA contexts). 2× GPU: put the
proof on `RELIQUARY_PROOF_GPU` and vLLM on `RELIQUARY_GEN_GPU`. The local
prediction grader uses the validator's own grader code; install `runsc` for
sandboxing or set `RELIQUARY_ALLOW_UNSANDBOXED_GRADER=1` on an isolated box (the
pass/fail result is identical either way — `runsc` only adds OS isolation).

---

## Tunables (all via env; see `opencode/.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `RELIQUARY_POOL_TARGET_DEPTH` / `_MAX_DEPTH` | 24 / 48 | how many ready in-zone groups to keep queued |
| `RELIQUARY_SCREEN_BATCH_PROMPTS` | 8 | candidate prompts screened per vLLM batch |
| `RELIQUARY_TARGET_K_LO` / `_K_HI` | 3 / 5 | keep groups the proxy scores `k∈[lo,hi]/8` |
| `RELIQUARY_SIGMA_MARGIN` | 0.0 | headroom over `SIGMA_MIN=0.43`; auto-nudges up on `out_of_zone` verdicts |
| `RELIQUARY_DRAND_BOUNDARY_MARGIN_S` | 0.4 | sleep past a drand boundary before firing (avoids `STALE/FUTURE_ROUND`) |
| `RELIQUARY_MAX_FIRE_PER_WINDOW` | 8 | distinct prompts per window (never > the per-hotkey cap of 8) |

---

## Protocol-compliance guardrails (built in)

* exactly `M_ROLLOUTS = 8` rollouts per group, prompt tokens via `encode_prompt`;
* each completion truncated at (and including) the first EOS; groups with > 1
  cap-truncated rollout are dropped (`MAX_TRUNCATED_PER_SUBMISSION = 1`);
* the drand round is read **immediately before each POST**, boundary-safe;
* the envelope signature binds `(hotkey, window, prompt_idx, merkle_root,
  checkpoint_hash, drand_round, randomness, nonce)` exactly as the validator
  expects;
* `checkpoint_hash` always reflects the current published revision; the pool
  refuses to fire stale-checkpoint groups;
* targeting the high-σ core keeps zeros below the reward-shape share threshold,
  and a defensive check skips groups that look like manufactured losers.

## Tests

```bash
python -m mining.tests.test_pregen_sketch     # pregen ≡ validator (no GPU needed)
```

## Adding OpenMath later

Create `mining/openmath/` mirroring `opencode/`:
* `oracle.py` → not needed; OpenMath labels are public, so reward is computed
  directly with `env.compute_reward` (no grader, no reconstruction);
* `frontier.py` → screen with the real local reward vector (binary `{0,1}`) → σ;
* `miner.py` → identical wiring, `ENV_NAME = "openmathinstruct"`, and submit the
  verified local reward instead of the `0.0` placeholder (pass `reward_for=` to
  `fire_group`).

`common/` (vLLM, pregen, pool, fire, state) is reused unchanged. To mine both at
once, run two miners or extend the pool with both env names (the pool is already
keyed by env name).
