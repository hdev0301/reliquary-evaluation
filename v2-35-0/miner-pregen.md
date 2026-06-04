# Fast miner: vLLM generation + pregeneration

This is the implementation behind the in-place rewrite of `reliquary mine`. It
keeps **every validator invariant intact** — it only does the legitimate work
faster and earlier.

## Why it's faster (and still 100% in-rule)

The verifier's behavioural checks all run on (a) the submitted **token ids** and
(b) values the validator **recomputes with its own HF forward**. Nothing
inspects the sampler. So:

- **Generation can be vLLM.** Tokens are sampled by vLLM 0.20.x; the GRAIL
  proof + `token_logprobs` are still computed by the bit-identical HF forward
  (`flash_attention_2`, `LAYER_INDEX=-1`, raw/un-temperature-scaled logits) the
  validator re-runs. The verifier even guards against "vLLM→HF drift", so this
  is a supported architecture, not a trick.

- **The proof is randomness-free until the last step.** A GRAIL commitment is
  `sketch[pos] = (buckets[pos] · r_vec) mod PRIME_Q`. `buckets` depends only on
  `(weights, tokens)`; only `r_vec = generate_r_vec(state.randomness)` depends on
  the per-window drand seed. So the whole expensive forward can run **per
  checkpoint, ahead of time**, and window-open is a microsecond integer
  projection + sign + POST. `grail_cache.py` splits `create_commitments_batch`
  at the bucketing line; `tests/unit/test_grail_cache.py` proves the cached
  replay is bit-identical.

## Components

| Module | Role |
|---|---|
| `miner/vllm_backend.py` | vLLM 0.20.x batched group sampling (gen only). EOS-normalised; cap-truncations dropped. |
| `miner/grail_cache.py` | `compute_buckets` (cache, randomness-free) + `project_buckets` (window-open, bit-identical to `create_commitments_batch`). |
| `miner/pregen.py` | Owns the models on a daemon thread. Generates → HF forward → caches buckets+logprobs → **zone-filters** (σ≥0.43, exact `verifier.is_in_zone`) → **safety-screens** every behavioural gate with margin. Keyed by checkpoint revision. |
| `miner/engine.py` | Async control loop: poll `/state`, drive checkpoint reloads, **burst-fire** up to 8 distinct non-cooldown prompts in the first drand round, verdict poller (`/verdicts/{hotkey}`). |

### Pregeneration safety filters (computed free from the HF forward)

Every prepared rollout is pre-screened with margin so the live path has ~zero
behavioural rejections:

| Validator gate | Floor | Pregen margin |
|---|---|---|
| `OUT_OF_ZONE` | σ ≥ 0.43 (⇔ 2–6/8 correct) | exact `is_in_zone` |
| `BAD_TERMINATION` | `p_stop ≥ 0.01`, last tok ∈ EOS set | `p_stop ≥ 0.02` |
| `TOKEN_TAMPERED` | every tok HF-prob ≥ `1e-10` | ≥ `1e-8` |
| `BOXED_ANSWER_TAMPERED` | boxed toks ≥ `0.001` (unless argmax<0.99) | ≥ `0.005` |
| `DISTRIBUTION_SUSPICIOUS` | median>0.30, q10>0.025 | median>0.35, q10>0.05 |
| `LOGPROB_MISMATCH` | median `exp(|Δ|)−1 ≤ 0.10` | HF logprobs submitted; completion ≥ `CHALLENGE_K=32` |

We require all 8 of a prompt's samples to be genuine EOS-terminated rollouts —
**no cherry-picking** among extra generations (prompt selection is encouraged;
reward-vector shaping is not).

## Frontier prediction (prompt selection)

The pregenerator takes a `candidate_sampler(n, exclude) -> list[int]`. Default is
uniform-random over the env (minus cooldown / already-prepared). Plug your
`winners.jsonl`/`controls.jsonl` σ-predictor in here to raise the in-zone hit
rate (fewer wasted forwards, deeper pool). This is the lever `docs/mining.md`
calls "predict which prompts pass the zone filter".

## Deploy (single H200)

> The repo on disk has no working venv, and an orphan `VLLM::EngineCore`
> (~85 GB) + the stock miner (~8 GB) currently hold the GPU. Free them first.

```bash
# 1. Stop the stock miner + orphan vLLM to free the H200
kill "$(cat /root/miner.pid)" 2>/dev/null   # stock miner (PID 108537)
# kill the orphan VLLM::EngineCore too (find it via nvidia-smi)

# 2. Rebuild the environment — PROVEN recipe (verified 2026-06-02 on the H200 box).
#    A single `pip install -e .` FAILS: pip backtracks bittensor -> 9.0.0 (the floor),
#    which pins fastapi~=0.110 and conflicts with vLLM's fastapi>=0.115. Install staged,
#    bittensor BEFORE vllm, so the bittensor setuptools~=70 vs vLLM >=80 clash is a
#    harmless post-hoc warning (setuptools ends at 80; both run fine).
cd /root/reliquary
python3.12 -m venv .venv
.venv/bin/pip install -U pip wheel
.venv/bin/pip install 'bittensor==10.2.1'     # fastapi 0.136, setuptools 70
.venv/bin/pip install 'vllm==0.20.2'          # -> torch 2.11.0+cu130, transformers 5.9, setuptools 80
.venv/bin/pip install -e . --no-deps          # the reliquary package only
.venv/bin/pip install datasets aiobotocore boto3 botocore typer rich tenacity pyarrow safetensors bitsandbytes httpx requests
# flash-attn for flash_attention_2 (GRAIL must match the validator). There is NO
# torch2.11 wheel and no nvcc here for a source build — but the cu13/torch2.10 wheel
# imports and runs correctly on torch 2.11:
.venv/bin/pip install --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
# Verified working set: torch 2.11.0+cu130 · vllm 0.20.2 · transformers 5.9.0 ·
# bittensor 10.2.1 · flash-attn 2.8.1 (flash_attention_2 routes correctly).

# 3. Run (single GPU: vLLM gmu=0.55 leaves room for the HF proof model)
reliquary mine \
  --network finney --netuid 81 \
  --wallet-name ronnywebdev --hotkey hdev0301 \
  --checkpoint Qwen/Qwen3-4B-Instruct-2507 \
  --validator-url http://86.38.238.30:8080 \
  --gpu-memory-utilization 0.55 \
  --pool-size 64 --gen-batch 16 --max-new-tokens 2048 \
  --log-level INFO
```

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--gpu-memory-utilization` | 0.55 | vLLM's share. Single GPU: keep ≤0.7 so the HF proof model + fp32-logit forward fit in the rest. 2 GPUs: vLLM gets GPU0 at 0.9, HF proof on GPU1. |
| `--pool-size` | 64 | Target ready in-zone groups (~8 windows × 8 slots). Raise for deeper lookahead. |
| `--gen-batch` | 16 | Candidate prompts per vLLM pregeneration batch. |
| `--max-new-tokens` | 2048 | Per-rollout completion cap (protocol cap 8192). Lower = faster/cheaper; prompts whose solutions exceed it are dropped (cap-truncated). |

## Verification status

- `grail_cache` bit-identity: **passing on real torch** (fp32 + bf16, multiple shapes).
- Adversarial workflow (9 agents): all three load-bearing claims **HOLD, high confidence** —
  (1) generation/proof artifacts are randomness-free → safe to pregenerate per checkpoint;
  (2) vLLM-sampled + HF-proved tokens pass every check identically to HF-sampled;
  (3) cached-bucket projection reproduces `create_commitments_batch` bit-for-bit.
- Exact-match details confirmed against validator source: prompt tokenization
  (`add_special_tokens=False`, no chat template), zone σ (population std, `is_in_zone`),
  raw-logit logprobs, EOS set resolution, drand_round zero-tolerance.
