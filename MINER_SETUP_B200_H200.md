# Reliquary Miner Setup — B200 / H200

End-to-end bring-up for a single-GPU Blackwell (B200) or Hopper (H200)
box, running the reliquary miner with vLLM + flash-attention.

Tested on driver 580 / CUDA 13 / Python 3.12 / Ubuntu container.

The hardware differences between B200 and H200 are tuning, not
compatibility — the same install sequence works on both. Spots where
they differ are flagged inline.

---

## 0. Prerequisites

```bash
nvidia-smi          # driver >= 575; GPU reports as B200 (sm_100) or H200 (sm_90a)
python --version    # 3.12.x
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

`nvcc` is **not** required. Every install in this guide uses prebuilt
wheels.

---

## 1. Create + activate the venv

Use a **standalone uv venv at `/root/.venv`**, not the project-managed
`.venv` inside `/root/reliquary/`. The project's `pyproject.toml` /
lockfile pins old torch versions that fight this stack, and `uv run`
silently switches to that venv if you cd into the project.

```bash
uv venv /root/.venv --python 3.12
source /root/.venv/bin/activate
```

Re-source `/root/.venv/bin/activate` in every shell from now on.

---

## 2. Install the stack (order matters)

```bash
# 1. Torch — anchors the cuda runtime version for everything downstream.
uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# 2. vLLM nightly — needed for transformers 5.x compat and B200/H200 kernels.
uv pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly

# 3. transformers 5.x — what reliquary targets.
uv pip install "transformers>=5.8"

# 4. flash-attn — prebuilt wheel for torch 2.11+cu130 exists.
#    --no-build-isolation is critical: uses the already-installed torch
#    instead of fetching a stale one for the build sandbox.
uv pip install --no-build-isolation flash-attn

# 5. Reliquary's own dependencies.
cd /root/reliquary
uv pip install bittensor typer httpx huggingface_hub
```

If you need to install reliquary itself, prefer `--no-deps` so it
doesn't try to re-resolve torch:

```bash
uv pip install -e . --no-deps
```

---

## 3. Verify the stack

```bash
python -c "
import torch, vllm, transformers, flash_attn
print('torch        ', torch.__version__, torch.version.cuda)
print('vllm         ', vllm.__version__)
print('transformers ', transformers.__version__)
print('flash_attn   ', flash_attn.__version__)
from flash_attn import flash_attn_func
q=k=v=torch.randn(1,8,16,64,device='cuda',dtype=torch.bfloat16)
print('flash_attn fwd OK:', flash_attn_func(q,k,v).shape)
print('device cap:', torch.cuda.get_device_capability(0))
"
```

Expected:

| Package      | Version                 |
|--------------|-------------------------|
| torch        | `2.11.x+cu130` (cuda `13.0`) |
| vllm         | `0.20.x.devNNN` (nightly) |
| transformers | `5.8.x`                 |
| flash_attn   | `2.8.x`                 |

- `flash_attn fwd OK: torch.Size([1, 8, 16, 64])`
- `device cap: (10, 0)` for B200, `(9, 0)` for H200

If any of these are off, **fix it before moving on** — don't try to
launch the miner with a misaligned stack; you'll just chase ABI errors.

---

## 4. Drop the v4 overlays onto reliquary

Required:

```bash
cp main-v4.py         /root/reliquary/reliquary/cli/main.py
cp vllm_adapter-v4.py /root/reliquary/vllm_adapter.py
cp engine-v4.py       /root/reliquary/reliquary/miner/engine.py
```

Recommended (not hardware-specific; needed for competitive earnings):

```bash
cp submitter-v4.py    /root/reliquary/reliquary/miner/submitter.py   # v4.2 multi-validator broadcast
cp math-v4.py         /root/reliquary/reliquary/environment/math.py  # math env fixes
```

Without `submitter-v4`, the miner falls back to single-validator
submission and logs a warning at startup. You keep working, but lose
the ~N× weight multiplier from broadcasting.

---

## 5. Env vars required at launch

```bash
# vLLM probes deep_gemm for FP8 GEMM warmup. Qwen3-4B is bf16, so
# deep_gemm wouldn't be used anyway, and without the lib installed
# the probe crashes EngineCore. Disable it.
export VLLM_USE_DEEP_GEMM=0

# Keep vLLM out of Python logging (main-v4 also sets this; harmless to set explicitly).
export VLLM_CONFIGURE_LOGGING=0
export PYTHONUNBUFFERED=1

# B200: FLASHINFER attention backend is typically 10-20% faster than the default.
# H200: leave unset — the default is already optimal.
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Reliquary essentials
export HF_TOKEN=<your huggingface token>
# (wallet path defaults to ~/.bittensor/wallets — override if needed)
```

Persist these in your launch script, systemd unit, or shell init.

---

## 6. Launch the miner

```bash
cd /root/reliquary
python -m reliquary.cli.main mine \
  --use-vllm \
  --network finney \
  --netuid 81 \
  --wallet-name <wallet-name> \
  --hotkey <hotkey-name> \
  --checkpoint Qwen/Qwen3-4B-Instruct-2507 \
  --vllm-gpu 0 \
  --proof-gpu 0 \
  --gpu-memory-utilization 0.78 \
  --max-model-len 8192 \
  --kv-cache-dtype fp8 \
  --log-level INFO \
  --log-file /var/log/reliquary-miner.log
```

Hardware tuning notes:

| Flag                          | B200 (192 GB) | H200 (141 GB) |
|-------------------------------|---------------|---------------|
| `--gpu-memory-utilization`    | 0.85 (room to spare) | 0.78         |
| `VLLM_ATTENTION_BACKEND`      | `FLASHINFER`  | unset (default) |
| `--kv-cache-dtype`            | `fp8`         | `fp8`        |

---

## 7. Sanity-check the first run

Watch the log for these in order:

1. `[stage:start] vllm.LLM build ...` — vLLM construct begins
2. `heartbeat label='LLM(...) construct' tick=N elapsed=Xs gpu_mem=...`
   — every 2 s for first 20 s, then every 5 s
3. `Using TRTLLM prefill attention (auto-detected).` (or
   `FLASHINFER` if you overrode)
4. `vLLM build done: ready on cuda:0` (~60-120 s total)
5. `[stage:done] vllm.LLM build ...`
6. HF proof model load (~10-15 s)
7. `Miner ready (backend=vllm). Entering main loop.`

If you don't see any heartbeats for 20+ seconds during the construct,
your stderr is being buffered (not unbuffered). Confirm `PYTHONUNBUFFERED=1`
and re-run.

---

## 8. Known traps (in order of how often we hit them)

### `uv run` silently switches venvs

`uv run` inside `/root/reliquary/` uses the project's `.venv` (which
has old torch pins from the lockfile), not `/root/.venv`. Symptom:
`flash_attn` and `vllm` mysteriously revert to wrong versions after
a single `uv run` invocation.

**Fix:** Always `source /root/.venv/bin/activate` first, then run
`python -m ...` directly. Never `uv run` while in the reliquary
project directory.

### flash-attn ABI mismatch after any torch change

flash-attn is compiled against a specific torch version. Any
`uv pip install --reinstall torch` (including transitive bumps from
other installs) breaks it. Symptom:

```
ImportError: .../flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so:
undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...
```

**Fix:**

```bash
uv pip install --reinstall --no-build-isolation flash-attn
```

### transformers 5.x removed `all_special_tokens_extended`

vLLM 0.10 calls `tokenizer.all_special_tokens_extended`, which
transformers 5.x dropped. Symptom:

```
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
```

**Fix:** `vllm_adapter-v4.py` has a `_patch_transformers_for_vllm()`
shim that runs at import time. Don't remove it. (vLLM nightly handles
this internally, but the shim is defensive and harmless.)

### libcudart.so.13 not found

vLLM and torch are out of sync — one is cu128, the other cu130.
Symptom:

```
ImportError: libcudart.so.13: cannot open shared object file
```

**Fix:** Reinstall the full stack in the order in step 2. Don't try
to surgically reinstall only one package.

### DeepGEMM not available

vLLM nightly probes `deep_gemm` during FP8 GEMM warmup. Symptom:

```
RuntimeError: DeepGEMM backend is not available or outdated.
```

**Fix:** `export VLLM_USE_DEEP_GEMM=0`. Don't install deep_gemm
unless you're actually running an FP8-quantized model (Qwen3-4B is
bf16, so deep_gemm wouldn't be used anyway).

### Project venv vs standalone venv

Quick check which venv is active:

```bash
which python                                     # should be /root/.venv/bin/python
python -c "import sys; print(sys.executable)"
uv pip list 2>/dev/null | grep -E "^(torch|vllm|transformers|flash-attn) "
```

If `which python` shows `/root/reliquary/.venv/...`, you're in the
wrong venv. Run `source /root/.venv/bin/activate`.

---

## 9. Optional — FA3 (Hopper/Blackwell-optimized flash-attention)

FA2 (which we installed in step 2) works on B200/H200, but FA3 has
hand-tuned kernels for sm_90a and sm_100 with measurably higher
throughput. It requires building from source (needs nvcc) and a
~30-60 minute build.

Only worth doing if you're squeezing the last few percent of decode
throughput. Skip otherwise.

```bash
# Install CUDA 13 toolkit (no apt route works cleanly here on most
# VPS containers; the .run installer is simplest):
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run
sh cuda_13.0.0_580.65.06_linux.run --toolkit --silent --override --toolkitpath=/usr/local/cuda-13.0
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version  # expect 13.x

# Build FA3
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper
export TORCH_CUDA_ARCH_LIST="10.0"   # B200; use "9.0a" for H200
export MAX_JOBS=8
python setup.py install
```

Then change the FA import in reliquary code from
`from flash_attn import flash_attn_func` to
`from flash_attn_interface import flash_attn_func`.

---

## 10. Cross-box reproduction

To bring a new B200/H200 box online quickly:

1. SSH in, confirm `nvidia-smi` shows the right GPU + driver ≥ 575.
2. Steps 1 → 3 (venv + install + verify) — ~5 minutes.
3. Copy over the v4 overlays (step 4) and wallet keys.
4. Set env vars (step 5).
5. Launch (step 6).

Total: ~10-15 minutes per box, dominated by package downloads.
