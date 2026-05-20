#!/usr/bin/env bash
# Launch script for the reliquary miner on a single-H200 box, configured to
# match the dependency stack and validator behaviour proven working in
# previous runs of this session.
#
# Environment layout this script expects:
#   * Python venv at $VENV (default /root/.venv) created with `uv venv`
#   * In that venv:
#       torch 2.11.x          (cu130 wheel — works against system cu12.8)
#       vllm 0.21.x           (cu13-linked; flashinfer JIT against cu12.8)
#       transformers 5.8.x
#       flash_attn 2.8.x      (cu12+torch2.11 prebuilt wheel)
#       flashinfer 0.6.8.x    (cu12 prebuilt, no JIT pre-compile needed)
#       bittensor 10.2.x      (installed by this script if missing)
#       reliquary             (editable install of /root/reliquary)
#       ninja                 (used at runtime by flashinfer's JIT)
#   * System CUDA toolkit at /usr/local/cuda → /usr/local/cuda-12.8
#     (full dev install: nvcc, includes, libcudart symlinks, stubs/)
#   * NVIDIA H200 single GPU (sm_90, 143 GiB)
#   * Bittensor wallet at ~/.bittensor/wallets/$WALLET/hotkeys/$HOTKEY
#   * HF token written to ~/.cache/huggingface/token (mode 600). The token
#     is NEVER passed on the command line or via env var — huggingface_hub
#     reads it from the cache file directly.
#
# To run:  ./launch-miner.sh    (logs to ~/reliquary-miner.log by default)
# To stop: pkill -f 'reliquary mine'  (then watch GPU clears with nvidia-smi)

set -euo pipefail

# ───────────────────────── tunables (env-overridable) ──────────────────────
VENV=${VENV:-/root/.venv}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
LOG=${LOG:-$HOME/reliquary-miner.log}

WALLET=${WALLET:-ronnywebdev}
HOTKEY=${HOTKEY:-hdev0301}
NETUID=${NETUID:-81}
NETWORK=${NETWORK:-finney}
# Pin the validator URL during subnet-launch phase (owner validator's hotkey
# 5CXzFHfeiJ4Xkiirq4ej1MrRVCd789wEJXhpf2ZKRW6MNFJF may not hold validator_permit
# yet, breaking metagraph auto-discovery). Set to "" to let the miner discover
# from metagraph instead.
VALIDATOR_URL=${VALIDATOR_URL:-http://86.38.238.30:8080}

# Model + mining tuning — matches what the H200 + this venv can sustain.
CHECKPOINT=${CHECKPOINT:-Qwen/Qwen3-4B-Instruct-2507}
USE_VLLM=${USE_VLLM:-true}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.55}
PREGEN_BUFFER=${PREGEN_BUFFER:-4}
STATE_POLL_MS=${STATE_POLL_MS:-100}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}     # protocol cap; 1/8 truncation budget
MIN_SIGMA=${MIN_SIGMA:-0.43}               # validator's steady-state SIGMA_MIN
PRESCREEN_K=${PRESCREEN_K:-4}
DIFFICULTY_BLACKLIST=${DIFFICULTY_BLACKLIST:-4096}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# ───────────────────────── pre-flight checks ──────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "$VENV/bin/python" ]] || die "venv not found at $VENV (set VENV=...)"
[[ -x "$VENV/bin/reliquary" ]] || die "$VENV doesn't have the reliquary CLI; run 'uv pip install -e /root/reliquary --python $VENV/bin/python'"
[[ -x "$VENV/bin/ninja" ]] || die "ninja missing from $VENV (flashinfer JIT needs it); run 'uv pip install ninja --python $VENV/bin/python'"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "CUDA toolkit nvcc not at $CUDA_HOME/bin/nvcc"

[[ -s "$HOME/.cache/huggingface/token" ]] \
  || die "HF token missing at ~/.cache/huggingface/token (mode 600). Write your read-token there before launching."

[[ -f "$HOME/.bittensor/wallets/$WALLET/hotkeys/$HOTKEY" ]] \
  || die "wallet hotkey not found at ~/.bittensor/wallets/$WALLET/hotkeys/$HOTKEY"

# Verify the venv actually has the modules — fast import smoke test.
"$VENV/bin/python" -c "import torch, vllm, transformers, flash_attn, flashinfer" 2>/dev/null \
  || die "venv import smoke test failed — one of {torch,vllm,transformers,flash_attn,flashinfer} broken in $VENV"

# Auto-install bittensor if missing. The user's freshly-rebuilt venv tends
# to miss this since it's not in vLLM's deps tree.
if ! "$VENV/bin/python" -c "import bittensor" 2>/dev/null; then
  echo "[launch] bittensor missing; installing into $VENV..."
  /root/.local/bin/uv pip install --python "$VENV/bin/python" "bittensor>=10,<11" \
    || die "bittensor install failed"
fi

# Kill any orphan EngineCore / resource_tracker from a previous crashed run.
# vLLM holds the GPU until its EngineCore subprocess actually exits, and a
# leaked multiprocessing.resource_tracker keeps shared-memory segments alive
# (and therefore GPU memory). Skipping this leads to OOM on launch.
orphans=$(pgrep -f 'VLLM::EngineCore|multiprocessing.resource_tracker|reliquary mine' 2>/dev/null || true)
if [[ -n "$orphans" ]]; then
  echo "[launch] killing orphan processes: $orphans"
  kill -9 $orphans 2>/dev/null || true
  sleep 3
fi

# ───────────────────────── runtime env vars ───────────────────────────────
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Prepend venv bin + CUDA bin to PATH so:
#   * `ninja` is reachable from the vLLM EngineCore subprocess (flashinfer JIT)
#   * `nvcc` is reachable for any lazy CUDA compile
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"

# vLLM forks an EngineCore subprocess. If CUDA is initialised in the parent
# first (which it is — main.py probes torch.cuda.device_count() before
# constructing vLLM), the forked child crashes with
#   "Cannot re-initialize CUDA in forked subprocess"
# Forcing 'spawn' gives a fresh Python process that initialises CUDA cleanly.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Helps any kernel that respects this hint (flash-attn build, flashinfer JIT,
# torch's lazy compile). H200 = sm_90.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"9.0+PTX"}

# Bittensor reads these for network selection; passing them via env is
# equivalent to the CLI flags but cleaner for restarts.
export BT_NETWORK="$NETWORK"
export NETUID

# ───────────────────────── build & exec ───────────────────────────────────
ARGS=(
  --network "$NETWORK"
  --netuid "$NETUID"
  --wallet-name "$WALLET"
  --hotkey "$HOTKEY"
  --checkpoint "$CHECKPOINT"
  --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
  --pregen-buffer-size "$PREGEN_BUFFER"
  --state-poll-ms "$STATE_POLL_MS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --min-sigma "$MIN_SIGMA"
  --prescreen-k "$PRESCREEN_K"
  --difficulty-blacklist-size "$DIFFICULTY_BLACKLIST"
  --log-level "$LOG_LEVEL"
)
if [[ "$USE_VLLM" == "true" ]]; then
  ARGS+=(--use-vllm)
else
  ARGS+=(--no-vllm)
fi
if [[ -n "$VALIDATOR_URL" ]]; then
  ARGS+=(--validator-url "$VALIDATOR_URL")
fi

echo "[launch] venv=$VENV cuda=$CUDA_HOME wallet=$WALLET/$HOTKEY backend=$([ "$USE_VLLM" = true ] && echo vLLM || echo HF)"
echo "[launch] logging to $LOG"
cd /root/reliquary
exec "$VENV/bin/reliquary" mine "${ARGS[@]}" >> "$LOG" 2>&1
