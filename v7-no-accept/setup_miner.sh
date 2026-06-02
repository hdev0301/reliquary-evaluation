#!/bin/bash
# Reliquary miner — local environment bootstrap.
#
# Prepares everything needed to run the miner via
# `scripts/launch_miner.sh`:
#   * uv installer (if missing)
#   * a fresh venv at $VENV_DIR (default /root/.venv) on Python 3.12
#   * reliquary installed editable from $INSTALL_DIR (default /root/reliquary)
#   * torch 2.11.0+cu130 — matches the validator's build
#   * transformers 5.8.0 — matches the validator
#   * kernels — HF kernels-hub client; transformers 5.8 falls back to
#     ``kernels-community/flash-attn2`` here when compiled flash-attn
#     isn't available (validator runs FA2; the miner's sketch path
#     must match its numerics or sketch_diff_max blows up)
#   * supabase python client — backs the prepared-prompt cache
#     (reliquary/miner/prompt_picker.py). Tracks per-(ckpt_n,
#     prompt_idx) σ outcomes so the miner avoids prompts known to land
#     out-of-zone (σ < 0.43) for the current checkpoint. Optional —
#     leave SUPABASE_URL/KEY empty to fall back to random prompt pick.
#   * a populated scripts/.env (wallet, HF token, FA2, paths, Supabase)
#
# Usage:
#   # Edit WALLET_* + HF_TOKEN + SUPABASE_* below, then:
#   bash /root/reliquary/setup_miner.sh
#
# Then run the miner:
#   cd $INSTALL_DIR && source scripts/.env && bash scripts/launch_miner.sh
#
# Tail the log:
#   tail -F /root/miner.log | grep --line-buffered -E "submitted|zone-skip|verdict|reason="

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# Operator configuration — EDIT THESE before running.
# ─────────────────────────────────────────────────────────────────────

# Bittensor wallet (must already exist under /root/.bittensor/wallets/).
WALLET_NAME="${WALLET_NAME:-ronnywebdev}"
WALLET_HOTKEY="${WALLET_HOTKEY:-hdev0301}"

# Hugging Face token for pulling the validator's published checkpoint.
HF_TOKEN="${HF_TOKEN:-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}"

# Validator endpoint. The subnet-owner validator during launch phase:
VALIDATOR_URL="${VALIDATOR_URL:-http://86.38.238.30:8080}"

# Supabase prepared-prompt cache. Empty disables — miner falls back to
# uniform-random prompt pick. With it on:
#   * On startup (and on every ckpt_n advance), the miner hydrates a
#     set of "known-bad" prompt indices for the current ckpt_n.
#   * `pick_prompt_idx` skips known-bad in addition to the validator's
#     own cooldown_prompts.
#   * After each rollout group, the miner records (ckpt_n, prompt_idx,
#     k_correct, sigma, outcome) so future workers — and the same
#     worker after restart — avoid re-trying hopeless prompts.
#   * Cross-machine: all boxes sharing the same Supabase project
#     converge on a shared known-bad set.
# KEY must be a service_role JWT (anon/publishable can't write).
# Schema lives at $INSTALL_DIR/sql/supabase_schema.sql — paste it into
# your project's SQL editor once.
SUPABASE_URL="${SUPABASE_URL:-https://fwyglzgpodpflkdoibdh.supabase.co}"
SUPABASE_KEY="${SUPABASE_KEY:-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ3eWdsemdwb2RwZmxrZG9pYmRoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3OTcyNTM1NywiZXhwIjoyMDk1MzAxMzU3fQ.LsqslSQ1ruA7vuef4MUKMOhrun64eQR7aymaQMi3XGg}"

# Layout. Override via env vars if your tree is elsewhere.
INSTALL_DIR="${INSTALL_DIR:-/root/reliquary}"
VENV_DIR="${VENV_DIR:-/root/.venv}"

# ─────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────

if [ ! -d "$INSTALL_DIR" ]; then
  echo "ERROR: $INSTALL_DIR does not exist. Clone the repo first:"
  echo "  git clone <repo-url> $INSTALL_DIR"
  exit 1
fi

if [ ! -f "/root/.bittensor/wallets/$WALLET_NAME/hotkeys/$WALLET_HOTKEY" ]; then
  echo "ERROR: wallet hotkey file missing at"
  echo "  /root/.bittensor/wallets/$WALLET_NAME/hotkeys/$WALLET_HOTKEY"
  echo "Create one with: btcli wallet new-hotkey --wallet.name $WALLET_NAME --wallet.hotkey $WALLET_HOTKEY"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi missing. The miner needs a CUDA-capable GPU."
  exit 1
fi

echo "[setup] GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2 | sed 's/^/  /'

# ─────────────────────────────────────────────────────────────────────
# uv installer
# ─────────────────────────────────────────────────────────────────────

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv (fast Python package manager)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] uv: $(uv --version)"

# ─────────────────────────────────────────────────────────────────────
# Venv (Python 3.12 — matches validator)
# ─────────────────────────────────────────────────────────────────────

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating venv at $VENV_DIR"
  uv venv "$VENV_DIR" --python 3.12
fi

# ─────────────────────────────────────────────────────────────────────
# Reliquary install (editable)
# ─────────────────────────────────────────────────────────────────────

echo "[setup] installing reliquary editable from $INSTALL_DIR"
VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$INSTALL_DIR"

# ─────────────────────────────────────────────────────────────────────
# Pin torch to validator's build (cu130, sm_120 Blackwell). The default
# torch wheel pulled by the bittensor / reliquary chain may not match —
# install cu130 explicitly so the miner's sketches match the validator's
# bit-for-bit.
# ─────────────────────────────────────────────────────────────────────

echo "[setup] pinning torch==2.11.0+cu130"
VIRTUAL_ENV="$VENV_DIR" uv pip install \
  'torch==2.11.0' \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple

# ─────────────────────────────────────────────────────────────────────
# Pin transformers to the validator's version
# ─────────────────────────────────────────────────────────────────────

echo "[setup] pinning transformers==5.8.0"
VIRTUAL_ENV="$VENV_DIR" uv pip install 'transformers==5.8.0'

# HF kernels-hub client. The miner runs the HF generation+sketch model
# with attn_implementation=flash_attention_2 (mandated by validator —
# sketches are bit-sensitive). On stacks where compiled flash-attn has
# no prebuilt wheel (torch 2.11 + cu130 + sm_120 Blackwell is one such),
# transformers 5.8 auto-falls back to ``kernels-community/flash-attn2``
# from the HF kernels hub IF the ``kernels`` pip package is installed.
# Without it, model load raises ``ImportError: the package for
# FlashAttention2 doesn't seem to be installed``.
echo "[setup] installing kernels (HF kernels-hub client for FA2 fallback)"
VIRTUAL_ENV="$VENV_DIR" uv pip install 'kernels'

# Supabase python client for the prepared-prompt cache.
echo "[setup] installing supabase (prepared-prompt cache)"
VIRTUAL_ENV="$VENV_DIR" uv pip install 'supabase'

# vLLM — batched M=8 rollout engine. reliquary/cli/main.py imports
# `vllm.LLM` at startup; without it the miner crashes with
# ModuleNotFoundError before the first window. vllm 0.21 is compatible
# with torch 2.11.0+cu130 (verified via uv resolver — preserves the
# pinned torch and transformers above; only pulls torchvision/audio at
# matching 2.11).
echo "[setup] installing vllm (batched rollout engine)"
VIRTUAL_ENV="$VENV_DIR" uv pip install 'vllm'

# ─────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────

echo "[setup] verifying stack"
"$VENV_DIR/bin/python" - <<'PY'
import torch, transformers, sys
print(f"python: {sys.version.split()[0]}")
print(f"torch:  {torch.__version__}  cuda={torch.version.cuda}")
print(f"transformers: {transformers.__version__}")
import vllm
print(f"vllm: {vllm.__version__}")
try:
    import supabase
    print(f"supabase: {supabase.__version__}")
except ImportError:
    print("supabase: (not installed; SUPABASE_URL=... will be a no-op)")
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"capability: {torch.cuda.get_device_capability(0)}")
PY

# ─────────────────────────────────────────────────────────────────────
# Populate scripts/.env — only env vars the codebase actually reads.
# ─────────────────────────────────────────────────────────────────────

ENV_FILE="$INSTALL_DIR/scripts/.env"
echo "[setup] writing $ENV_FILE"
cat > "$ENV_FILE" <<EOF
# Reliquary miner runtime environment — generated by setup_miner.sh.
# Only contains env vars actually consumed by the codebase.

# Network / chain (consumed by reliquary.infrastructure.chain).
export BT_NETWORK=finney
export NETUID=81

# Wallet — read by scripts/launch_miner.sh, passed to CLI as --wallet-name/--hotkey.
export BT_WALLET_NAME=$WALLET_NAME
export BT_HOTKEY=$WALLET_HOTKEY

# Cold-start checkpoint — read by scripts/launch_miner.sh as --checkpoint.
# At startup the miner pulls the validator's currently-published
# fine-tuned checkpoint (state.checkpoint_repo_id @ state.checkpoint_revision)
# and uses that instead. This var only matters if the validator's /state
# returns no published checkpoint yet (subnet cold-launch).
export RELIQUARY_CHECKPOINT=Qwen/Qwen3-4B-Instruct-2507

# Attention. Validator runs FlashAttention 2; sketch commitments are
# bit-sensitive to attention kernel variance. Setting GRAIL_ATTN_IMPL
# is the only honoured way to override (reliquary/constants.py:56).
# flash-attn proper has no prebuilt wheel for torch 2.11+cu130+py3.12+
# sm_120 Blackwell; transformers 5.8 auto-falls back to
# kernels-community/flash-attn2 on the HF kernels hub when the
# 'kernels' pip pkg is installed (handled above).
export GRAIL_ATTN_IMPL=flash_attention_2

# HF token to pull validator's published checkpoint.
export HF_TOKEN=$HF_TOKEN
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Paths — read by scripts/launch_miner.sh.
export RELIQUARY_INSTALL_DIR=$INSTALL_DIR
export RELIQUARY_VENV=$VENV_DIR

# Drand — read by scripts/launch_miner.sh.
export RELIQUARY_USE_DRAND=1

# Validator URL — pinned during the subnet-launch phase. After the
# owner validator gains validator_permit the miner could auto-discover
# via the metagraph; leaving this set is safer for now.
export RELIQUARY_VALIDATOR_URL=$VALIDATOR_URL

# OpenMathInstruct dataset shards to load locally. The dataset is 32
# shards / 14M problems total; the default 2 shards (~880k problems,
# ~1 GB on disk) is plenty for prompt diversity. Raise if you start
# exhausting in-zone prompts after long runs.
export RELIQUARY_OMI_SHARDS=2

# CUDA allocator. expandable_segments coalesces reservations under
# the gen+sketch fragmentation pattern.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Supabase prepared-prompt cache (reliquary/miner/prompt_picker.py).
# Empty disables — miner falls back to uniform-random prompt pick.
# KEY must be the service_role JWT (anon/publishable can't write).
# Schema: \$INSTALL_DIR/sql/supabase_schema.sql — paste into the
# project's SQL editor once before first launch.
export RELIQUARY_SUPABASE_URL=$SUPABASE_URL
export RELIQUARY_SUPABASE_KEY=$SUPABASE_KEY

# Sigma floor for the local zone-skip pre-check. Default 0.43 matches
# the validator's steady-state SIGMA_MIN (reliquary/constants.py:211).
# Set to 0.33 if you know the validator is in bootstrap mode
# (BOOTSTRAP_SIGMA_MIN). Lower threshold = more submissions but more
# rejects; higher threshold = fewer submissions but cleaner accept rate.
export RELIQUARY_SIGMA_MIN=0.43

# Max tokens per rollout. Protocol cap is 8192; running at the cap on
# HF transformers .generate() takes ~3-5 min for one M=8 batch — far
# longer than the 60s window, so every batch crosses a window boundary
# and gets abandoned by the race guard. Lowering this trades cap-
# truncation risk (MAX_TRUNCATED_PER_SUBMISSION=5 per batch) for fitting
# inside a window. 1024 tokens × 8 ≈ 30s on Blackwell — fits comfortably.
# Keep at the protocol cap. Rollouts that EOS naturally avoid the
# cap-truncation budget; rollouts that hit 8192 take verifier Path 1
# (auto-pass) and dodge the vLLM↔HF p_stop drift that triggers
# bad_termination at shorter caps.
export RELIQUARY_MAX_NEW_TOKENS=8192

export PYTHONUNBUFFERED=1
EOF

# ─────────────────────────────────────────────────────────────────────
# Patch launch_miner.sh defaults if needed.
# ─────────────────────────────────────────────────────────────────────

LAUNCH="$INSTALL_DIR/scripts/launch_miner.sh"
if [ -f "$LAUNCH" ]; then
  if ! grep -q "RELIQUARY_VENV" "$LAUNCH"; then
    echo "[setup] patching $LAUNCH to honour RELIQUARY_VENV"
    sed -i '0,/PID_FILE=/s||VENV_DIR="${RELIQUARY_VENV:-/root/.venv}"\nPID_FILE=|' "$LAUNCH"
    sed -i 's|.venv/bin/python|"$VENV_DIR/bin/python"|g' "$LAUNCH"
  fi
fi

echo
echo "✅ Setup complete."
echo
if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_KEY:-}" ]; then
  echo "Supabase prepared-prompt cache is configured. If you haven't yet,"
  echo "paste $INSTALL_DIR/sql/supabase_schema.sql into your Supabase"
  echo "project's SQL editor before the first launch."
  echo
fi
echo "Next steps:"
echo "  cd $INSTALL_DIR"
echo "  source scripts/.env"
echo "  bash scripts/launch_miner.sh"
echo
echo "Tail the log:"
echo "  tail -F /root/miner.log | grep --line-buffered -E \"submitted|zone-skip|verdict|reason=\""
