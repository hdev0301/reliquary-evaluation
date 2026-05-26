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
#   * vllm 0.20.2 (auto-selects FlashAttention 2 backend)
#   * kernels — HF kernels hub client; transformers falls back to
#     ``kernels-community/flash-attn2`` here when local flash-attn
#     isn't compiled (validator runs FA2; the miner's HF sketch must
#     match its numerics or sketch_diff_max blows up)
#   * supabase — persistence cache (dud_set + pregen_batches across
#     machines/restarts/checkpoints)
#   * a populated scripts/.env (wallet, HF token, FA2, EosGuard, paths)
#
# Usage:
#   # Edit the WALLET_* and HF_TOKEN values below FIRST, then:
#   bash /root/setup_miner.sh
#
# Then run the miner:
#   cd $INSTALL_DIR && source scripts/.env && bash scripts/launch_miner.sh
#
# Tail the log:
#   tail -F /root/miner.log | grep --line-buffered -E "submitted|pregen ready|verdict|reason="
#
# Check accepts on the validator:
#   HK=$(python3 -c "import json; print(json.load(open('/root/.bittensor/wallets/$WALLET_NAME/hotkeys/$WALLET_HOTKEY'))['ss58Address'])")
#   curl -s "http://$VALIDATOR_IP:$VALIDATOR_PORT/verdicts/$HK" | python3 -m json.tool

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

# Supabase persistence cache — shared across machines so the local
# miner + sibling scrape_intel.py + sibling prep_dataset.py boxes all
# hydrate from the same (dud_set, known_good, pregen_batches) state.
# The KEY must be a service_role JWT (anon/publishable can't write).
# Leave empty to disable persistence on this machine.
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
# Pin torch to validator's build. The default torch wheel pulled by
# the bittensor / reliquary chain may not match — explicitly install
# the cu130 build to match the validator's sketches bit-for-bit.
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

# ─────────────────────────────────────────────────────────────────────
# vLLM 0.20.x — fast generation engine used when RELIQUARY_USE_VLLM=1.
# GRAIL sketches still run on transformers (vLLM doesn't expose per-
# layer hidden states), so the HF model loads in addition to vLLM.
# On a Blackwell / 80+ GB card both fit with gpu_mem_util=0.50.
# Skip with INSTALL_VLLM=0 if you don't want it (e.g. small VRAM box).
# ─────────────────────────────────────────────────────────────────────

if [ "${INSTALL_VLLM:-1}" = "1" ]; then
  echo "[setup] installing vllm==0.20.2 (downgrades starlette + adds ~50 deps)"
  VIRTUAL_ENV="$VENV_DIR" uv pip install 'vllm==0.20.2'
  echo "[setup] installing supabase (persistence cache for prompt outcomes + batches)"
  VIRTUAL_ENV="$VENV_DIR" uv pip install 'supabase'
else
  echo "[setup] skipping vllm (INSTALL_VLLM=0)"
fi

# HF kernels hub client. The miner's HF sketch model runs with
# attn_implementation=flash_attention_2 to match the validator's FA2
# forward (sketch_diff_max collapses when both kernels agree). On
# stacks where flash-attn proper has no prebuilt wheel (torch 2.11 +
# cu130 + sm_120 Blackwell is one such), transformers 5.8 auto-falls
# back to ``kernels-community/flash-attn2`` from the HF hub IF the
# ``kernels`` pip package is installed. Without it, model load raises
# ``ImportError: the package for FlashAttention2 doesn't seem to be
# installed``.
echo "[setup] installing kernels (HF kernels-hub client for FA2 fallback)"
VIRTUAL_ENV="$VENV_DIR" uv pip install 'kernels'

# ─────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────

echo "[setup] verifying stack"
"$VENV_DIR/bin/python" - <<'PY'
import torch, transformers, sys
print(f"python: {sys.version.split()[0]}")
print(f"torch:  {torch.__version__}  cuda={torch.version.cuda}")
print(f"transformers: {transformers.__version__}")
try:
    import vllm
    print(f"vllm: {vllm.__version__}")
except ImportError:
    print("vllm: (not installed; RELIQUARY_USE_VLLM=1 will fail)")
try:
    import supabase
    print(f"supabase: {supabase.__version__}")
except ImportError:
    print("supabase: (not installed; RELIQUARY_SUPABASE_URL=... will be a no-op)")
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"capability: {torch.cuda.get_device_capability(0)}")
PY

# ─────────────────────────────────────────────────────────────────────
# Populate scripts/.env
# ─────────────────────────────────────────────────────────────────────

ENV_FILE="$INSTALL_DIR/scripts/.env"
echo "[setup] writing $ENV_FILE"
cat > "$ENV_FILE" <<EOF
# Reliquary runtime environment — generated by setup_miner.sh

# Network / chain
export BT_NETWORK=finney
export NETUID=81

# Wallet
export BT_WALLET_NAME=$WALLET_NAME
export BT_HOTKEY=$WALLET_HOTKEY

# Base model. The validator may push fine-tuned checkpoints over HF;
# this is the cold-start default if the validator isn't reachable yet.
export RELIQUARY_CHECKPOINT=Qwen/Qwen3-4B-Instruct-2507

# Attention. Validator runs FlashAttention 2; match it on both gen and
# sketch paths or sketch_diff_max blows up and the validator rejects
# with bad_termination. vLLM 0.20.2 auto-selects FLASH_ATTN backend
# (FA2) on Hopper/Blackwell. HF transformers needs attn_impl set
# explicitly to flash_attention_2; on stacks without compiled
# flash-attn, transformers 5.8 falls back to kernels-community/
# flash-attn2 from HF hub (requires the ``kernels`` pip pkg installed
# above).
export RELIQUARY_ATTN_IMPL=flash_attention_2
export GRAIL_ATTN_IMPL=flash_attention_2

# HF token to pull published checkpoints from the validator's HF repo.
export HF_TOKEN=$HF_TOKEN
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Pin paths
export RELIQUARY_INSTALL_DIR=$INSTALL_DIR
export RELIQUARY_VENV=$VENV_DIR

# Max generation length. Protocol cap is 8192; keep at the max so
# rollouts that hit our cap also satisfy the validator's max-length
# termination path (prompt_length + completion_length >= 8192).
export RELIQUARY_MAX_NEW_TOKENS=8192

# Drand
export RELIQUARY_USE_DRAND=1

# Validator URL — pinned during the subnet-launch phase. After the
# owner validator gains validator_permit, this can be left empty and
# the miner auto-discovers via the metagraph.
export RELIQUARY_VALIDATOR_URL=$VALIDATOR_URL

# Validator-only (left empty for miner)
export RELIQUARY_EXTERNAL_IP=
export RELIQUARY_EXTERNAL_PORT=

# R2 / S3 (validator-only; safe to leave dummy on miner)
export R2_ENDPOINT_URL=https://s3.us-east-1.amazonaws.com
export R2_ACCOUNT_ID=dummy
export R2_ACCESS_KEY_ID=
export R2_SECRET_ACCESS_KEY=
export R2_REGION=us-east-1
export R2_BUCKET_ID=grail-catalyst-test

# State HMAC (validator-only)
export GRAIL_STATE_HMAC_KEY=catalyst-local-test-hmac

# Smart-miner knobs (engine-side defaults are fine; these are
# convenience env overrides if you want to tune without code changes).
export RELIQUARY_GEN_BATCH_PROMPTS=2
export RELIQUARY_PRESCREEN_ROLLOUTS=8
export RELIQUARY_PRESCREEN_MAX_TOKENS=1024
export RELIQUARY_PREGEN_CAPACITY=48
export RELIQUARY_SHARE_MODEL=1

# vLLM gen engine. When set, generation uses vLLM 0.20.x (~2-3x
# throughput vs transformers.generate). GRAIL sketches still run
# on the HF model. Two model instances on GPU: vLLM gpu_mem_util=0.50
# + HF ~8 GB for Qwen3-4B in bf16. Requires INSTALL_VLLM=1 above.
# Comment out (or set to 0) to fall back to pure-HF generation.
export RELIQUARY_USE_VLLM=1

# VllmV1EosGuard logits processor. Registered on the vLLM engine via
# cli/main.py to mask EOS tokens to -inf at decode positions where raw
# p(EOS) falls below RELIQUARY_VLLM_EOS_THRESHOLD. Without it, vLLM
# samples EOS at borderline-confidence positions that the validator's
# HF forward then rejects with bad_termination(low_p_stop). The IDS
# env var is required — the guard's __init__ raises if absent.
# Qwen3-Instruct's generation_config.eos_token_id lists both
# <|im_end|>=151645 and <|endoftext|>=151643; include both so a stop
# on either trims correctly downstream.
export RELIQUARY_VLLM_EOS_IDS=151643,151645
# Threshold = validator's MIN_EOS_PROBABILITY floor (0.005) plus
# headroom. Going lower lets more rollouts terminate but adds
# low_p_stop preverify-skips; going higher reduces drift cases but
# more rollouts hit max_new_tokens cap and need regen.
export RELIQUARY_VLLM_EOS_THRESHOLD=0.008

# CUDA allocator. vLLM + HF on the same GPU fragment the bf16 arena
# enough that a single 5 GiB log_softmax for the sketch can fail to
# allocate even when nominally free VRAM is enough. expandable_segments
# coalesces reservations to fix this.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Submitter-only mode. When set, the local pregen worker is disabled
# and this miner only submits batches that some other machine wrote
# into Supabase pregen_batches (via scripts/prep_dataset.py). Default
# OFF — most operators want self-contained gen+submit.
# export RELIQUARY_DISABLE_LOCAL_GEN=1

# Supabase persistence cache (optional). When both URL + KEY are set,
# the miner hydrates dud_set + known_good + pregen_queue from a shared
# Supabase project on every ckpt advance, and persists outcomes +
# batches as it runs. Cross-machine + cross-restart durability for the
# expensive pregen work. Same project should be set on sibling boxes
# running scripts/scrape_intel.py and scripts/prep_dataset.py. The KEY
# must be the service_role JWT (anon/publishable can't write). Schema:
# $INSTALL_DIR/sql/supabase_schema.sql — paste into your project's SQL
# editor once. Leave empty to disable.
export RELIQUARY_SUPABASE_URL=$SUPABASE_URL
export RELIQUARY_SUPABASE_KEY=$SUPABASE_KEY

export PYTHONUNBUFFERED=1
EOF

# ─────────────────────────────────────────────────────────────────────
# Patch launch_miner.sh defaults if needed.
# ─────────────────────────────────────────────────────────────────────

LAUNCH="$INSTALL_DIR/scripts/launch_miner.sh"
if [ -f "$LAUNCH" ]; then
  if ! grep -q "RELIQUARY_VENV" "$LAUNCH"; then
    echo "[setup] patching $LAUNCH to honour RELIQUARY_VENV"
    # Insert VENV_DIR line near the top of the file
    sed -i '0,/PID_FILE=/s||VENV_DIR="${RELIQUARY_VENV:-/root/.venv}"\nPID_FILE=|' "$LAUNCH"
    # Replace `.venv/bin/python` with $VENV_DIR/bin/python
    sed -i 's|.venv/bin/python|"$VENV_DIR/bin/python"|g' "$LAUNCH"
  fi
fi

echo
echo "✅ Setup complete."
echo
echo "Next steps:"
echo "  cd $INSTALL_DIR"
echo "  source scripts/.env"
echo "  bash scripts/launch_miner.sh"
echo
echo "Tail the log:"
echo "  tail -F /root/miner.log | grep --line-buffered -E \"submitted|pregen ready|verdict|reason=\""
