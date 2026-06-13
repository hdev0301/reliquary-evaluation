#!/bin/bash
# ---------------------------------------------------------------------------
# Reliquary optimized-miner environment setup (uv, validator-compatible).
#
# Reproducible, idempotent provisioning for a FRESH dev/mining machine. Builds
# TWO uv venvs because the validator-matched proof stack and a transformers-5.9
# -compatible vLLM cannot share one torch:
#
#   .venv        main / PROOF venv — EXACTLY mirrors the validator Dockerfile:
#                torch 2.7.0+cu128, flash-attn 2.8.3, flash-linear-attention
#                0.5.0, transformers 5.9.0, reliquary. Runs the miner + GRAIL
#                proofs. No vLLM here.
#   .venv-vllm   generation venv — vLLM (its own newer torch). Runs only the
#                token-id worker (mining.common.vllm_worker). Its torch never
#                touches the GRAIL proof, so it is decoupled from consensus.
#
# Usage:
#     bash mining/setup.sh                # build both venvs
#     bash mining/setup.sh --with-oracle  # also build the frontier oracle
#
# Validator-compat invariants (do NOT relax for mainnet): transformers pinned
# exact (PROMPT_MISMATCH), flash-attn present (GRAIL sketch kernel), flash-linear
# -attention 0.5.0 (Qwen3.5 GatedDeltaNet kernel), bf16, same GPU class (H200).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

WITH_ORACLE=0
for arg in "$@"; do
  case "$arg" in
    --with-oracle) WITH_ORACLE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }

# Versions pinned to the validator Dockerfile — keep in sync if it bumps.
TORCH_VER="2.7.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
FLA_VER="0.5.0"

# --- 0. uv -----------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "uv: $(uv --version)"

TF_PIN="$(grep -oE 'transformers==[0-9.]+' pyproject.toml | head -1)"
[ -n "$TF_PIN" ] || { echo "could not read transformers pin from pyproject.toml" >&2; exit 1; }

# --- 1. PROOF venv (.venv) — exact validator stack -------------------------
log "building proof venv (.venv): torch ${TORCH_VER}+cu128, ${TF_PIN}, flash-attn 2.8.3, fla ${FLA_VER}"
uv venv --python 3.12 .venv
PROOF_PY="$REPO_DIR/.venv/bin/python"
# torch first, from the CUDA 12.8 index (matches the validator exactly).
uv pip install --python "$PROOF_PY" "torch==${TORCH_VER}" --index-url "$TORCH_INDEX"
# flash-attn prebuilt wheel (no nvcc build) + Qwen3.5 GatedDeltaNet kernel.
uv pip install --python "$PROOF_PY" "$FA_URL"
uv pip install --python "$PROOF_PY" "flash-linear-attention==${FLA_VER}"
# reliquary + deps; transformers re-pinned so nothing bumps it off 5.9.0.
uv pip install --python "$PROOF_PY" -e . "$TF_PIN"
# bittensor substrate fixup (same as the validator Dockerfile).
uv pip install --python "$PROOF_PY" 'async-substrate-interface<2.0.0' || true
uv pip install --python "$PROOF_PY" --force-reinstall --no-deps scalecodec==1.2.12 || true

# --- 2. vLLM venv (.venv-vllm) — generation only ---------------------------
log "building vLLM venv (.venv-vllm)"
uv venv --python 3.12 .venv-vllm
VLLM_PY="$REPO_DIR/.venv-vllm/bin/python"
# Pin transformers so vLLM's resolver stays consistent; torch floats to vLLM's.
uv pip install --python "$VLLM_PY" "$TF_PIN" vllm

# --- 3. verify -------------------------------------------------------------
log "verifying proof stack"
"$PROOF_PY" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| dev", torch.cuda.device_count())
assert transformers.__version__.startswith("5.9."), f"transformers must be 5.9.x, got {transformers.__version__}"
print("transformers", transformers.__version__, "(validator-pinned)")
import flash_attn; print("flash_attn", flash_attn.__version__)
import fla; print("flash-linear-attention", getattr(fla, "__version__", "ok"))
import reliquary, datasets, bittensor  # noqa
import mining.tests.test_pregen_sketch as t
t.test_pregen_split_matches_verifier()
PY
log "verifying vLLM venv"
"$VLLM_PY" -c "import vllm; print('vllm', vllm.__version__)"

# --- 4. optional frontier oracle -------------------------------------------
ORACLE_PATH="${RELIQUARY_OCI_ORACLE_PATH:-mining/state/opencode_oracle.json.gz}"
if [ "$WITH_ORACLE" = "1" ] && [ ! -f "$ORACLE_PATH" ]; then
  log "building frontier oracle (streams nvidia/OpenCodeInstruct; minutes)"
  "$PROOF_PY" -m mining.scripts.build_local_oracle --out "$ORACLE_PATH"
fi

log "done. Next:"
echo "    cp mining/opencode/.env.example mining/opencode/.env   # edit wallet + validator"
[ -f "$ORACLE_PATH" ] || echo "    .venv/bin/python -m mining.scripts.build_local_oracle --out $ORACLE_PATH"
echo "    source mining/opencode/.env && bash mining/opencode/run.sh"
