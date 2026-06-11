#!/bin/bash
# Launch the optimized OpenCode miner (vLLM + pregen + local frontier oracle).
#
# Usage:
#     cp mining/opencode/.env.example mining/opencode/.env  # then edit
#     source mining/opencode/.env
#     bash mining/opencode/run.sh
#
# Prerequisites (build once via mining/setup.sh):
#   - .venv         validator-matched proof venv (torch 2.7.0, transformers 5.9.0, flash-attn)
#   - .venv-vllm    vLLM generation venv
#   - the frontier oracle: .venv/bin/python -m mining.scripts.build_local_oracle

set -euo pipefail

REPO_DIR="${RELIQUARY_INSTALL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG_FILE="${RELIQUARY_MINER_LOG:-/root/opencode_miner.log}"
PID_FILE="${RELIQUARY_MINER_PID:-/root/opencode_miner.pid}"

: "${BT_WALLET_NAME?set BT_WALLET_NAME (source mining/opencode/.env)}"
: "${BT_HOTKEY?set BT_HOTKEY (source mining/opencode/.env)}"

# OpenCode prompt-only mode is mandatory on miners (never load hidden cases).
export RELIQUARY_OCI_PROMPT_ONLY=1
export RELIQUARY_ENVIRONMENTS=opencodeinstruct
export RELIQUARY_OCI_SUBSET_REPO="${RELIQUARY_OCI_SUBSET_REPO:-R0mAI/opencodeinstruct-prompts}"
export RELIQUARY_OCI_ORACLE_PATH="${RELIQUARY_OCI_ORACLE_PATH:-mining/state/opencode_oracle.json.gz}"
# Separate vLLM venv python (generation worker). Absolute path so the miner can
# spawn it from any cwd.
export RELIQUARY_VLLM_PYTHON="${RELIQUARY_VLLM_PYTHON:-$REPO_DIR/.venv-vllm/bin/python}"

cd "$REPO_DIR"

if [ ! -f "$RELIQUARY_OCI_ORACLE_PATH" ]; then
  echo "ERROR: frontier oracle missing at $RELIQUARY_OCI_ORACLE_PATH" >&2
  echo "Build it once: .venv/bin/python -m mining.scripts.build_local_oracle --out '$RELIQUARY_OCI_ORACLE_PATH'" >&2
  exit 1
fi
if [ ! -x "$RELIQUARY_VLLM_PYTHON" ]; then
  echo "WARNING: vLLM venv python not found at $RELIQUARY_VLLM_PYTHON" >&2
  echo "Run mining/setup.sh to build it, or the miner falls back to in-process vLLM" >&2
  echo "(which is NOT validator-compatible with transformers==5.9.0)." >&2
fi

pkill -9 -f "mining.opencode.miner" 2>/dev/null || true
pkill -9 -f "mining.common.vllm_worker" 2>/dev/null || true
# vLLM spawns an EngineCore child that survives the worker pkill and keeps the
# GPU pinned. Kill anything from the vLLM venv, then any remaining GPU compute
# app (safe on a dedicated mining box — set RELIQUARY_NO_GPU_REAP=1 to skip).
pkill -9 -f "$REPO_DIR/.venv-vllm" 2>/dev/null || true
if [ "${RELIQUARY_NO_GPU_REAP:-0}" != "1" ]; then
  for gpid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$gpid" 2>/dev/null || true
  done
fi
sleep 2
rm -f "$LOG_FILE" "$PID_FILE"

# Launch the miner via uv in the validator-matched .venv; it spawns the vLLM
# generation worker in .venv-vllm itself.
nohup uv run --no-sync python -m mining.opencode.miner > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Launched OpenCode miner PID=$(cat "$PID_FILE"), log: $LOG_FILE"
