#!/bin/bash
# Launch the optimized OpenMath miner (vLLM + pregen + answer-match oracle).
#
# Usage:
#     cp mining/openmath/.env  # edit it
#     source mining/openmath/.env
#     bash mining/openmath/run.sh
#
# Prerequisites (build once via mining/setup.sh — SAME venvs as opencode):
#   - .venv         validator-matched proof venv (torch 2.7.0, transformers 5.9.0)
#   - .venv-vllm    vLLM generation venv
# OpenMath needs NO frontier-oracle sidecar and NO sandbox grader: the dataset's
# public expected_answer is the ground truth and the validator's own matcher
# grades it. The OpenMathInstruct shards download on first run (RELIQUARY_OMI_SHARDS).
#
# ── CO-LOCATION WARNING ──────────────────────────────────────────────────────
# This miner shares the repo, the .venv-vllm worker module, and (by default)
# GPU 0 with the opencode miner. Running BOTH on one box safely requires
# DIFFERENT GPUs (set RELIQUARY_GEN_GPU/PROOF_GPU here away from opencode's).
# To avoid killing a co-located opencode miner, this script auto-detects it and
# then SKIPS the broad vLLM/GPU reap (so you may need to clear an orphaned
# openmath vLLM EngineCore by hand, or — recommended — run openmath on its own
# box/device). On a dedicated box it reaps orphaned engines just like opencode.

set -euo pipefail

REPO_DIR="${RELIQUARY_INSTALL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG_FILE="${RELIQUARY_MINER_LOG:-/root/openmath_miner.log}"
PID_FILE="${RELIQUARY_MINER_PID:-/root/openmath_miner.pid}"

: "${BT_WALLET_NAME?set BT_WALLET_NAME (source mining/openmath/.env)}"
: "${BT_HOTKEY?set BT_HOTKEY (source mining/openmath/.env)}"

export RELIQUARY_ENVIRONMENTS=openmathinstruct
export RELIQUARY_OMI_SHARDS="${RELIQUARY_OMI_SHARDS:-4}"
# Separate vLLM venv python (generation worker). Absolute path so the miner can
# spawn it from any cwd.
export RELIQUARY_VLLM_PYTHON="${RELIQUARY_VLLM_PYTHON:-$REPO_DIR/.venv-vllm/bin/python}"

cd "$REPO_DIR"

if [ ! -x "$RELIQUARY_VLLM_PYTHON" ]; then
  echo "WARNING: vLLM venv python not found at $RELIQUARY_VLLM_PYTHON" >&2
  echo "Run mining/setup.sh to build it, or the miner falls back to in-process vLLM" >&2
  echo "(which is NOT validator-compatible with transformers==5.9.0)." >&2
fi

# Always reap our own previous instance.
pkill -9 -f "mining.openmath.miner" 2>/dev/null || true

# The vLLM worker module (mining.common.vllm_worker) and a broad GPU reap are
# SHARED with opencode, so only run them when no opencode miner is present.
if pgrep -f "mining.opencode.miner" >/dev/null 2>&1; then
  echo "NOTE: opencode miner detected — skipping broad vLLM/GPU reap to avoid killing it." >&2
  echo "      Ensure this miner uses a DIFFERENT GPU (RELIQUARY_GEN_GPU/PROOF_GPU) and" >&2
  echo "      clear any orphaned openmath vLLM EngineCore by hand if a restart OOMs." >&2
else
  pkill -9 -f "mining.common.vllm_worker" 2>/dev/null || true
  pkill -9 -f "$REPO_DIR/.venv-vllm" 2>/dev/null || true
  if [ "${RELIQUARY_NO_GPU_REAP:-0}" != "1" ]; then
    for gpid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -9 "$gpid" 2>/dev/null || true
    done
  fi
fi
sleep 2
rm -f "$LOG_FILE" "$PID_FILE"

# Launch the miner via uv in the validator-matched .venv; it spawns the vLLM
# generation worker in .venv-vllm itself.
nohup uv run --no-sync python -m mining.openmath.miner > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Launched OpenMath miner PID=$(cat "$PID_FILE"), log: $LOG_FILE"
