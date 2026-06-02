#!/bin/bash
# Launch a Reliquary miner in the background. Source scripts/.env first.
#
# Usage:
#     source scripts/.env
#     bash scripts/launch_miner.sh

set -e

INSTALL_DIR="${RELIQUARY_INSTALL_DIR:-/root/reliquary}"
LOG_FILE="${RELIQUARY_MINER_LOG:-/root/miner.log}"
PID_FILE="${RELIQUARY_MINER_PID:-/root/miner.pid}"
# Validator runs torch 2.11.0+cu130 / transformers 5.8.0 in /root/.venv;
# match it exactly. Override with RELIQUARY_VENV if you maintain a separate one.
VENV_DIR="${RELIQUARY_VENV:-/root/.venv}"

: "${BT_WALLET_NAME?BT_WALLET_NAME not set; source scripts/.env}"
: "${BT_HOTKEY?BT_HOTKEY not set; source scripts/.env}"

cd "$INSTALL_DIR"

find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# Graceful shutdown first. SIGKILL on a CUDA-using process leaves the GPU
# in an EPERM-on-open state (NVML "Unknown Error") that recovers only via
# host-level driver reload or instance reboot — neither available from
# inside the container. Send SIGTERM, give the asyncio loop ~10 s to
# tear down model handles cleanly, then SIGKILL only if it's still alive.
#
# Also kill any orphaned vLLM EngineCore workers — they're spawned as
# subprocesses with their own process name ("VLLM::EngineCore") and
# survive a TERM/KILL on the parent if the parent died abnormally
# (e.g. during a failed ckpt-reload). A leftover EngineCore holds
# ~50 GB of VRAM and starves the next vLLM init.
if pgrep -f "reliquary.*main.*mine" >/dev/null 2>&1; then
  pkill -TERM -f "reliquary.*main.*mine" 2>/dev/null || true
  for _ in $(seq 1 20); do
    pgrep -f "reliquary.*main.*mine" >/dev/null 2>&1 || break
    sleep 0.5
  done
  pkill -9 -f "reliquary.*main.*mine" 2>/dev/null || true
  sleep 1
fi
# Sweep any orphaned vLLM workers (no graceful TERM — the parent's
# already gone, no asyncio loop to drain).
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 1
rm -f "$LOG_FILE" "$PID_FILE"

drand_flag="--use-drand"
if [ "${RELIQUARY_USE_DRAND:-1}" != "1" ]; then
  drand_flag="--no-use-drand"
fi

validator_url_arg=""
if [ -n "${RELIQUARY_VALIDATOR_URL:-}" ]; then
  validator_url_arg="--validator-url ${RELIQUARY_VALIDATOR_URL}"
fi

nohup "$VENV_DIR/bin/python" -m reliquary.cli.main mine \
    --network "$BT_NETWORK" \
    --netuid "$NETUID" \
    --wallet-name "$BT_WALLET_NAME" \
    --hotkey "$BT_HOTKEY" \
    --checkpoint "${RELIQUARY_CHECKPOINT:-gpt2}" \
    --log-level INFO \
    $drand_flag $validator_url_arg \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Launched miner PID=$(cat "$PID_FILE"), log: $LOG_FILE"
