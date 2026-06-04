#!/bin/bash
# Submit box (GPU-FREE): pulls honest prepared groups from Supabase and submits them
# to the validator. Best run on a low-latency box near the validator to win the slot race.
set -euo pipefail
cd /root/reliquary || exit 1
pkill -9 -f "supabase_pipeline/consumer.py" 2>/dev/null || true
sleep 1
set -a
source /root/reliquary/scripts/.env
source /root/sn81-miner/supabase_pipeline/.env
set +a
export BT_HOTKEY=stardev
: "${RELIQUARY_VALIDATOR_URL:?set RELIQUARY_VALIDATOR_URL (consumer needs an explicit validator)}"
mkdir -p /root/sn81-miner/logs
nohup /root/reliquary/.venv/bin/python /root/sn81-miner/supabase_pipeline/consumer.py \
  > /root/sn81-miner/logs/sb_consumer.log 2>&1 &
echo $! > /root/sn81-miner/sb_consumer.pid
echo "consumer PID=$(cat /root/sn81-miner/sb_consumer.pid) | log=/root/sn81-miner/logs/sb_consumer.log"
