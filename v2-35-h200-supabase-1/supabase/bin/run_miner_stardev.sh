#!/bin/bash
# stardev miner — SUPABASE-BACKED (one-shot). "The miner" is now the split pipeline:
#   producer (GPU pregen -> Supabase `pregen` table, CURATE=0 honest)
#   consumer (GPU-free: pull prepared groups from Supabase -> bind/sign/submit)
# Both run here as separate processes. For supervised running + pool auto-switch use
# autopool_stardev.sh. The old monolithic launcher is saved as
# run_miner_stardev_monolithic.sh.bak.
#
# Usage: run_miner_stardev.sh [POOL_FILENAME]   (producer pool; default inzone_pool.json)
set -euo pipefail
POOL="${1:-inzone_pool.json}"
SP=/root/sn81-miner/supabase_pipeline

echo "[1/2] launching producer (GPU pregen -> Supabase) pool=${POOL} ..."
bash "$SP/run_producer.sh" "$POOL"
sleep 5
echo "[2/2] launching consumer (Supabase -> submit) ..."
bash "$SP/run_consumer.sh"

echo ""
echo "combined Supabase miner up:"
echo "  producer  pid=$(cat /root/sn81-miner/sb_producer.pid 2>/dev/null)  log=logs/sb_producer.log"
echo "  consumer  pid=$(cat /root/sn81-miner/sb_consumer.pid 2>/dev/null)  log=logs/sb_consumer.log"
echo "tail -f /root/sn81-miner/logs/sb_consumer.log   # submitted window=… / verdict ACCEPTED"
