#!/bin/bash
# Launch the Reliquary pregeneration miner (GSM8K-filtered, cap 1024, oversample 64).
cd /root/reliquary || exit 1
# free GPU + stop any prior miner
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
pkill -9 -f "reliquary.cli.main mine" 2>/dev/null || true
sleep 2
set -a; source scripts/.env; set +a
rm -f /root/miner.log
nohup .venv/bin/python -m reliquary.cli.main mine \
  --network finney --netuid 81 --wallet-name ronnywebdev --hotkey hdev0301 \
  --checkpoint Qwen/Qwen3-4B-Instruct-2507 --validator-url http://86.38.238.30:8080 \
  --gpu-memory-utilization 0.55 --pool-size 64 --gen-batch 16 \
  --max-new-tokens 1536 --no-frontier --oversample 128 \
  --decool-snipe --prompt-sources "augmented_gsm8k,augmented_math" \
  --log-level INFO > /root/miner.log 2>&1 &
echo $! > /root/miner.pid
echo "launched PID=$(cat /root/miner.pid)"
