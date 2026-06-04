#!/bin/bash
# GPU box: honest pregen -> Supabase (table=pregen). Genuine CURATE=0 groups with full
# GRAIL artifacts for the stardev hotkey. Does NOT submit (the consumer does).
# Usage: run_producer.sh [POOL_FILENAME]   (default inzone_pool.json; under data/)
set -euo pipefail
cd /root/reliquary || exit 1

POOL="${1:-${STARDEV_POOL:-inzone_pool.json}}"
POOL_PATH="/root/sn81-miner/data/${POOL}"
[ -f "$POOL_PATH" ] || { echo "pool not found: $POOL_PATH" >&2; exit 1; }

# stop only the producer + reap orphaned vLLM EngineCore, then wait for the GPU to free
pkill -9 -f "supabase_pipeline/producer.py" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
for _ in $(seq 1 20); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1); [ "${u:-9999}" -lt 2000 ] 2>/dev/null && break; sleep 1; done

set -a
source /root/reliquary/scripts/.env
source /root/sn81-miner/supabase_pipeline/.env
set +a
export BT_HOTKEY=stardev
export RELIQUARY_CURATE=0                 # HONEST: first-8 natural in-zone groups
export RELIQUARY_OMI_SHARDS=2
export RELIQUARY_GPU_MEM=0.78
export RELIQUARY_GEN_BATCH=40
export RELIQUARY_POOL_SIZE=96
export RELIQUARY_OVERSAMPLE=40
export RELIQUARY_MAX_NEW_TOKENS=1024
export RELIQUARY_PROMPT_IDX_FILE="$POOL_PATH"
export PRODUCER_TARGET_SS58=$(/root/reliquary/.venv/bin/python -c "import json;print(json.load(open('/root/.bittensor/wallets/ronnywebdev/hotkeys/stardevpub.txt'))['ss58Address'])")
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/sn81-miner/logs
rm -f /root/sn81-miner/logs/sb_producer.log
nohup /root/reliquary/.venv/bin/python /root/sn81-miner/supabase_pipeline/producer.py \
  > /root/sn81-miner/logs/sb_producer.log 2>&1 &
echo $! > /root/sn81-miner/sb_producer.pid
echo "producer PID=$(cat /root/sn81-miner/sb_producer.pid) pool=${POOL} | log=/root/sn81-miner/logs/sb_producer.log"
