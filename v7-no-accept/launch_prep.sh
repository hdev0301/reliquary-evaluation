#!/bin/bash
# Spawn N HF prep workers per GPU. Single-engine HF: each worker
# classifies prompts (writes prompt_outcomes) and saves submittable
# rollouts for "good" prompts to pregen_batches in one pass.
#
# Each worker is an independent process pinned to one GPU via
# CUDA_VISIBLE_DEVICES. Multiple workers per GPU time-slice the device
# so one worker's CPU work (Supabase upsert, reward compute, tokenizer
# decode) overlaps with another's GPU work — pushes util from ~40%
# (one worker) to ~90%+ on a 4B model.
#
# Memory per worker on Qwen3-4B bf16: ~9 GiB model + ~5-8 GiB KV cache
# at PREP_PROMPTS_PER_BATCH=2, PREP_MAX_NEW_TOKENS=2048. On a 46 GiB
# L40 you can fit 2 workers; on a 24 GiB Ampere keep it at 1.
#
# Tunables (set in scripts/prep.env or environment):
#   PREP_WORKERS_PER_GPU=2   workers spawned per GPU
#   PREP_PROMPTS_PER_BATCH=2 prompts per HF .generate() call
#   PREP_NUM_PROMPTS=0       0 = run forever
#
# Usage:
#     source scripts/prep.env
#     bash scripts/launch_prep.sh
#
# Logs go to /root/prep_gpu<G>_w<W>.log. Tail all with:
#     tail -F /root/prep_gpu*.log | grep --line-buffered "status="

set -e

INSTALL_DIR="${RELIQUARY_INSTALL_DIR:-/root/reliquary}"
VENV_DIR="${RELIQUARY_VENV:-/root/.venv}"

cd "$INSTALL_DIR"

NUM_PROMPTS="${PREP_NUM_PROMPTS:-0}"
WORKERS_PER_GPU="${PREP_WORKERS_PER_GPU:-1}"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$N_GPUS" -lt 1 ]; then
  echo "ERROR: no GPUs detected." >&2
  exit 1
fi
TOTAL_WORKERS=$((N_GPUS * WORKERS_PER_GPU))
echo "[prep] detected $N_GPUS GPU(s); spawning $WORKERS_PER_GPU worker(s)/GPU = $TOTAL_WORKERS total"

# Reap any previously-running prep workers cleanly. vLLM's python parent
# owns spawned VLLM::EngineCore + torch _inductor compile_worker
# subprocesses; killing only the parent leaves the EngineCores holding
# GPU memory, so the next launch hits "free memory < gpu memory util"
# and fails. Reap all layers explicitly (one pkill per pattern — pkill's
# ERE doesn't accept `\|` as alternation).
for pat in "prep_prompt_outcomes" "prep_pregen_hf" "VLLM::EngineCore" "torch._inductor.compile_worker"; do
  pkill -TERM -f "$pat" 2>/dev/null || true
done
for _ in $(seq 1 10); do
  if ! pgrep -f "prep_prompt_outcomes" >/dev/null 2>&1 \
     && ! pgrep -f "prep_pregen_hf" >/dev/null 2>&1 \
     && ! pgrep -f "VLLM::EngineCore" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
for pat in "prep_prompt_outcomes" "prep_pregen_hf" "VLLM::EngineCore" "torch._inductor.compile_worker"; do
  pkill -9 -f "$pat" 2>/dev/null || true
done
sleep 1

for i in $(seq 0 $((N_GPUS - 1))); do
  for w in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    LOG="/root/prep_gpu${i}_w${w}.log"
    rm -f "$LOG"
    # Each worker is wrapped in a respawn loop so an in-loop ckpt-advance
    # detection (sys.exit(0) inside prep_prompt_outcomes.py) is followed
    # by a fresh process that downloads the new ckpt and reinits HF.
    # The TERM trap lets the next launch_prep.sh invocation kill the
    # wrapper cleanly instead of racing against a fresh python child.
    setsid bash -c '
      trap "exit 0" TERM
      GPU_IDX='"$i"'
      WORKER_IDX='"$w"'
      LOG="/root/prep_gpu${GPU_IDX}_w${WORKER_IDX}.log"
      while true; do
        CUDA_VISIBLE_DEVICES="$GPU_IDX" PATH="'"$VENV_DIR"'/bin:$PATH" \
          "'"$VENV_DIR"'/bin/python" '"$INSTALL_DIR"'/scripts/prep_prompt_outcomes.py \
          --num-prompts "'"$NUM_PROMPTS"'"
        ec=$?
        echo "[respawn] gpu=$GPU_IDX worker=$WORKER_IDX exited code=$ec; restarting in 5s"
        sleep 5
      done
    ' >> "$LOG" 2>&1 &
    echo "[prep] gpu=$i worker=$w wrapper_pid=$! log=$LOG"
  done
done

echo
echo "All $TOTAL_WORKERS workers launched. Tail logs with:"
echo "  tail -F /root/prep_gpu*.log | grep --line-buffered \"status=\""
