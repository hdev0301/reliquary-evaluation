#!/bin/bash
# Supervise the SUPABASE-BACKED stardev miner: producer (GPU pregen->Supabase) +
# consumer (Supabase->submit). Restarts either on crash. If NO accepts occur for
# >= SWITCH_AFTER_WINDOWS (default 15) validator windows, toggles the PRODUCER's
# prompt pool (inzone_pool.json <-> inzone_pool_qwen35.json) and relaunches the
# producer; the consumer keeps running and submits whatever the producer puts in
# Supabase. Honest throughout (producer is CURATE=0).
#
# Window + accept signals are read from the CONSUMER log (it runs MiningEngine.mine_window).
# Usage: nohup bash autopool_stardev.sh >/root/sn81-miner/logs/autopool.log 2>&1 &
# Stop : kill "$(cat /root/sn81-miner/autopool.pid)"; \
#        pkill -f supabase_pipeline/producer.py; pkill -f supabase_pipeline/consumer.py; pkill -f 'VLLM::EngineCore'
set -uo pipefail
cd /root/reliquary || exit 1

DIR=/root/sn81-miner
SP="$DIR/supabase_pipeline"
CLOG="$DIR/logs/sb_consumer.log"
SUP="$DIR/logs/autopool.log"
PPID="$DIR/sb_producer.pid"
CPID="$DIR/sb_consumer.pid"
SELF="$DIR/autopool.pid"
POOLS=("inzone_pool.json" "inzone_pool_qwen35.json")
SWITCH_AFTER_WINDOWS="${SWITCH_AFTER_WINDOWS:-15}"
POLL="${POLL:-20}"

echo $$ > "$SELF"
log(){ echo "$(date -u +%H:%M:%S) | autopool | $*" >> "$SUP"; }
alive(){ local p; p=$(cat "$1" 2>/dev/null || echo ""); [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
start_producer(){ anchor=""; log "producer launch pool=${POOLS[$idx]}"; bash "$SP/run_producer.sh" "${POOLS[$idx]}" >>"$SUP" 2>&1 || log "producer launcher rc!=0"; }
start_consumer(){ log "consumer launch"; bash "$SP/run_consumer.sh" >>"$SUP" 2>&1 || log "consumer launcher rc!=0"; }
cur_window(){ grep -oE 'window advanced to [0-9]+' "$CLOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$'; }
last_accept(){
  { grep -oE 'verdict ACCEPTED win=[0-9]+' "$CLOG" 2>/dev/null | grep -oE '[0-9]+$'
    grep -E 'submitted window=[0-9]+ .*accepted=True' "$CLOG" 2>/dev/null | grep -oE 'window=[0-9]+' | grep -oE '[0-9]+$'
  } | sort -n | tail -1
}

idx="${START_IDX:-0}"
anchor=""

log "supervisor start (combined producer+consumer; switch_after=${SWITCH_AFTER_WINDOWS} windows poll=${POLL}s)"
start_producer
sleep 5
start_consumer

while true; do
  sleep "$POLL"

  alive "$PPID" || { log "producer not running -> relaunch pool=${POOLS[$idx]}"; start_producer; }
  alive "$CPID" || { log "consumer not running -> relaunch"; start_consumer; }

  cw=$(cur_window); [ -z "$cw" ] && continue        # consumer not yet in the window loop
  if [ -z "$anchor" ]; then anchor="$cw"; log "mining from window=$cw pool=${POOLS[$idx]}"; fi

  acc=$(last_accept)
  if [ -n "$acc" ] && [ "$acc" -gt "$anchor" ] 2>/dev/null; then
    anchor="$acc"; log "ACCEPT at window=$acc (streak reset)"
  fi

  streak=$(( cw - anchor ))
  if [ "$streak" -ge "$SWITCH_AFTER_WINDOWS" ]; then
    log "no accepts for ${streak} windows (>= ${SWITCH_AFTER_WINDOWS}) -> SWITCH producer pool"
    idx=$(( (idx + 1) % ${#POOLS[@]} ))
    start_producer    # relaunch producer on the other pool; consumer keeps running
  fi
done
