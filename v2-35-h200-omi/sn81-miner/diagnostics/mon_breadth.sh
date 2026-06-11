#!/bin/bash
# Track BREADTH (prompts screened/min) + keepers under the new oversample-160 config.
L=/root/miner.log
START=$(date +%s)
while true; do
  now=$(date +%s); el=$(( (now-START)/60 ))
  scr=$(grep -ac "screen:" $L 2>/dev/null)
  prom=$(grep -a "screen:" $L 2>/dev/null | sed -E 's/.*screen: ([0-9]+)\/.*/\1/' | awk '{s+=$1} END{print s+0}')
  batch=$(grep -ac "pregen batch detail" $L 2>/dev/null)
  kept=$(grep -a "pregen batch detail" $L 2>/dev/null | sed -E 's/.*kept=([0-9]+).*/\1/' | awk '{s+=$1} END{print s+0}')
  notcur=$(grep -a "pregen batch detail" $L 2>/dev/null | sed -E 's/.*not_curatable=([0-9]+).*/\1/' | awk '{s+=$1} END{print s+0}')
  acc=$(grep -aci "accepted" $L 2>/dev/null)
  hot=$(grep -a "hot=" $L 2>/dev/null | tail -1 | sed -E 's/.*hot=([0-9]+).*/\1/')
  echo "t=${el}m screen_cycles=$scr promising_total=$prom deep_batches=$batch kept=$kept not_curatable=$notcur accepted=$acc hot=${hot:-0}"
  sleep 240
done
