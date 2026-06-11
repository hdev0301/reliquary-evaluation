#!/bin/bash
L=/root/sn81-miner/logs/miner.log
for i in $(seq 1 90); do
  if grep -aqE "verdict (ACCEPTED|REJECTED)" "$L" 2>/dev/null; then
    echo "--- VERDICT ---"; grep -aoE "verdict ACCEPTED win=[0-9]+|verdict REJECTED win=[0-9]+ .*reason=[a-z_]+" "$L" | tail -4
    echo "--- batch (avg_n_correct>0 = numeric pool extracting) ---"; grep -a "pregen batch detail" "$L" | tail -3
    exit 0
  fi
  sleep 12
done
echo "=== status (no verdict yet) ==="; grep -a "screen:" "$L" | tail -2; grep -a "pregen batch detail" "$L" | tail -3
