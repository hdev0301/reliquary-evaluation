#!/bin/bash
# Zero-GPU watcher: the validator is actively patching Qwen3.5 (bad_termination is
# validator-side). Watch origin/main HEAD; when it advances (a new operator fix lands)
# OR a validator file we care about changes, exit so the agent re-assesses + test-mines.
cd /root/reliquary || exit 2
INTERVAL=600          # 10 min, zero GPU
MAX_ITERS=36          # ~6h then return for re-assessment
base=$(git rev-parse origin/main 2>/dev/null)
echo "watch start: origin/main=$base"
for i in $(seq 1 $MAX_ITERS); do
  git fetch origin --quiet 2>/dev/null
  now=$(git rev-parse origin/main 2>/dev/null)
  if [ "$now" != "$base" ]; then
    echo "SIGNAL=main_advanced iter=$i : origin/main $base -> $now"
    echo "--- new commits ---"; git log --oneline "$base..$now"
    echo "--- validator files touched ---"; git diff --name-only "$base" "$now" | grep -E "validator/|constants.py|protocol/" || echo "(none)"
    exit 0
  fi
  # also surface any new termination/qwen branches
  nb=$(git branch -r 2>/dev/null | grep -iE "term|trunc|qwen|eos|stop|p_stop" | tr -d ' ')
  echo "$(date +%H:%M:%S) iter=$i/$MAX_ITERS no main change | branches: $(echo $nb | tr '\n' ' ')"
  sleep $INTERVAL
done
echo "SIGNAL=timeout : ${MAX_ITERS} polls, origin/main unchanged. Re-assess / do a definitive test-mine."
exit 0
