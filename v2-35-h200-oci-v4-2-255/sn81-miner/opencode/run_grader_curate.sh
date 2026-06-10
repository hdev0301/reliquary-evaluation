#!/bin/bash
# OpenCode CURATED-SUBMISSION launcher.
#
# Curates the SUBMISSION (not the pool): the miner generates an oversample, grades its
# OWN completions through the real grader server (exact validator parity), and ships a
# scattered k-correct/(8-k)-wrong 8 => in-zone BY CONSTRUCTION. This is the fix the
# curated POOL couldn't be (pre-screening can't predict the safety-filtered shipped 8).
#
# Checkpoint-robust: grading is re-done online each window, so the pool never goes stale
# on republish (test cases are checkpoint-independent). Single H200: grader server = CPU,
# miner = GPU -> they coexist. Run under tmux/nohup.
set -u
SN81="${SN81:-/root/sn81-miner}"; REPO="${REPO:-/root/reliquary}"
SOCK="${GRADER_SOCKET_PATH:-/tmp/reliquary-grader.sock}"
POOLSZ="${GRADER_POOL_SIZE:-16}"
GLOG="$SN81/logs/grader_server.log"

# 1) ensure the grader server is up (CPU; not killed by run_miner's GPU-proc sweep)
if pgrep -f "reliquary.environment.grader.server" >/dev/null; then
  echo "grader server already running (pid $(pgrep -f 'grader.server' | head -1))"
else
  echo "starting grader server (pool=$POOLSZ, socket=$SOCK) ..."
  rm -f "$SOCK"
  ( cd "$REPO" && set -a && source scripts/.env 2>/dev/null; set +a
    nohup "$REPO/.venv/bin/python" -m reliquary.environment.grader.server \
      --pool-size "$POOLSZ" --socket "$SOCK" >> "$GLOG" 2>&1 & )
  sleep 8
fi

# 2) smoke-test exact-parity grading before trusting it
"$REPO/.venv/bin/python" - "$SOCK" <<'PY'
import sys
from reliquary.environment.grader_client import GraderClient
gc = GraderClient(sys.argv[1])
case = [{"entry":{"kind":"function","name":"add"},"args":[2,3],"kwargs":{},"expected":5,"compare":"exact"}]
ok = gc.evaluate_cases("def add(a,b):\n    return a+b\n", case, 5.0)
bad = gc.evaluate_cases("def add(a,b):\n    return a-b\n", case, 5.0)
print(f"grader smoke: correct={ok} wrong={bad}")
sys.exit(0 if (ok == 1.0 and bad == 0.0) else 1)
PY
[ $? -ne 0 ] && { echo "FATAL: grader smoke test failed — is the server up? see $GLOG"; exit 1; }

# 3) ensure a cached-case pool + cases exist
[ -s "$SN81/data/oci_cases_cache.json" ] || { echo "FATAL: no cases cache ($SN81/data/oci_cases_cache.json) — run build_opencode_pool.py first"; exit 1; }
[ -s "$SN81/data/inzone_pool_opencode.json" ] || { echo "FATAL: no pool ($SN81/data/inzone_pool_opencode.json)"; exit 1; }

# 4) launch the miner in grader-curate mode (run_miner kills only GPU procs + the miner)
echo "launching miner GRADER_CURATE=1 ..."
cd "$REPO" && GRADER_CURATE=1 MODE=opencode bash "$SN81/bin/run_miner.sh" "$@"
