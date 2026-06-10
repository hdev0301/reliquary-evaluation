#!/bin/bash
# Build the OPENCODE DIFFICULTY-SCREENED mining pool — the top-miner pattern.
# -----------------------------------------------------------------------------
# WHY (vs build_frontier_pool.sh's broad-canon pool): at a strong validator
# checkpoint the model solves ~90% of the gradeable universe 8/8 (allcorr), so a
# broad pool is SATURATED -> the miner's screen finds ~0 in-zone prompts -> ~0
# pregen yield -> empty store at window-open -> batch_filled + prompt_in_cooldown.
# Top volume miners instead mine a pool that is DIFFICULTY-SCREENED against the
# CURRENT checkpoint: keep ONLY prompts whose pass-fraction is intermediate (in the
# validator's in-zone band, 2..6 of 8 correct), where BOTH a correct and a wrong
# rollout are abundant -> k=4 (zone-center, sigma=0.5, fat margin over SIGMA_MIN
# 0.43) is reliably assemblable -> real in-zone groups ready every window.
#
# This wraps dataprep/build_opencode_pool.py: (a) routes outputs into opencode/,
# (b) auto-resolves the validator's CURRENT checkpoint to a local snapshot,
# (c) screens the KNOWN-gradeable pool (not random mirror draws), (d) keeps the
# intermediate band [PLOW,PHIGH], (e) ATOMICALLY swaps the active pool in (backup
# first) only if the result is non-trivial.
#
# NEEDS THE GPU — stop the live miner first (the builder loads vLLM at GPU_MEM).
# The local grader server can stay up (build grades in-process by default).
# Rebuild whenever the validator checkpoint advances (the pool is checkpoint-specific).
#
# Usage:
#   bash /root/sn81-miner/opencode/build_pool.sh                      # defaults
#   NCAND=5000 M=16 PLOW=0.20 PHIGH=0.80 bash .../build_pool.sh       # tune
#   IDXFILE=/path/idxs.json bash .../build_pool.sh                    # screen a specific gradeable list
#   REGRADE=opencode/data/oci_gen_cache_seed7.json PLOW=0.25 PHIGH=0.75 bash .../build_pool.sh  # re-band, NO GPU
set -u
SN81="${SN81:-/root/sn81-miner}"
REPO="${REPO:-/root/reliquary}"
VALIDATOR_URL="${VALIDATOR_URL:-http://86.38.238.30:8080}"
OC="$SN81/opencode"
PY="$REPO/.venv/bin/python"

# opencode-only output dirs (build_opencode_pool.py honors these env vars)
export RELIQUARY_DATA_DIR="$OC/data"
export RELIQUARY_DIAG_DIR="$OC/diagnostics"
mkdir -p "$RELIQUARY_DATA_DIR" "$RELIQUARY_DIAG_DIR" "$OC/logs"

NCAND="${NCAND:-9000}"           # #3 bigger pool: how many KNOWN-gradeable prompts to screen (one GPU pass)
M="${M:-16}"; SEED="${SEED:-7}"
PLOW="${PLOW:-0.20}"             # keep prompts with intermediate pass-fraction (validator zone ~[0.25,0.75]);
PHIGH="${PHIGH:-0.80}"          #   [0.20,0.80] -> both correct+wrong abundant at oversample -> k=4 assemblable
MIN_KEEP="${MIN_KEEP:-150}"      # refuse to swap in a degenerate pool
USE_GRADER="${USE_GRADER:-0}"    # 0 = fast local parallel sandbox grading (== validator logic); 1 = serial grader socket
COOLDOWN_FILTER="${COOLDOWN_FILTER:-1}"  # #1: drop already-cooled prompts from the kept pool (cooled_idx.json U /state)
ACTIVE="$RELIQUARY_DATA_DIR/inzone_pool_opencode.json"
TMP_OUT="$RELIQUARY_DATA_DIR/inzone_pool_screened.tmp.json"
COOLED="$RELIQUARY_DATA_DIR/cooled_idx.json"
LOG="$OC/logs/build_opencode.log"

# --- resolve the validator's CURRENT checkpoint to a local snapshot path ---
CKPT="${CHECKPOINT:-}"
if [ -z "$CKPT" ]; then
  STATE="$(curl -s --max-time 15 "$VALIDATOR_URL/state" 2>/dev/null || true)"
  REPO_ID="$(printf '%s' "$STATE" | "$PY" -c "import sys,json;print(json.load(sys.stdin).get('checkpoint_repo_id',''))" 2>/dev/null || true)"
  REV="$(printf '%s' "$STATE" | "$PY" -c "import sys,json;print(json.load(sys.stdin).get('checkpoint_revision',''))" 2>/dev/null || true)"
  CN="$(printf '%s' "$STATE" | "$PY" -c "import sys,json;print(json.load(sys.stdin).get('checkpoint_n','?'))" 2>/dev/null || true)"
  if [ -n "$REPO_ID" ] && [ -n "$REV" ]; then
    CACHE_DIR="/root/.cache/huggingface/hub/models--${REPO_ID//\//--}/snapshots/$REV"
    [ -d "$CACHE_DIR" ] && CKPT="$CACHE_DIR"
    echo "[build_pool] validator checkpoint: $REPO_ID @ ${REV:0:12} (n=$CN)"
  fi
fi
if [ -z "$CKPT" ]; then   # fallback: most-recent cached reliquary-sn-v23 snapshot
  CKPT="$(ls -dt /root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/*/ 2>/dev/null | head -1)"
fi
[ -n "$CKPT" ] && [ -e "$CKPT" ] || { echo "FATAL: could not resolve a checkpoint snapshot (set CHECKPOINT=...)"; exit 1; }

# --- candidate gradeable idx list: subsample the FULL gradeable universe (NOT the
# already-screened active pool — that would be degenerate, re-screening the same set).
# Auto-build the universe from oci_cases_cache.json (every prompt with structured cases). ---
GRADEABLE_SRC="${GRADEABLE_SRC:-$RELIQUARY_DATA_DIR/gradeable_universe.json}"
if [ -z "${IDXFILE:-}" ] && [ -z "${REGRADE:-}" ]; then
  IDXFILE="$RELIQUARY_DATA_DIR/screen_candidates.json"
  "$PY" - "$GRADEABLE_SRC" "$RELIQUARY_DATA_DIR/oci_cases_cache.json" "$IDXFILE" "$NCAND" "$SEED" <<'PYC'
import json, random, sys, os
src, cache_p, out, ncand, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
if not os.path.exists(src):
    cache = json.load(open(cache_p)); uni = []
    for k, v in cache.items():
        try: i = int(k)
        except Exception: continue
        if isinstance(v, dict) and v.get("cases"): uni.append(i)
    uni = sorted(set(uni)); json.dump(uni, open(src, "w"))
    print("[build_pool] built gradeable universe: %d idxs -> %s" % (len(uni), src))
p = json.load(open(src)); random.seed(seed); random.shuffle(p)
sub = sorted(p[:ncand]); json.dump(sub, open(out, "w"))
print("[build_pool] candidates: %d gradeable idxs from %s -> %s" % (len(sub), src, out))
PYC
fi

echo "=== opencode DIFFICULTY-SCREEN build | ckpt=$(basename "$CKPT") | n=$NCAND m=$M band=[$PLOW,$PHIGH] grader=$USE_GRADER ===" | tee "$LOG"
cd "$REPO" || { echo "FATAL: REPO '$REPO' not found"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; source scripts/.env 2>/dev/null; set +a

ARGS=(--checkpoint "$CKPT" --cases-source reconstruct --m "$M" --seed "$SEED" --max-candidates "$NCAND"
      --p-low "$PLOW" --p-high "$PHIGH" --out "$TMP_OUT")
[ -n "${IDXFILE:-}" ] && ARGS+=(--idx-file "$IDXFILE")
[ -n "${REGRADE:-}" ] && ARGS+=(--regrade-from "$REGRADE")     # re-grade persisted completions, skip GPU
[ -n "${CASES_ONLY:-}" ] && ARGS+=(--cases-only)               # reconstruct cases then STOP (no GPU)
[ "$USE_GRADER" = "1" ] && ARGS+=(--use-grader)

"$PY" "$SN81/dataprep/build_opencode_pool.py" "${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG"

# --- #1 COOLDOWN FILTER: drop already-cooled prompts from the screened set ---
# The validator cools a prompt permanently+globally once any miner wins it, but only
# publishes a PARTIAL set in /state. We union /state with idxs we've observed rejected
# PROMPT_IN_COOLDOWN in our own miner logs (which /state omits) + any prior blocklist,
# persist to cooled_idx.json (the running miner also loads this via RELIQUARY_COOLED_IDX_PATH),
# and strip them from the new pool so we never ship a known-dead prompt.
if [ "$COOLDOWN_FILTER" = "1" ] && [ -s "$TMP_OUT" ] && [ -z "${CASES_ONLY:-}" ]; then
  "$PY" - "$TMP_OUT" "$COOLED" "$VALIDATOR_URL" <<'PYF' 2>&1 | tee -a "$LOG"
import sys, json, re, glob, urllib.request
tmp, cooled_p, vurl = sys.argv[1], sys.argv[2], sys.argv[3]
log_idxs=set()
for lp in glob.glob("/root/sn81-miner/logs/miner.log*"):
    try:
        for line in open(lp, errors="ignore"):
            if "prompt_in_cooldown" in line:
                m=re.search(r"prompt=(\d+)", line)
                if m: log_idxs.add(int(m.group(1)))
    except FileNotFoundError: pass
state_idxs=set()
try:
    d=json.load(urllib.request.urlopen(vurl+"/state", timeout=10))
    state_idxs=set(int(x) for x in d.get("cooldown_prompts", []))
except Exception as e:
    print("  #1 cooldown filter: /state fetch err", e)
prior=set()
try: prior=set(int(i) for i in json.load(open(cooled_p)))
except Exception: pass
cooled = log_idxs | state_idxs | prior
json.dump(sorted(cooled), open(cooled_p, "w"))
pool = json.load(open(tmp))
kept = [i for i in pool if i not in cooled]
json.dump(kept, open(tmp, "w"))
print(f"  #1 cooldown filter: pool {len(pool)} -> {len(kept)} (dropped {len(pool)-len(kept)} cooled; blocklist={len(cooled)})")
PYF
fi

# --- validate + ATOMIC swap into the active pool ---
[ -n "${CASES_ONLY:-}" ] && { echo "=== cases-only: no pool written ==="; exit 0; }
if [ ! -s "$TMP_OUT" ]; then echo "FATAL: builder wrote no pool ($TMP_OUT) — active pool UNCHANGED"; exit 1; fi
N=$("$PY" -c "import json;print(len(json.load(open('$TMP_OUT'))))" 2>/dev/null || echo 0)
if [ "$N" -lt "$MIN_KEEP" ]; then
  echo "REFUSING swap: difficulty-screened pool too small ($N < MIN_KEEP=$MIN_KEEP). Kept temp at $TMP_OUT; active pool UNCHANGED."
  echo "  -> widen band (PLOW/PHIGH) or raise NCAND and re-run (regrade is free: REGRADE=opencode/data/oci_gen_cache_seed$SEED.json)."
  exit 2
fi
[ -s "$ACTIVE" ] && cp "$ACTIVE" "$ACTIVE.bak-prescreen-$(date -u +%Y%m%d-%H%M%S)"
mv "$TMP_OUT" "$ACTIVE"
echo "=== SWAPPED IN difficulty-screened pool: $N intermediate idxs -> $ACTIVE (backup kept) | log: $LOG ==="
echo "    next: rm -f $RELIQUARY_DATA_DIR/hot_pool.json ; restart miner to load the new pool (TARGET_K=4)."
