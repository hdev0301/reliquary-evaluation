#!/bin/bash
# Build the BROAD-FRONTIER opencode pool — divergence-refresh + canon-filter the gradeable universe.
#
# NO GPU, NO model load — pure CPU/network. Safe to run alongside a live miner (it only writes the
# pool file, which the miner re-reads on restart). This REPLACES the GPU scatter-screen
# (build_pool.sh): we no longer pin a checkpoint-specific in-zone list — instead we hand the miner's
# FRONTIER PREDICTOR (feature-based, online SGD, persisted to /root/frontier_model.npz) the whole
# gradeable universe and let it discover/adapt to the CURRENT checkpoint as it mines. That kills the
# "rebuild every checkpoint bump" treadmill.
#
# Three steps:
#   1. DIVERGENCE REFRESH — pull /verdicts/<hotkey>; recover the prompt idxs the validator
#      proof-rejected (out_of_zone / reward_mismatch = our reconstructed cases diverge from the
#      validator's pinned HIDDEN cases). Persist to divergent_idx.json so we stop re-mining them.
#   2. GRADEABLE LIST — candidate universe = every oci_cases_cache.json entry that HAS cases.
#   3. CANON + BURN — drop divergent, keep the lowest-sha256(prompt_idx) KEEP_FRAC (these win the
#      validator's canonical top-8 at seal -> fewer batch_filled). Write inzone_pool_opencode.json.
#
# Steady-state loop:  grow_data.sh  ->  build_frontier_pool.sh  ->  (restart) run_miner.sh
#
# Run:  bash /root/sn81-miner/opencode/build_frontier_pool.sh
#       HOTKEY=hdev0301 KEEP_FRAC=0.4 RESTART=1 bash /root/sn81-miner/opencode/build_frontier_pool.sh
set -u
SN81="${SN81:-/root/sn81-miner}"
REPO="${REPO:-/root/reliquary}"
OC="$SN81/opencode"
OCDATA="$OC/data"
PY="$REPO/.venv/bin/python"
LOG="$OC/logs/build_frontier_pool.log"

HOTKEY="${HOTKEY:-hdev0301}"
WALLET_NAME="${WALLET_NAME:-ronnywebdev}"
VALIDATOR_URL="${VALIDATOR_URL:-http://86.38.238.30:8080}"
KEEP_FRAC="${KEEP_FRAC:-0.7}"          # was 0.4: too-tight canon set = everyone targets same prompts = prompt_in_cooldown. 0.7 widens the live surface (validator only sha256-tiebreaks on overflow).
MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-1200}"  # only mine short-input prompts (top miner uid 165: prompt chars p95=max=1201) -> shorter sequence, cheaper verify, more seal wins
RESTART="${RESTART:-0}"                # 1 = restart the miner at the end to APPLY the new pool
# HYBRID overlay: a dense, GPU-screened idx list (build_pool.sh output, known-scatter on the
# current ckpt). If present it is UNIONed onto the broad base and placed FIRST (the frontier/
# sampler front-loads it) so we get screen-grade DENSITY without ever shrinking below the broad
# base (= never re-starves when the dense set goes stale). Empty/missing -> pure broad base.
OVERLAY="${OVERLAY:-}"

mkdir -p "$OC/logs"
echo "=== build_frontier_pool $(date -u +%H:%M:%S) | hotkey=$HOTKEY keep_frac=$KEEP_FRAC ===" | tee -a "$LOG"

HK_SS58=$("$PY" -c "import json,os;print(json.load(open(os.path.expanduser('~/.bittensor/wallets/$WALLET_NAME/hotkeys/$HOTKEY')))['ss58Address'])" 2>/dev/null)
# NON-FATAL: without ss58 we skip the /verdicts divergence refresh but STILL build the pool, so a
# wallet/auth hiccup never strands the miner with no pool (this runs as the automation's apply step).
[ -n "$HK_SS58" ] || echo "WARN: could not resolve ss58 (wallet=$WALLET_NAME hotkey=$HOTKEY) -> skipping divergence refresh; pool still builds" | tee -a "$LOG"

"$PY" - "$HK_SS58" "$OCDATA" "$VALIDATOR_URL" "$KEEP_FRAC" "$MAX_PROMPT_CHARS" "$OVERLAY" <<'PY' 2>&1 | tee -a "$LOG"
import sys, json, hashlib, os, urllib.request, shutil, statistics
ss58, OCDATA, VURL, keep = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
maxchars = int(sys.argv[5])
overlay_p = sys.argv[6] if len(sys.argv) > 6 else ""
cache_p = f"{OCDATA}/oci_cases_cache.json"
div_p   = f"{OCDATA}/divergent_idx.json"
cand_p  = f"{OCDATA}/oci_cached_candidates_v2.json"
pool_p  = f"{OCDATA}/inzone_pool_opencode.json"
MIRROR_N = 60000   # opencode idxs are positions in the ~50k prompt mirror; scan a bit past it

def load(p, d):
    try: return json.load(open(p))
    except Exception: return d
def ckey(i): return hashlib.sha256(int(i).to_bytes(8, "big", signed=False)).digest()

# --- 2. gradeable universe (compute first so we can validate divergent against it) ---
cache = load(cache_p, {})
gradeable = sorted(int(k) for k, v in cache.items()
                   if v and v.get("cases") and len(v.get("prompt", "")) <= maxchars)
gset = set(gradeable)
print(f"  [gradeable] {len(gradeable)} prompts with cases AND prompt<={maxchars} chars (cache universe={len(cache)})")
if not gradeable:
    print("  FATAL: no gradeable candidates — run grow_data.sh first"); sys.exit(1)
json.dump(gradeable, open(cand_p, "w"))

# --- 1. divergence refresh (validator proof-rejects = case parity failures) ---
divergent = set(int(i) for i in load(div_p, []))
n0 = len(divergent)
try:
    V = json.load(urllib.request.urlopen(f"{VURL}/verdicts/{ss58}", timeout=20))["verdicts"]
    leads = {v["prompt_hash_lead"] for v in V
             if v.get("reject_reason") in ("out_of_zone", "reward_mismatch") and v.get("prompt_hash_lead")}
    need = set(leads); found = {}
    for i in range(MIRROR_N):
        if not need: break
        l = hashlib.sha256(i.to_bytes(8, "big", signed=False)).hexdigest()[:12]
        if l in need:
            found[l] = i; need.discard(l)
    # only burn idxs that are actually gradeable opencode candidates (guards against hash
    # collision with stale math verdicts on the same shared hotkey)
    new = {i for i in found.values() if i in gset}
    divergent |= new
    print(f"  [diverge] {len(V)} verdicts, {len(leads)} divergent leads -> +{len(divergent)-n0} new gradeable (total {len(divergent)})")
except Exception as e:
    print(f"  [diverge] WARN: /verdicts fetch failed ({e}); keeping existing {n0} divergent")
json.dump(sorted(divergent), open(div_p, "w"))

# --- 3. canon-filter (lowest-sha256) minus divergent ---
clean = [i for i in gradeable if i not in divergent]
ordered = sorted(clean, key=ckey)
pool = sorted(int(i) for i in ordered[:max(1, round(len(ordered) * keep))])

# --- 3b. HYBRID: union the dense GPU-screened overlay on top, priority-first ---
# Overlay idxs are known to SCATTER on the (recent) checkpoint. Keep only gradeable, non-divergent
# ones; place them FIRST so the frontier/sampler front-loads proven scatter, then the canon base.
# This gives screen-grade density while the broad base guarantees we never starve when it goes stale.
ov = []
if overlay_p and os.path.exists(overlay_p):
    ov = [int(i) for i in load(overlay_p, []) if int(i) in gset and int(i) not in divergent]
    new_beyond_canon = len([i for i in ov if i not in set(pool)])
    ovset = set(ov)
    pool = ov + [i for i in pool if i not in ovset]
    print(f"  [overlay] {overlay_p}: {len(ov)} dense scatter idxs unioned first "
          f"(+{new_beyond_canon} beyond the canon cut); pool now {len(pool)}")

if os.path.exists(pool_p):
    shutil.copy(pool_p, pool_p + ".bak")
json.dump(pool, open(pool_p, "w"))
cm = statistics.mean(int.from_bytes(ckey(i)[:4], "big") for i in pool)
am = statistics.mean(int.from_bytes(ckey(i)[:4], "big") for i in clean)
print(f"  [canon] gradeable-divergent={len(clean)}, keep_frac={keep} -> {len(pool)} idxs "
      f"(sha-prefix mean {cm:.2e} < {am:.2e}, lower = wins more seal ties)")
print(f"  WROTE {pool_p}: {len(pool)} idxs")
PY
rc=${PIPESTATUS[0]}
[ "$rc" = "0" ] || { echo "FATAL: pool build failed (rc=$rc)" | tee -a "$LOG"; exit "$rc"; }

echo "=== build_frontier_pool DONE $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
if [ "$RESTART" = "1" ]; then
  echo "[bfp] restarting miner to apply the new pool ..." | tee -a "$LOG"
  HOTKEY="$HOTKEY" setsid bash "$OC/run_miner.sh" < /dev/null >> "$LOG" 2>&1 &
  echo "[bfp] miner restart launched (HOTKEY=$HOTKEY)." | tee -a "$LOG"
else
  echo "[bfp] restart the miner to USE the new pool:  HOTKEY=$HOTKEY bash $OC/run_miner.sh" | tee -a "$LOG"
fi
