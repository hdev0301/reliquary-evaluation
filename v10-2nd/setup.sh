#!/usr/bin/env bash
#
# setup.sh — Take a fresh GPU box to a runnable Reliquary SN81 miner, reproducing
#            the EXACT pinned environment from the working H200 box so the same
#            build also runs on the Blackwell RTX PRO 6000 (sm_120 / CUDA 13).
#
# This script does the SETUP half only. It does NOT launch the miner. The launch
# half is /root/run_miner.sh (curation pipeline), left untouched as the entry point.
#
# Ground-truth pins (verified on the working H200 box, torch arch_list incl. sm_120):
#   torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0  (cu130 index)
#   vllm==0.20.2  flash_attn==2.8.1 (cu13 direct wheel)
#   flashinfer-python==0.6.8.post1  flashinfer-cubin==0.6.8.post1
#   bittensor==10.2.1 bittensor-drand==1.3.0 bittensor-wallet==4.0.1
#   async-substrate-interface==1.6.4  scalecodec==1.2.12  (mandatory reconciliation)
#   transformers==5.9.0 tokenizers==0.22.2 datasets==4.8.5 safetensors==0.7.0
#   reliquary 0.1.0 installed EDITABLE (pip install -e .)
#
# Usage:
#   bash /root/setup.sh                 # idempotent; skips already-done steps
#   bash /root/setup.sh --force         # clobber existing venv / .env / data files
#   FORCE=1 bash /root/setup.sh         # same as --force
#   RELIQUARY_INSTALL_DIR=/root/reliquary PYTHON=python3.12 \
#     RELIQUARY_REPO_URL=https://github.com/reliquadotai/reliquary.git bash /root/setup.sh
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Parameters (override via environment)
# --------------------------------------------------------------------------- #
INSTALL_DIR="${RELIQUARY_INSTALL_DIR:-/root/reliquary}"
PYTHON="${PYTHON:-python3.12}"
# Fork that carries the curation patches (pregen.py) used by run_miner.sh.
# The working box's repo origin is reliquadotai/reliquary; if you mine off a
# private fork with the curation patches, override RELIQUARY_REPO_URL.
REPO_URL="${RELIQUARY_REPO_URL:-https://github.com/reliquadotai/reliquary.git}"   # <-- set to YOUR fork if it carries the curation patches
BRANCH="${RELIQUARY_BRANCH:-main}"

VENV="$INSTALL_DIR/.venv"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

# Data-file directory (runtime artifacts live at /root, NOT in the repo)
DATA_DIR="${RELIQUARY_DATA_DIR:-/root}"
INZONE_POOL="$DATA_DIR/inzone_pool.json"
HOT_POOL="$DATA_DIR/hot_pool.json"
SUBMITTED_IDX="$DATA_DIR/submitted_idx.json"

# Pinned versions / sources (verbatim from ground truth)
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
TORCH_VER="2.11.0"
TV_VER="0.26.0"
TA_VER="2.11.0"
VLLM_VER="0.20.2"
FLASHINFER_VER="0.6.8.post1"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
BITTENSOR_VER="10.2.1"
BT_DRAND_VER="1.3.0"
BT_WALLET_VER="4.0.1"
ASI_VER="1.6.4"           # async-substrate-interface
SCALECODEC_VER="1.2.12"
TRANSFORMERS_VER="5.9.0"
TOKENIZERS_VER="0.22.2"
DATASETS_VER="4.8.5"
SAFETENSORS_VER="0.7.0"

# HF predownloads (all PUBLIC, ungated)
MINER_MODEL="R0mAI/reliquary-sn-v23"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

# Blackwell / CUDA 13 driver floor
DRIVER_MIN_MAJOR=580
MIN_DISK_GB=60

# --force handling
FORCE="${FORCE:-0}"
MISSING_CRITICAL=0   # set if copy-from-source artifacts (inzone_pool / curation-patched repo) are absent
NOT_READY=0          # set if the GPU/driver cannot run the cu130/sm_120 stack
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "WARN: unknown arg '$arg' (ignored)" >&2 ;;
  esac
done

banner() { printf '\n\033[1;36m==== %s ====\033[0m\n' "$*"; }
ok()     { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn()   { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()    { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
banner "(1) PREFLIGHT CHECKS"
# --------------------------------------------------------------------------- #
[ "$(id -u)" = "0" ] || warn "not running as root — apt + /root writes may fail."

# OS
if [ -r /etc/os-release ]; then
  . /etc/os-release
  echo "OS: ${PRETTY_NAME:-unknown}"
  case "${ID:-}" in
    ubuntu|debian) : ;;
    *) warn "expected Ubuntu/Debian; apt steps may not apply on '${ID:-?}'." ;;
  esac
else
  warn "/etc/os-release missing — cannot identify OS."
fi

# Python presence (need 3.12.x to match the cp312 wheels)
if command -v "$PYTHON" >/dev/null 2>&1; then
  PYV="$("$PYTHON" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"
  echo "Python: $PYTHON -> $PYV"
  case "$PYV" in
    3.12.*) ok "python 3.12 present (matches cp312 wheels)" ;;
    *) warn "the pinned wheels are cp312 (flash_attn) — '$PYTHON' is $PYV. Install python3.12 or set PYTHON=python3.12." ;;
  esac
else
  warn "$PYTHON not found yet — will be installed in step (2)."
fi

# nvidia-smi + driver version
if command -v nvidia-smi >/dev/null 2>&1; then
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "GPU: ${GPU:-unknown}   Driver: ${DRV:-unknown}"
  if [ -n "${DRV:-}" ]; then
    DRV_MAJOR="${DRV%%.*}"
    if [ "$DRV_MAJOR" -ge "$DRIVER_MIN_MAJOR" ] 2>/dev/null; then
      ok "driver $DRV >= ${DRIVER_MIN_MAJOR} (CUDA 13 / sm_120 capable)"
    else
      warn "driver $DRV is below the CUDA 13 floor (>= ${DRIVER_MIN_MAJOR}). torch 2.11.0+cu130 will FAIL to init CUDA on the Blackwell RTX PRO 6000. Upgrade the NVIDIA driver to 580+ (working H200 box is on 595.45.04) before mining."
      NOT_READY=1
    fi
  else
    warn "could not read driver version from nvidia-smi."
  fi
else
  warn "nvidia-smi not found — no NVIDIA driver detected. A 580+ driver is REQUIRED for the cu130 stack on Blackwell."
fi

# Disk space at the install target
AVAIL_GB="$(df -PBG "${INSTALL_DIR%/*}" 2>/dev/null | awk 'NR==2{gsub("G","",$4);print $4}')"
if [ -n "${AVAIL_GB:-}" ]; then
  echo "Free disk at $(dirname "$INSTALL_DIR"): ${AVAIL_GB}G"
  if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ] 2>/dev/null; then
    warn "only ${AVAIL_GB}G free; the venv (~15G) + 2 HF models (~16G) + datasets need >= ${MIN_DISK_GB}G."
  else
    ok "disk space OK (>= ${MIN_DISK_GB}G)"
  fi
fi

[ "$FORCE" = "1" ] && warn "--force enabled: existing venv / .env / data files MAY be overwritten."

# --------------------------------------------------------------------------- #
banner "(2) APT SYSTEM PACKAGES"
# --------------------------------------------------------------------------- #
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  echo "apt-get update + install python3.12 toolchain, git, build-essential, wget/curl ..."
  apt-get update -qq
  apt-get install -y -qq \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git build-essential wget curl ca-certificates
  ok "system packages installed"
  command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON still not found after apt install. Aborting."
else
  warn "apt-get not available — skipping system package install. Ensure $PYTHON, git, build-essential, wget are present."
fi

# --------------------------------------------------------------------------- #
banner "(3) CLONE / LOCATE REPO"
# --------------------------------------------------------------------------- #
if [ -d "$INSTALL_DIR/.git" ] || [ -f "$INSTALL_DIR/pyproject.toml" ]; then
  ok "repo already present at $INSTALL_DIR (not re-cloning)"
  if [ -d "$INSTALL_DIR/.git" ]; then
    echo "    remote: $(git -C "$INSTALL_DIR" remote get-url origin 2>/dev/null || echo '?')"
    echo "    (run 'git -C $INSTALL_DIR pull' manually if you want the latest; not auto-pulled to avoid clobbering local curation patches)"
  fi
else
  case "$REPO_URL" in
    *"<"*|*"YOUR"*|"") die "RELIQUARY_REPO_URL is not set to a real repo. Edit setup.sh REPO_URL or export RELIQUARY_REPO_URL=<your fork with curation patches>." ;;
  esac
  echo "git clone $REPO_URL ($BRANCH) -> $INSTALL_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR" \
    || die "git clone failed. Check RELIQUARY_REPO_URL / network."
  ok "cloned repo"
fi
[ -f "$INSTALL_DIR/pyproject.toml" ] || die "no pyproject.toml at $INSTALL_DIR — wrong repo or partial clone."

# --------------------------------------------------------------------------- #
banner "(4) CREATE VENV"
# --------------------------------------------------------------------------- #
if [ -d "$VENV" ]; then
  if [ "$FORCE" = "1" ]; then
    warn "--force: removing existing venv $VENV"
    rm -rf "$VENV"
    "$PYTHON" -m venv "$VENV" || die "venv creation failed."
    ok "recreated venv"
  else
    ok "venv exists at $VENV (reusing; use --force to recreate)"
  fi
else
  "$PYTHON" -m venv "$VENV" || die "venv creation failed (is python3.12-venv installed?)."
  ok "created venv at $VENV"
fi
[ -x "$PIP" ] || die "pip not found in venv ($PIP)."

echo "Upgrading pip / setuptools / wheel ..."
"$PIP" install --upgrade pip setuptools wheel
ok "installer tooling upgraded"

# --------------------------------------------------------------------------- #
banner "(5) PIP INSTALL — EXACT ORDERED SEQUENCE"
# --------------------------------------------------------------------------- #
# Order is LOAD-BEARING:
#   torch(cu130) -> flash_attn(cu13 wheel) -> flashinfer -> vllm -> bittensor
#   -> scalecodec reconcile -> transformers/datasets -> pip install -e .  (LAST)
# Never run a later bare `pip install torch` or unpinned torch-needing pkg after
# this block — it can swap the cu130 build for the cu12 PyPI default.

echo "[5.1] torch ${TORCH_VER} / torchvision ${TV_VER} / torchaudio ${TA_VER}  (cu130 index)"
echo "      arch_list of this build includes sm_120 -> runs on Blackwell RTX PRO 6000."
"$PIP" install \
  "torch==${TORCH_VER}" "torchvision==${TV_VER}" "torchaudio==${TA_VER}" \
  --index-url "$TORCH_INDEX" \
  || die "torch cu130 install failed (check the cu130 index is reachable)."
ok "torch cu130 trio installed"

echo "[5.2] flash_attn 2.8.1  (direct cu13/cp312/cxx11abiTRUE wheel — do NOT rename)"
# The wheel filename carries the version/platform tags pip parses (%2B == '+').
# wget -P /tmp without renaming, then install the local file (Dockerfile idiom).
FA_TMP="/tmp/$(basename "${FLASH_ATTN_URL//%2B/+}")"
if [ ! -f "$FA_TMP" ]; then
  wget -q --show-progress -O "$FA_TMP" "$FLASH_ATTN_URL" \
    || die "flash_attn wheel download failed: $FLASH_ATTN_URL"
fi
"$PIP" install "$FA_TMP" \
  || die "flash_attn install failed (ABI mismatch? torch must be ${TORCH_VER}+cu130 first)."
ok "flash_attn 2.8.1 installed"

echo "[5.3] flashinfer-python / flashinfer-cubin ${FLASHINFER_VER}  (explicit pin)"
"$PIP" install \
  "flashinfer-python==${FLASHINFER_VER}" "flashinfer-cubin==${FLASHINFER_VER}" \
  || die "flashinfer install failed."
ok "flashinfer pinned"

echo "[5.4] vllm ${VLLM_VER}  (hard-pins torch ${TORCH_VER} — already satisfied, no torch swap)"
"$PIP" install "vllm==${VLLM_VER}" \
  || die "vllm install failed."
ok "vllm installed"

echo "[5.5] bittensor ${BITTENSOR_VER} + bittensor-drand ${BT_DRAND_VER} + bittensor-wallet ${BT_WALLET_VER}"
"$PIP" install \
  "bittensor==${BITTENSOR_VER}" "bittensor-drand==${BT_DRAND_VER}" "bittensor-wallet==${BT_WALLET_VER}" \
  || die "bittensor install failed."
ok "bittensor stack installed"

echo "[5.6] bittensor reconciliation: async-substrate-interface==${ASI_VER} + force scalecodec==${SCALECODEC_VER}"
# bittensor 10.2.1 ALREADY pins async-substrate-interface<2.0.0 and scalecodec==1.2.12
# correctly; these are defensive pin-locks (belt-and-suspenders for future patch
# releases), NOT a fix for an active 2.x pull. Plain installs (no --no-deps, which
# could strip scalecodec's own transitive deps).
"$PIP" install "async-substrate-interface==${ASI_VER}" \
  || die "async-substrate-interface pin failed."
"$PIP" install "scalecodec==${SCALECODEC_VER}" \
  || die "scalecodec pin failed."
ok "bittensor scalecodec reconciliation applied"

echo "[5.7] transformers ${TRANSFORMERS_VER} / tokenizers ${TOKENIZERS_VER} / datasets ${DATASETS_VER} / safetensors ${SAFETENSORS_VER}"
"$PIP" install \
  "transformers==${TRANSFORMERS_VER}" "tokenizers==${TOKENIZERS_VER}" \
  "datasets==${DATASETS_VER}" "safetensors==${SAFETENSORS_VER}" \
  || die "transformers/datasets stack install failed."
ok "HF model/data stack installed"

echo "[5.8] reliquary 0.1.0 EDITABLE (pip install -e .)  — LAST so small deps resolve against the pinned heavy stack"
"$PIP" install -e "$INSTALL_DIR" \
  || die "editable reliquary install failed."
# boto3/botocore arrive via `pip install -e .`; pin to the proven versions so a
# later-dated fresh box can't pull a botocore that violates aiobotocore's ceiling.
# (wandb is NOT installed on the working miner box — omitted.)
"$PIP" install "boto3==1.43.0" "botocore==1.43.0" >/dev/null 2>&1 || warn "boto3/botocore pin non-fatal failure (only needed for R2 upload)."
"$PIP" check || warn "pip reports dependency conflicts (review the lines above)."
ok "reliquary installed editable"

# --------------------------------------------------------------------------- #
banner "(6) VERIFY GPU / sm_120 / PIN INTEGRITY"
# --------------------------------------------------------------------------- #
"$PY" - <<'PYEOF' || die "post-install verification FAILED — a pin drifted or CUDA is not visible. Do not launch the miner."
import torch
al = torch.cuda.get_arch_list()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch_list:", al)
assert "+cu130" in torch.__version__, f"torch is not the cu130 build: {torch.__version__}"
assert "sm_120" in al, f"arch_list missing sm_120 (Blackwell not supported by this torch build): {al}"
import vllm, flash_attn, flashinfer, transformers
import importlib.metadata as _md
_sc = _md.version("scalecodec")   # scalecodec exposes NO module __version__; read pip metadata
print("vllm", vllm.__version__, "flash_attn", flash_attn.__version__,
      "flashinfer", getattr(flashinfer, "__version__", "?"),
      "transformers", transformers.__version__, "scalecodec", _sc)
assert _sc == "1.2.12", f"scalecodec drifted: {_sc}"
import bittensor  # import after scalecodec to confirm the reconciliation held
print("bittensor", bittensor.__version__)
if torch.cuda.is_available():
    print("CUDA OK ->", torch.cuda.get_device_name(0))
else:
    print("WARNING: torch.cuda.is_available()==False (driver/CUDA not initialized on this box yet)")
PYEOF
ok "verification passed: cu130 torch, sm_120 present, vllm/flash_attn/scalecodec/bittensor import clean"
if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  ok "GPU visible to torch"
else
  warn "torch cannot see a CUDA device yet. Imports are fine, but the GPU/driver is not ready — fix the driver (>= ${DRIVER_MIN_MAJOR}) before mining."
  NOT_READY=1
fi

# --------------------------------------------------------------------------- #
banner "(7) PREDOWNLOAD HF MODELS (public, ungated)"
# --------------------------------------------------------------------------- #
# Predownload to ~/.cache/huggingface/hub to avoid a cold-start stall. HF_TOKEN
# NOT required (both repos are public). NOTE: the validator pins the miner model
# to a specific revision; a blanket 'main' pull may not match the published
# revision the day you run — the miner re-fetches the correct revision at runtime.
[ -f "$INSTALL_DIR/scripts/.env" ] && { set -a; . "$INSTALL_DIR/scripts/.env"; set +a; } || true
for REPO in "$BASE_MODEL" "$MINER_MODEL"; do
  echo "huggingface download: $REPO (allow_patterns: model*.safetensors, config.json, tokenizer*, *.json)"
  "$PY" - "$REPO" <<'PYEOF' || warn "predownload of model failed (network/HF). Not fatal — the miner will fetch at runtime."
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
p = snapshot_download(
    repo_id=repo,
    allow_patterns=["*.safetensors", "config.json", "tokenizer*", "*.json"],
)
print("  cached ->", p)
PYEOF
done
ok "model predownload attempted (BASE + miner checkpoint)"
warn "Dataset nvidia/OpenMathInstruct-2 (RELIQUARY_OMI_SHARDS=2 shards) is NOT predownloaded here; the miner loads it on first run. Keep RELIQUARY_OMI_SHARDS=2 — inzone_pool.json idxs index that fixed 2-shard ordering."

# --------------------------------------------------------------------------- #
banner "(8) RUNTIME DATA FILES"
# --------------------------------------------------------------------------- #
mkdir -p "$DATA_DIR"

# hot_pool.json + submitted_idx.json: safe to start empty (self-fill / no history)
for f in "$HOT_POOL" "$SUBMITTED_IDX"; do
  if [ -f "$f" ] && [ "$FORCE" != "1" ]; then
    ok "$(basename "$f") exists (kept; use --force to reset to [])"
  else
    [ -f "$f" ] && warn "--force: resetting $(basename "$f") to []"
    printf '[]' > "$f"
    ok "$(basename "$f") initialized to []"
  fi
done

# inzone_pool.json: the load-bearing 179,707-idx curation pool. CANNOT be cleanly
# regenerated on a bare box (format_analysis.py needs harvested caches and yields
# a DIFFERENT/smaller pool). Must be COPIED from the working box. Do not fake it.
if [ -f "$INZONE_POOL" ]; then
  N="$("$PY" -c "import json;print(len(json.load(open('$INZONE_POOL'))))" 2>/dev/null || echo '?')"
  ok "inzone_pool.json present ($N idxs)"
  [ "$N" = "179707" ] || warn "inzone_pool.json has $N idxs (working box has 179707). If this is a regenerated/partial pool, the miner will under-perform."
else
  warn "================================================================"
  warn "MISSING: $INZONE_POOL"
  warn "This is the load-bearing curation pool (179,707 OpenMathInstruct-2"
  warn "prompt idxs) that run_miner.sh mines via --prompt-idx-file."
  warn "It is a RUNTIME ARTIFACT, not in the repo and not pip-installable,"
  warn "and CANNOT be faithfully regenerated on a bare box (format_analysis.py"
  warn "needs /root/topminer_vectors.jsonl + /root/topminer_accepts.jsonl and"
  warn "produces a DIFFERENT pool). YOU MUST COPY IT from the source box:"
  warn "    scp SRC:/root/inzone_pool.json $INZONE_POOL"
  warn "The miner will NOT work correctly without it."
  warn "================================================================"
  MISSING_CRITICAL=1
fi
# The curation patches + launcher are copy-from-source too — flag if absent.
grep -q "RELIQUARY_CURATE" "$INSTALL_DIR/reliquary/miner/pregen.py" 2>/dev/null || { warn "curation patch missing in pregen.py — copy the modified repo from source."; MISSING_CRITICAL=1; }
[ -f /root/run_miner.sh ] || { warn "/root/run_miner.sh missing — copy the launcher from source."; MISSING_CRITICAL=1; }

# --------------------------------------------------------------------------- #
banner "(9) scripts/.env"
# --------------------------------------------------------------------------- #
ENV_FILE="$INSTALL_DIR/scripts/.env"
ENV_EXAMPLE="$INSTALL_DIR/scripts/.env.example"
if [ -f "$ENV_FILE" ] && [ "$FORCE" != "1" ]; then
  ok "scripts/.env exists (kept; use --force to overwrite from .env.example)"
else
  [ -f "$ENV_FILE" ] && warn "--force: overwriting existing scripts/.env"
  if [ -f "$ENV_EXAMPLE" ]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    # SCRUB secrets committed in the template (.env.example ships a real-looking HF
    # token). Blank them so a new box never silently inherits someone else's token.
    sed -i -E 's/^(export (HF_TOKEN|HUGGING_FACE_HUB_TOKEN|R2_ACCESS_KEY_ID|R2_SECRET_ACCESS_KEY|GRAIL_STATE_HMAC_KEY))=.*/\1=/' "$ENV_FILE"
  else
    : > "$ENV_FILE"
    warn ".env.example missing — created empty scripts/.env."
  fi
  # Append the 5 keys the runtime reads that .env.example omits (+ OMI shards),
  # without clobbering committed values. These mirror the working box's .env.
  {
    echo ""
    echo "# ---- keys required by the runtime (added by setup.sh) ----"
    echo "export RELIQUARY_INSTALL_DIR=$INSTALL_DIR"
    echo "export VLLM_WORKER_MULTIPROC_METHOD=spawn"
    echo "export GRAIL_ATTN_IMPL=flash_attention_2   # ONLY this is read by the code; RELIQUARY_ATTN_IMPL=sdpa above is ignored"
    echo "export RELIQUARY_OVERSAMPLE=64"
    echo "export RELIQUARY_PROMPT_SOURCES=gsm8k,augmented_gsm8k"
    echo "export RELIQUARY_OMI_SHARDS=2   # MUST stay 2: inzone_pool.json idxs index this fixed shard ordering"
    echo "export RELIQUARY_CHECKPOINT=$BASE_MODEL   # production checkpoint (overrides the gpt2 example default)"
  } >> "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  ok "created scripts/.env from .env.example + appended runtime keys (mode 600)"
  warn "SECURITY: secrets inherited from .env.example were BLANKED. HF auth is NOT required (all repos are public) — leave HF_TOKEN empty unless you actually need it."
  echo ""
  echo "Fill these keys in $ENV_FILE before launching:"
  echo "    BT_WALLET_NAME           (e.g. ronnywebdev)"
  echo "    BT_HOTKEY                (e.g. hdev0301 — must be REGISTERED on subnet 81)"
  echo "    HF_TOKEN                 *** SECRET *** (optional; models are public, leave blank if unused)"
  echo "    HUGGING_FACE_HUB_TOKEN   *** SECRET *** (optional; mirror of HF_TOKEN)"
  echo "    GRAIL_STATE_HMAC_KEY     *** SECRET *** (any stable string)"
  echo "    R2_ACCESS_KEY_ID         *** SECRET *** (optional; only for dataset upload)"
  echo "    R2_SECRET_ACCESS_KEY     *** SECRET *** (optional; only for dataset upload)"
  echo "    R2_ENDPOINT_URL / R2_ACCOUNT_ID / R2_REGION / R2_BUCKET_ID  (optional, non-secret)"
  echo "    RELIQUARY_EXTERNAL_IP / RELIQUARY_EXTERNAL_PORT  (optional; validator-only)"
fi

# Ensure the bittensor wallet dir exists
mkdir -p /root/.bittensor/wallets
ok "ensured /root/.bittensor/wallets exists"

# --------------------------------------------------------------------------- #
banner "(10) NEXT STEPS"
# --------------------------------------------------------------------------- #
cat <<EOF

Setup of the Reliquary miner environment is complete (or surfaced what is missing).

REMAINING MANUAL STEPS before the miner will mine correctly:

  1) WALLET (hotkey secret + coldkeypub only; coldkey secret NOT needed on a miner):
       Option A (copy):   scp -r SRC:/root/.bittensor/wallets/ronnywebdev \\
                              /root/.bittensor/wallets/ronnywebdev
       Option B (regen):  $VENV/bin/btcli wallet regen-hotkey \\
                              --wallet.name ronnywebdev --wallet.hotkey hdev0301   # needs mnemonic
                          $VENV/bin/btcli wallet regen-coldkeypub \\
                              --wallet.name ronnywebdev --ss58 <coldkey_ss58>
     The hotkey MUST already be REGISTERED on subnet 81 (NETUID=81, finney) — that
     is on-chain state, not a file. Reuse the same registered hotkey (one UID); do
     NOT register a new one unless you want a separate UID. chmod the keys ~600.

  2) inzone_pool.json: copy it from the source box if step (8) warned it is missing:
       scp SRC:/root/inzone_pool.json $INZONE_POOL
     (Also copy /root/format_analysis.py, /root/harvest_inzone.py,
      /root/topminer_vectors.jsonl, /root/topminer_accepts.jsonl if you want to be
      able to regenerate the pool later — none reproduce the exact pool on a bare box.)

  3) scripts/.env: fill BT_WALLET_NAME / BT_HOTKEY and any secrets you use; confirm
     RELIQUARY_CHECKPOINT=$BASE_MODEL and GRAIL_ATTN_IMPL=flash_attention_2.

  4) DRIVER: confirm nvidia-smi shows a driver >= ${DRIVER_MIN_MAJOR} (CUDA 13 / sm_120).
     torch.cuda.is_available() above must be True before launching.

  5) LAUNCH (this script does NOT launch): use the curation launcher
       bash /root/run_miner.sh
     It sources scripts/.env, exports the curation knobs (RELIQUARY_CURATE=1,
     CURATE_TARGET_K=6, two-stage screen [0.03,0.97], hot-pool path/frac/cap) and
     runs: reliquary.cli.main mine --checkpoint $BASE_MODEL --two-stage
           --prompt-idx-file $INZONE_POOL --oversample 160 --pool-size 96 ...
     Then tail /root/miner.log.

  *** GRAIL PROOF SMOKE-TEST CAVEAT (NEW GPU ARCH) ***
  This is a DIFFERENT GPU arch (Blackwell sm_120) than where the pool/checkpoint
  were proven (H200 sm_90). The GRAIL proof binds generated logits to the model
  on THIS hardware; cross-arch numeric drift can cause logprob_mismatch rejects
  even with identical pins. Before a full run, do ONE short window
  (RELIQUARY_MAX_NEW_TOKENS small / low oversample) and confirm the validator
  ACCEPTS the submission (no integrity/logprob rejects in miner.log) before
  committing to the full curation config.

EOF
if [ "$MISSING_CRITICAL" = "1" ] || [ "$NOT_READY" = "1" ]; then
  echo ""
  warn "########################################################################"
  [ "$MISSING_CRITICAL" = "1" ] && warn "# SETUP INCOMPLETE — copy missing artifacts from source (see warnings above):"
  [ "$MISSING_CRITICAL" = "1" ] && warn "#   inzone_pool.json, the curation-patched repo, and/or /root/run_miner.sh"
  [ "$NOT_READY" = "1" ]        && warn "# GPU NOT READY — driver < ${DRIVER_MIN_MAJOR} or CUDA unavailable; do NOT launch yet."
  warn "########################################################################"
  exit 2
fi
ok "DONE — environment ready. Fill scripts/.env + wallet, then: bash /root/run_miner.sh"
