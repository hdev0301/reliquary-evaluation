#!/usr/bin/env bash
# Build the OCI rootfs for the grader sandbox.
#
# Strategy: extract `python:3.12-slim` Docker image's rootfs into
# reliquary/environment/grader/bundle/rootfs/, then copy worker.py
# into /opt/worker.py inside the rootfs. The bundle config.json
# (sibling file) references this rootfs.
#
# Idempotent: re-running rebuilds the rootfs from scratch.
set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-/opt/reliquary/reliquary/environment/grader/bundle}"
ROOTFS="${BUNDLE_DIR}/rootfs"
WORKER_SRC="${WORKER_SRC:-/opt/reliquary/reliquary/environment/grader/worker.py}"
# Pinned by digest: the CPython patch release in the sandbox rootfs decides
# the grader's pass/total, so a floating `python:3.12-slim` tag would let two
# validators grade identical code differently (cross-box determinism). Bump
# the digest deliberately. (Resolve a new one with:
#   docker buildx imagetools inspect python:3.12-slim --format '{{.Manifest.Digest}}')
PY_IMAGE="${PY_IMAGE:-python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203}"

echo "[build_grader_bundle] BUNDLE_DIR=${BUNDLE_DIR}"
echo "[build_grader_bundle] ROOTFS=${ROOTFS}"
echo "[build_grader_bundle] WORKER_SRC=${WORKER_SRC}"

if [[ ! -f "${WORKER_SRC}" ]]; then
  echo "ERROR: worker.py not found at ${WORKER_SRC}" >&2
  exit 1
fi

# Clean any previous rootfs.
rm -rf "${ROOTFS}"
mkdir -p "${ROOTFS}"

# Pull and export the python:3.12-slim rootfs.
# We use `docker create` + `docker export` to materialize a flat tarball,
# then untar into the bundle directory. Requires Docker in the build env.
CID="$(docker create "${PY_IMAGE}" /bin/true)"
trap 'docker rm -f "${CID}" >/dev/null 2>&1 || true' EXIT
docker export "${CID}" | tar -x -C "${ROOTFS}"

# Drop the worker.py into /opt inside the rootfs.
mkdir -p "${ROOTFS}/opt"
install -m 0644 "${WORKER_SRC}" "${ROOTFS}/opt/worker.py"

# Sanity check.
if [[ ! -x "${ROOTFS}/usr/local/bin/python3" ]]; then
  echo "ERROR: python3 not found in rootfs at /usr/local/bin/python3" >&2
  exit 1
fi

echo "[build_grader_bundle] done. Bundle ready at ${BUNDLE_DIR}"
