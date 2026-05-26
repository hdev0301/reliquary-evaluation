"""
prep_supervisor.py — keep prep_dataset.py running against the validator's
current published checkpoint.

The supervisor polls the validator's /state endpoint. When the published
revision changes, it SIGTERMs the running prep child, waits for it to exit,
and respawns it pinned to the new revision via --repo-id / --revision. It
also respawns on unexpected child exit (crash, OOM, etc.).

Usage:

    python scripts/prep_supervisor.py --cuda 0 --hotkey <ss58_or_label>

The supervisor and the prep child share one log file; supervisor lines are
prefixed `supervisor`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("supervisor")


def fetch_ckpt(validator_url: str) -> tuple[str, str, int] | None:
    try:
        with urllib.request.urlopen(
            validator_url.rstrip("/") + "/state", timeout=15
        ) as r:
            st = json.loads(r.read())
        return st["checkpoint_repo_id"], st["checkpoint_revision"], int(st["checkpoint_n"])
    except Exception as e:
        logger.warning("validator /state failed: %s", e)
        return None


def spawn(repo_id: str, rev: str, args, log_path: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "prep_dataset.py"),
        "--cuda", str(args.cuda),
        "--repo-id", repo_id,
        "--revision", rev,
    ]
    if args.hotkey:
        cmd += ["--hotkey", args.hotkey]
    if args.max_new_tokens is not None:
        cmd += ["--max-new-tokens", str(args.max_new_tokens)]
    if args.batch_prompts is not None:
        cmd += ["--batch-prompts", str(args.batch_prompts)]
    if args.prescreen_rollouts is not None:
        cmd += ["--prescreen-rollouts", str(args.prescreen_rollouts)]
    if args.skip_prescreen_threshold is not None:
        cmd += ["--skip-prescreen-threshold", str(args.skip_prescreen_threshold)]

    log_f = open(log_path, "a")
    log_f.write(
        f"\n=== supervisor: spawning prep_dataset rev={rev} at {time.strftime('%FT%T')} ===\n"
    )
    log_f.flush()
    # Reduce CUDA fragmentation. With batch_prompts × M_ROLLOUTS sequences at
    # 8192 tokens, the KV cache plus activations can leave ~25 GB reserved
    # but unallocated, which causes spurious OOMs. expandable_segments lets
    # PyTorch reclaim that.
    spawn_env = os.environ.copy()
    spawn_env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    return subprocess.Popen(
        cmd,
        env=spawn_env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )


def prune_hf_cache(repo_id: str, keep_rev: str) -> int:
    """Delete all HF cache snapshots for ``repo_id`` except ``keep_rev``,
    plus any blobs no longer referenced by ``keep_rev``. Returns bytes
    freed. Never raises — best-effort.
    """
    try:
        cache_root = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        model_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
        snapdir = model_dir / "snapshots"
        blobdir = model_dir / "blobs"
        if not snapdir.is_dir() or not blobdir.is_dir():
            return 0
        keep_snap = snapdir / keep_rev
        if not keep_snap.is_dir():
            # keep_rev hasn't finished downloading yet — don't prune,
            # would risk deleting blobs the in-progress download needs.
            return 0
        referenced: set[str] = set()
        for f in keep_snap.iterdir():
            if f.is_symlink():
                try:
                    target = f.resolve()
                    if target.parent == blobdir.resolve():
                        referenced.add(target.name)
                except Exception:
                    pass
        bytes_freed = 0
        for d in snapdir.iterdir():
            if d.name != keep_rev:
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
        for blob in blobdir.iterdir():
            if blob.name in referenced:
                continue
            # Skip anything that isn't a pure-hash blob name. HF uses
            # ``<hash>.incomplete`` (and lock files etc.) for in-progress
            # downloads, and deleting those breaks the active download.
            if "." in blob.name:
                continue
            try:
                bytes_freed += blob.stat().st_size
                blob.unlink()
            except Exception:
                pass
        return bytes_freed
    except Exception:
        logger.exception("prune_hf_cache failed")
        return 0


def stop_child(proc: subprocess.Popen, grace_secs: int) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_secs)
    except subprocess.TimeoutExpired:
        logger.warning("child pid=%d did not exit after %ds; SIGKILL", proc.pid, grace_secs)
        proc.kill()
        proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--validator-url",
        default=os.environ.get("RELIQUARY_VALIDATOR_URL"),
    )
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--hotkey", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--batch-prompts", type=int, default=None)
    ap.add_argument("--prescreen-rollouts", type=int, default=None)
    ap.add_argument("--skip-prescreen-threshold", type=float, default=None)
    ap.add_argument("--poll-secs", type=int, default=30)
    ap.add_argument("--shutdown-grace-secs", type=int, default=30)
    ap.add_argument("--log", default="/tmp/prep_supervisor.log")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | supervisor | %(message)s",
    )

    if not args.validator_url:
        logger.error("need --validator-url or RELIQUARY_VALIDATOR_URL env")
        return 2

    stopping = False

    def handle_signal(signo, frame):
        nonlocal stopping
        logger.info("received signal %d; shutting down", signo)
        stopping = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    current_rev: str | None = None
    last_repo_id: str | None = None
    proc: subprocess.Popen | None = None

    try:
        while not stopping:
            info = fetch_ckpt(args.validator_url)

            if info is not None:
                repo_id, rev, ckpt_n = info

                if rev != current_rev:
                    if proc is not None:
                        logger.info(
                            "ckpt advance %s -> %s; stopping child pid=%d",
                            (current_rev or "?")[:12], rev[:12], proc.pid,
                        )
                        stop_child(proc, args.shutdown_grace_secs)
                    logger.info(
                        "spawning prep for ckpt=%s repo=%s n=%d",
                        rev[:12], repo_id, ckpt_n,
                    )
                    proc = spawn(repo_id, rev, args, args.log)
                    current_rev = rev
                    last_repo_id = repo_id
                    # Prune stale ckpt caches so we don't fill disk. We
                    # prune BEFORE the new child finishes downloading,
                    # which is safe — old snapshots are inert and new
                    # download writes to blobs/ as completed files.
                    freed = prune_hf_cache(repo_id, rev)
                    if freed > 0:
                        logger.info(
                            "pruned HF cache: %.1f GB freed (kept %s)",
                            freed / (1024 ** 3), rev[:12],
                        )
                elif proc is not None and proc.poll() is not None:
                    logger.warning(
                        "child exited code=%s; respawning on same ckpt=%s",
                        proc.returncode, rev[:12],
                    )
                    proc = spawn(repo_id, rev, args, args.log)
                else:
                    # Steady state on same ckpt — opportunistic prune in
                    # case the post-advance prune was too early (new
                    # snapshot dir hadn't been written yet).
                    freed = prune_hf_cache(repo_id, rev)
                    if freed > 0:
                        logger.info(
                            "pruned HF cache (steady-state): %.1f GB "
                            "freed (kept %s)",
                            freed / (1024 ** 3), rev[:12],
                        )
            else:
                # Validator unreachable; let any running child keep going.
                # If the child also died, respawn on the last-known ckpt so
                # we don't sit idle waiting for the validator to come back.
                if proc is not None and proc.poll() is not None:
                    if current_rev is not None:
                        logger.warning(
                            "child exited code=%s while validator down; "
                            "respawning on last-known ckpt=%s",
                            proc.returncode, current_rev[:12],
                        )
                        # Reuse last-known repo_id stash; if we never got one
                        # (e.g. validator was down at startup), fall back to
                        # the validator-pinned default.
                        proc = spawn(
                            last_repo_id or "R0mAI/reliquary-sn-v23",
                            current_rev, args, args.log,
                        )
                    else:
                        logger.warning(
                            "child exited code=%s and validator unreachable "
                            "with no known ckpt; will retry",
                            proc.returncode,
                        )

            for _ in range(args.poll_secs):
                if stopping:
                    break
                time.sleep(1)
    finally:
        if proc is not None:
            logger.info("supervisor exit: stopping child pid=%d", proc.pid)
            stop_child(proc, args.shutdown_grace_secs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
