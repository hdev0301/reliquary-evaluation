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
import signal
import subprocess
import sys
import time
import urllib.request

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

    log_f = open(log_path, "a")
    log_f.write(
        f"\n=== supervisor: spawning prep_dataset rev={rev} at {time.strftime('%FT%T')} ===\n"
    )
    log_f.flush()
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
    )


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
                elif proc is not None and proc.poll() is not None:
                    logger.warning(
                        "child exited code=%s; respawning on same ckpt=%s",
                        proc.returncode, rev[:12],
                    )
                    proc = spawn(repo_id, rev, args, args.log)
            else:
                # Validator unreachable; let any running child keep going.
                if proc is not None and proc.poll() is not None:
                    logger.warning(
                        "child exited code=%s and validator unreachable; will retry",
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
