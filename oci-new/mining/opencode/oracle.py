"""Local frontier oracle for OpenCode — a faithful proxy of the hidden grade.

OpenCode rewards are validator-authoritative: miners get prompts only, not the
hidden structured cases. But the validator's hidden cases are *derived* from
the **public** ``nvidia/OpenCodeInstruct`` ``unit_tests`` column by exactly the
pipeline in ``scripts/build_opencodeinstruct_subset.py`` (``structure_tests`` →
deterministic double-exec filter). So a miner can reconstruct the same
structured cases offline (``mining/scripts/build_local_oracle.py``), key them by
``sha256(prompt)`` (revision-independent), and grade its own completions with
the **validator's own grader** (``GraderClient`` → identical sandbox/worker/
comparison).

This is frontier *prediction*, not reward gaming:
  * we never submit a reconstructed reward — OpenCode submissions carry the
    0.0 placeholder, and the validator recomputes the real reward — so this
    can never cause ``REWARD_MISMATCH``;
  * we only use the predicted pass-vector to estimate σ and decide which
    prompts deserve a slot (the "cheap proxy to predict frontier likelihood"
    that ``docs/mining.md`` explicitly endorses).

Divergence from the true grade is small but nonzero (the validator may drop a
few cases its determinism filter rejects), which is why the selector keeps a σ
margin. If a future task makes cases private/generated, swap in the
self-consistency proxy (``mining.opencode.frontier``) — the rest is unchanged.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)


def prompt_sha(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode()).hexdigest()


class LocalOracle:
    """sha256(prompt) → reconstructed structured cases, + validator-grade IPC."""

    def __init__(self, sidecar_path: str) -> None:
        self.sidecar_path = sidecar_path
        self._by_sha: dict[str, list[dict]] = {}
        self._meta: dict[str, object] = {}
        self._grader = None  # GraderClient (lazy)

    # ------------------------------------------------------------------
    def load(self) -> "LocalOracle":
        path = self.sidecar_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"oracle sidecar not found: {path}. Build it once with "
                "`python -m mining.scripts.build_local_oracle` (see mining/README.md)."
            )
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            blob = json.load(fh)
        # cases are stored as JSON-encoded lists (Arrow-safe, like the env).
        raw = blob.get("by_sha", {})
        self._by_sha = {
            sha: (json.loads(v) if isinstance(v, str) else v) for sha, v in raw.items()
        }
        self._meta = blob.get("meta", {})
        logger.info(
            "loaded OpenCode oracle: %d prompts with reconstructed cases (%s)",
            len(self._by_sha), path,
        )
        return self

    def __len__(self) -> int:
        return len(self._by_sha)

    def has_cases(self, prompt_text: str) -> bool:
        return prompt_sha(prompt_text) in self._by_sha

    def cases_for(self, prompt_text: str) -> "list[dict] | None":
        return self._by_sha.get(prompt_sha(prompt_text))

    # ------------------------------------------------------------------
    # Grading via the validator's own grader (maximal fidelity).
    # ------------------------------------------------------------------
    def ensure_grader(self, *, allow_unsandboxed: bool = False) -> None:
        """Make sure a Reliquary grader server is reachable; start one if not.

        Reuses the validator/CLI launch logic so the local prediction grade is
        byte-for-byte the validator's grade. With ``runsc`` present it runs
        sandboxed; on an isolated lab box set
        ``RELIQUARY_ALLOW_UNSANDBOXED_GRADER=1``.
        """
        from reliquary.constants import GRADER_SOCKET_PATH
        from reliquary.environment.grader_client import GraderClient

        self._grader = GraderClient(GRADER_SOCKET_PATH)
        try:
            from reliquary.cli.main import _ensure_grader_running, _grader_is_running

            if not _grader_is_running(GRADER_SOCKET_PATH):
                if allow_unsandboxed:
                    os.environ.setdefault("RELIQUARY_ALLOW_UNSANDBOXED_GRADER", "1")
                _ensure_grader_running()
        except Exception:
            logger.exception(
                "could not auto-start grader; ensure one is running at %s",
                GRADER_SOCKET_PATH,
            )

    def grade(self, code: str, cases: list[dict]) -> float:
        """pass/total in [0,1] for one completion — identical to the validator."""
        from reliquary.constants import GRADER_EVAL_TIMEOUT_SECONDS
        from reliquary.environment.grader_client import GraderClient

        if self._grader is None:
            from reliquary.constants import GRADER_SOCKET_PATH

            self._grader = GraderClient(GRADER_SOCKET_PATH)
        return self._grader.evaluate_cases(
            code, cases, timeout_s=GRADER_EVAL_TIMEOUT_SECONDS
        )

    def grade_completions(
        self, completions: list[str], cases: list[dict], *, concurrency: int = 8
    ) -> list[float]:
        """Predicted reward for each completion (extracts code the env's way).

        Graded concurrently across the grader's warm worker pool — sequential
        grading is the screening bottleneck (each eval can hit the 5s timeout on
        pathological/looping model code).
        """
        from concurrent.futures import ThreadPoolExecutor

        from reliquary.environment.opencodeinstruct import _extract_python

        codes = [_extract_python(c or "") for c in completions]
        if concurrency <= 1 or len(codes) <= 1:
            return [self.grade(code, cases) for code in codes]
        with ThreadPoolExecutor(max_workers=min(concurrency, len(codes))) as ex:
            return list(ex.map(lambda code: self.grade(code, cases), codes))
