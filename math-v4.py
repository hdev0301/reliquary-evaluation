"""Hendrycks MATH environment with cohort metadata exposed for miner-side priors.

DEPLOYMENT: this file is the v4 patch to ``reliquary/environment/math.py`` —
copy it over the upstream file on the miner box:

    cp math-v4.py /root/reliquary/reliquary/environment/math.py

The ONLY behavioural change vs the upstream env is two extra keys
(``level`` and ``subject``) on the dict returned by ``get_problem``.

These fields are read **only** by ``engine-v4``'s ``PromptSelector`` to
inherit a strong Beta prior on cold prompts from observed solve rates
in the same (level, subject) cohort cell. The validator never reads
them — ``compute_reward`` is byte-for-byte unchanged. So this patch is
backward compatible with stock validators and any other miner stack
that ignores the extra dict keys.

Cohort definitions follow the original Hendrycks MATH taxonomy:
  - level   ∈ {"Level 1", ..., "Level 5"}                        — 5 buckets
  - subject ∈ {"Algebra", "Counting & Probability", "Geometry",
               "Intermediate Algebra", "Number Theory",
               "Prealgebra", "Precalculus"}                       — 7 buckets

(qwedsacf/competition_math exposes these as ``row["level"]`` and
``row["type"]``. We accept either ``type`` or ``subject`` defensively
in case future mirrors rename the column.)
"""

from __future__ import annotations

import hashlib
import re
from typing import ClassVar, Optional


# ---------------------------------------------------------------------------
# Answer extraction — balanced-brace parsing of the last \boxed{...} / \fbox{...}
# ---------------------------------------------------------------------------

def _last_boxed_only_string(text: str) -> Optional[str]:
    """Return the last \\boxed{...} / \\fbox{...} substring (including the
    wrapper), or None if no balanced wrapper is found.

    This is the Hendrycks-style parser: it walks braces to handle nested
    expressions like \\boxed{\\frac{1}{2}} that a regex cannot match.
    """
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx < 0:
        return None

    open_idx = text.index("{", idx)
    depth = 0
    for j in range(open_idx, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[idx : j + 1]
    return None


def _strip_boxed_wrapper(s: str) -> str:
    """If ``s`` starts with \\boxed{ or \\fbox{ and ends with a matching },
    return the inner content. Otherwise return ``s`` unchanged.
    """
    for prefix in (r"\boxed{", r"\fbox{"):
        if s.startswith(prefix) and s.endswith("}"):
            return s[len(prefix) : -1]
    return s


# ---------------------------------------------------------------------------
# Answer normalization — conservative LaTeX simplification for comparison
# ---------------------------------------------------------------------------

_TEXT_RE = re.compile(r"\\text\{([^}]*)\}")
_MBOX_RE = re.compile(r"\\mbox\{([^}]*)\}")


def _normalize_answer(s: str) -> str:
    """Conservative LaTeX normalization for equality comparison.

    Intentionally string-level only (no CAS): the rules below cover the
    transforms that actually occur in Hendrycks MATH ground truths without
    changing the meaning of the expression.
    """
    if s is None:
        return ""
    for macro in (r"\!", r"\,", r"\ ", r"\;", r"\:"):
        s = s.replace(macro, "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = _TEXT_RE.sub(r"\1", s)
    s = _MBOX_RE.sub(r"\1", s)
    s = s.replace(r"\$", "").replace("$", "")
    s = s.strip().rstrip(".").strip()
    s = re.sub(r"\s+", "", s)
    return s


def _compute_math_reward(problem: dict, completion: str) -> float:
    """Score a MATH completion.

    Returns 1.0 when the last ``\\boxed{...}`` in the completion, stripped
    and normalized, equals the ground-truth answer (also stripped/normalized).
    Returns 0.0 otherwise. Never raises.
    """
    try:
        boxed = _last_boxed_only_string(completion)
        if boxed is None:
            return 0.0
        candidate = _normalize_answer(_strip_boxed_wrapper(boxed))
        gt_raw = str(problem.get("ground_truth", ""))
        gt = _normalize_answer(_strip_boxed_wrapper(gt_raw))
        return 1.0 if candidate == gt and gt != "" else 0.0
    except Exception:
        return 0.0


class MATHEnvironment:
    """Environment backed by the full Hendrycks MATH set (12 500 problems).

    Ground truths are extracted once from the ``solution`` field by taking
    the content of the last \\boxed{...}; completions are scored with the
    same extraction against the completion text.

    Uses the ``qwedsacf/competition_math`` HF mirror.

    v4 addition: ``get_problem`` returns two extra keys (``level``,
    ``subject``) so a miner-side picker can group prompts by their MATH
    taxonomy cohort and inherit a strong Beta prior on cold prompts in an
    already-observed cohort. The validator never reads these fields — its
    ``compute_reward`` only touches ``ground_truth``.
    """

    name: str = "math"

    _dataset_cache: ClassVar[Optional[object]] = None

    def __init__(self) -> None:
        if MATHEnvironment._dataset_cache is None:
            import datasets as hf_datasets
            MATHEnvironment._dataset_cache = hf_datasets.load_dataset(
                "qwedsacf/competition_math", split="train"
            )
        self._dataset = MATHEnvironment._dataset_cache

    def __len__(self) -> int:
        return len(self._dataset)

    def get_problem(self, index: int) -> dict:
        """Return problem at *index* (wraps modulo dataset length).

        v4: also surfaces ``level`` and ``subject`` for miner-side cohort
        bucketing. Both fields are best-effort strings; empty when absent
        in the underlying row (the picker treats empty cohort as Beta(1, 1)
        fallback, identical to v3 behaviour).
        """
        idx = index % len(self._dataset)
        row = self._dataset[idx]
        question: str = row["problem"]
        solution: str = row["solution"]
        boxed = _last_boxed_only_string(solution)
        gt_str = _strip_boxed_wrapper(boxed) if boxed else ""
        problem_id = hashlib.sha256(question.encode()).hexdigest()[:16]

        # MATH taxonomy. Original Hendrycks uses ``type`` as the subject
        # column; some mirrors rename to ``subject``. Try both.
        level = row.get("level", "") if hasattr(row, "get") else ""
        subject = (
            (row.get("type", "") if hasattr(row, "get") else "")
            or (row.get("subject", "") if hasattr(row, "get") else "")
        )

        return {
            "prompt": question,
            "ground_truth": gt_str,
            "id": problem_id,
            "level": str(level) if level is not None else "",
            "subject": str(subject) if subject is not None else "",
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        """Score a completion using MATH boxed-answer reward."""
        return _compute_math_reward(problem, completion)
