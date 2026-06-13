"""Local frontier oracle for OpenMath — and it's the validator's own grader.

Unlike OpenCode (hidden structured cases → sandboxed code execution), OpenMath
rewards are answer-equality against the dataset's **public** ``expected_answer``,
which ``OpenMathInstructEnvironment.get_problem`` already returns as
``ground_truth``. The validator scores with ``_compute_omi_reward`` (last
``\\boxed{}`` matched by numeric value, else normalized string / LaTeX value
equality). We call that exact same function, so:

  * no sandbox, no runsc, no grader server, no hidden-case reconstruction;
  * our local reward == the validator's reward (same public answer, same
    matcher), so the proxy-vs-validator gap that drives OpenCode's
    ``out_of_zone`` is almost entirely absent here.

As with OpenCode we never submit the predicted reward — submissions carry the
0.0 placeholder and the validator recomputes — so this can't cause
``REWARD_MISMATCH``; it only predicts σ to decide which prompts deserve a slot.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Relative tolerance below which a clean-numeric boxed answer is treated as a
# PASS rather than a fail — the deployed validator is numerically tolerant
# (accepts 200/7 == 28.57, 0.333 == 1/3), so a "fail" must differ by MORE than
# this to be trusted (else the group screens in-zone locally but the validator
# rejects it OUT_OF_ZONE). Conservative default; raise via env to be stricter
# about what counts as a genuine wrong answer.
ROBUST_FAIL_MARGIN = float(os.environ.get("RELIQUARY_OMI_ROBUST_FAIL_MARGIN", "0.01"))


class MathOracle:
    """Stateless answer-equality grade — identical to the validator's grade.

    No sidecar file and no grader process: the ground truth is the public
    ``expected_answer`` carried on each problem, and grading is pure CPU
    (regex + a bounded sympy fallback for LaTeX expressions).
    """

    def grade(self, completion: str, ground_truth: str) -> float:
        """1.0 if the completion's final answer equals ``ground_truth``, else 0.0.

        Uses the EXACT ``_compute_omi_reward`` — the same function the validator
        runs as ``env.compute_reward``. OMI is NON-authoritative: the validator
        computes σ from the rewards the MINER submits and only verifies each
        against its own ``compute_reward`` (REWARD_MISMATCH on any disagreement).
        So the miner must (a) grade with this exact function and (b) submit the
        grades (see ``fire`` reward_for) — matching the validator bit-for-bit.
        Any "smarter"/tolerant local grade would diverge from the validator and
        trip REWARD_MISMATCH, so we deliberately do NOT second-guess it here.
        """
        from reliquary.environment.openmathinstruct import _compute_omi_reward

        return _compute_omi_reward({"ground_truth": ground_truth}, completion or "")

    @staticmethod
    def _is_clean_numeric(s: str) -> bool:
        """True iff ``s`` is a pure number / numeric LaTeX (no words, units, or
        free variables) — i.e. an unambiguous value any grader compares the same."""
        from reliquary.environment.openmathinstruct import (
            _as_number, _expr_str_is_safe, _latex_to_pyexpr,
        )

        if not s:
            return False
        if _as_number(s) is not None:
            return True
        eg = _latex_to_pyexpr(s)
        return eg is not None and _expr_str_is_safe(eg)

    @staticmethod
    def _str_value(s: str):
        """Float value of a normalized numeric string / numeric LaTeX (or None)."""
        from reliquary.environment.openmathinstruct import (
            _as_number, _expr_str_is_safe, _latex_to_pyexpr,
        )

        if not s:
            return None
        v = _as_number(s)
        if v is not None:
            return float(v)
        eg = _latex_to_pyexpr(s)
        if eg is None or not _expr_str_is_safe(eg):
            return None
        try:
            from sympy.parsing.sympy_parser import (
                implicit_multiplication_application, parse_expr, standard_transformations,
            )
            tr = standard_transformations + (implicit_multiplication_application,)
            val = complex(parse_expr(eg, transformations=tr).evalf())
            return val.real if abs(val.imag) < 1e-9 else None
        except Exception:
            return None

    @classmethod
    def _boxed_value(cls, completion: str):
        """Numeric value of the completion's last \\boxed{...} (or None)."""
        from reliquary.environment.openmathinstruct import (
            _last_boxed_only_string, _normalize_answer, _strip_boxed_wrapper,
        )

        bx = _last_boxed_only_string(completion)
        if bx is None:
            return None
        return cls._str_value(_normalize_answer(_strip_boxed_wrapper(bx)))

    def grade_completions(
        self, completions: list[str], ground_truth: str, *, concurrency: int = 8
    ) -> list[float]:
        """Predicted reward for each completion against the same public answer.

        ``_compute_omi_reward`` is CPU-light and does no IPC (sympy only on the
        rare LaTeX-expression fallback path), so grading inline is faster than a
        thread pool's overhead. ``concurrency`` is accepted for interface parity
        with the OpenCode oracle and otherwise ignored.
        """
        return [self.grade(c, ground_truth) for c in completions]
