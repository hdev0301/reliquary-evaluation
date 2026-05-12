"""Miner engine v4 — accuracy picker + batched GRAIL + serial pipeline.

DEPLOYMENT: this file replaces ``reliquary/miner/engine.py``. On the miner
box, alongside the math env patch:

    cp engine-v4.py /root/reliquary/reliquary/miner/engine.py
    cp math-v4.py   /root/reliquary/reliquary/environment/math.py

``launcher-v3.py`` and ``main-v3.py`` continue to work unchanged — the
MiningEngine constructor signature is identical to v3, including the
``stats_path`` kwarg. The vLLM adapter detection sentinel
(``_is_vllm_adapter``) is preserved so the existing vLLM path keeps
running. Stats files written by v3 load cleanly into v4 (v1 schema
migration is transparent).

What changed vs v3
==================

1. **Picker — three new accuracy levers.**

   a) **Cohort priors from MATH taxonomy.** Each prompt has a
      ``(level, subject)`` cell (5 levels × 7 subjects = 35 cells).
      When the same cohort has ≥ 8 observations, an unseen prompt
      inherits ``Beta(α_cohort, β_cohort)`` as its prior instead of
      the uninformative ``Beta(1, 1)``. Cold-start picks land in-zone
      far more often once a handful of cohorts have warmed up.

   b) **Checkpoint-versioned posterior decay.** Each prompt's
      ``(solves, attempts)`` is tagged with the ``checkpoint_n`` it
      was last updated under. When the validator publishes a new
      revision, ``posterior()`` lazily blends every prompt's accumulated
      counts toward its cohort prior by factor ``_CHECKPOINT_DECAY ** bumps``
      (default 0.5 per bump). Cohort stats decay slower
      (``_COHORT_DECAY = 0.85``). This kills v3's "data from last
      week's checkpoint poisoning today's pick" failure mode.

   c) **Congestion model + SUPERSEDED blacklist.** Track an EMA of
      how often each prompt newly enters ``cooldown_prompts`` — the
      proxy for "popular among smart miners". Down-weight ``zone_p``
      by that rate. Within each window, hard-blacklist any prompt
      that already returned SUPERSEDED to us — that race is lost.

   The picker is still best-of-K Thompson; K halves under race-mode
   pressure (``slots_filled ≥ 5``) for faster pick → faster submit.

2. **Batched GRAIL proof.** ``_build_rollout_submissions`` runs **one**
   padded HF forward pass for all M=8 rollouts in a group instead of
   M sequential ``[1, seq_len]`` calls. Saves 3–5 s/attempt on H200.

   Why this is safe (and how we double-check at runtime):

   - All M rollouts in a group share the prompt prefix.
   - We supply an ``attention_mask`` so right-padded tokens cannot
     contribute to or be attended-to by the real tokens, regardless
     of attention kernel.
   - For real positions, FA2 with mask reduces to the unmasked
     single-sequence forward used by the validator
     (``verifier.verify_commitment_proofs`` runs ``[1, seq_len]``
     unmasked). LayerNorm is per-token, attention softmax is per
     query-row, projections are per-token — no cross-row reductions.
     Bit-identity isn't guaranteed (kernel tile scheduling differs
     with batch size), but the differences sit far inside the GRAIL
     tolerance budget (``5000 + 5·√pos``).
   - Belt-and-suspenders: if **two consecutive** batched-proof
     submissions return ``GRAIL_FAIL``, the engine automatically falls
     back to per-rollout proofs for the rest of the session and logs
     a loud warning. Manual override is ``--per-rollout-proof``.

3. **Probe dropped by default.** Cohort priors (1.a) replace the
   information the cold-start probe used to provide. The probe code
   path is gone — it was costing 1–2 s/attempt for ever-diminishing
   value. Re-introducing it requires going back to v3.

4. **Schema-v2 posterior persistence.** New fields: per-prompt
   ``last_checkpoint``, ``cohort_counts``, ``arrival_ema``,
   ``superseded_lifetime``. v3 files load via a transparent migration
   (v1 ``counts`` become v2 ``counts`` with ``last_checkpoint=0`` and
   empty cohort table — they re-tag on the first new observation).

Engine invariants (preserved from v3)
=====================================

- Per-window randomness derived from ``block_hash(state.window_n)``.
- Bit-identical GRAIL forward path against the validator
  (``forward_single_layer`` / ``flash_attention_2``).
- Pre-submit ``/state`` recheck to skip doomed POSTs after a window
  rolled over during generation.
- Local OUT_OF_ZONE short-circuit.
- Atomic posterior persistence (no torn writes on miner kill).
- Two-GPU topology: ``vllm_gpu`` for generation, ``proof_gpu`` for
  GRAIL proofs (collapses to one GPU on single-card boxes via the
  ``effective_proof_gpu`` shim in ``main-v3.py``).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import random as _random

from reliquary.constants import (
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    SIGMA_MIN,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)
from reliquary.infrastructure import chain
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables — exposed as module constants so an operator can patch them
# without re-implementing the picker. Defaults are calibrated against the
# B_BATCH=8, M_ROLLOUTS=8, BOOTSTRAP=σ≥0.43 protocol.
# ---------------------------------------------------------------------------

# Emit a rolling counter summary every N submission attempts. Also governs
# how often _PromptStats is flushed to disk (atomic-replace).
_SUMMARY_EVERY = 10

# Default location for the on-disk posterior cache. Resolvable from the
# engine constructor via ``stats_path=None`` to disable persistence.
_DEFAULT_STATS_PATH = ".reliquary_miner_stats.json"

# Posterior schema version written by ``save_to``. v1 had only ``counts``
# and ``lengths``; v2 adds cohort_counts, last_checkpoint, arrival_ema,
# superseded_lifetime. ``load_from`` reads both.
_STATS_SCHEMA = 2

# Blend factor toward the cohort prior per checkpoint bump. 0.5 means
# half the evidence is discarded per publish; after 4 publishes the
# prompt is back to ~6% of its original posterior mass (the rest is
# cohort prior). Tuned to match the empirical drift we expect from a
# 5e-6 LR × 10 windows × small-batch GRPO update.
_CHECKPOINT_DECAY = 0.5

# Cohort stats decay slower than individual prompts because the cohort
# aggregates across many problems — the underlying skill being measured
# is more stable than per-prompt solve rate.
_COHORT_DECAY = 0.85

# Cohort prior is informative only after this many observations in the
# cell. Below threshold we fall back to Beta(1, 1). Cohort effective
# sample size is capped at 20 so one cohort can't dominate prompt-level
# updates that disagree with it (e.g. an unusually easy/hard outlier in
# a hard/easy cohort).
_COHORT_MIN_OBS = 8
_COHORT_ESS_CAP = 20.0

# Best-of-K candidates in the Thompson picker. Default 32 mirrors v3.
_CANDIDATES_DEFAULT = 32

# Race mode: when the window is filling up, halve K and bias toward
# faster prompts (lower historical completion length). Reduces
# pick-loop latency at the cost of less exploration.
_CANDIDATES_RACE = 16
_RACE_MODE_SLOTS_THRESHOLD = 5  # of B_BATCH=8

# Congestion tilt: how strongly to down-weight zone_p by arrival_rate.
# arrival_rate ∈ [0, 1] roughly: 1 = enters cooldown almost every window
# (everyone is winning it), 0 = no other miner ever wins it.
_CONGESTION_WEIGHT = 0.4

# EMA alpha for the arrival_rate signal. Slow on-update (responds over
# ~20 entries) and very slow on quiescent decay.
_ARRIVAL_EMA_ALPHA = 0.05
_ARRIVAL_QUIESCENT_DECAY = 1.0 - _ARRIVAL_EMA_ALPHA * 0.1
_ARRIVAL_PRUNE_BELOW = 1e-4

# Consecutive batched-proof GRAIL_FAIL count that triggers fallback to
# per-rollout proofs. 2 absorbs a one-off network/race anomaly while
# catching any actual numerical divergence quickly.
_BATCHED_PROOF_FAIL_THRESHOLD = 2

# Reward threshold for "solved" — MATH rewards are {0, 1} but a
# continuous reward env could ship via this engine unchanged.
_SOLVED_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Log formatting helpers — compact, single-line-per-event status
# ---------------------------------------------------------------------------

def _fmt_rewards(rewards: list[float]) -> str:
    """Compact reward array for logs.

    Integer-valued rewards (MATH: 0.0/1.0) → "[1,1,0,1,1,0,1,0]"
    Otherwise → "[1.00,0.00,...]" with 2 decimal places.
    """
    if all(r in (0.0, 1.0) for r in rewards):
        return "[" + ",".join("1" if r else "0" for r in rewards) + "]"
    return "[" + ",".join(f"{r:.2f}" for r in rewards) + "]"


def _short_level(level: str) -> str:
    """\"Level 3\" → \"L3\". Empty → \"-\"."""
    if not level:
        return "-"
    return level.replace("Level ", "L").strip()


def _short_subject(subject: str, max_len: int = 12) -> str:
    """Truncate long subject names for compact display."""
    if not subject:
        return "-"
    s = subject.strip()
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# GPU memory snapshot (NVML) — cheap, robust, no torch dep
# ---------------------------------------------------------------------------

_NVML_INITIALIZED = False
_NVML_AVAILABLE = False


def _ensure_nvml() -> bool:
    """Lazy-init pynvml exactly once. Returns False if unavailable.

    We deliberately don't fall back to ``torch.cuda.mem_get_info`` —
    that path can deadlock when called from a thread while vLLM holds
    the CUDA context. NVML is process-global and lock-free.
    """
    global _NVML_INITIALIZED, _NVML_AVAILABLE
    if _NVML_INITIALIZED:
        return _NVML_AVAILABLE
    _NVML_INITIALIZED = True
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        _NVML_AVAILABLE = True
    except Exception:
        _NVML_AVAILABLE = False
    return _NVML_AVAILABLE


def _gpu_mem_compact(gpu_id: int) -> str:
    """Return ``"used/total GB"`` or ``"n/a"`` if NVML unavailable.

    Used to embed a low-overhead memory marker into hot-path log lines
    (PICK / SUB) so a memory leak shows up as a monotonically-growing
    ``used`` field over the session — without the verbosity of the
    full ``used:X.X/free:Y.Y/total:Z.Z`` string we use in the vllm
    adapter heartbeats.
    """
    if not _ensure_nvml():
        return "n/a"
    try:
        import pynvml  # type: ignore
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return f"{info.used / 1e9:.1f}/{info.total / 1e9:.0f}GB"
    except Exception:
        return "err"


# ---------------------------------------------------------------------------
# Dual-emit: structured logger.info + raw stderr fallback
# ---------------------------------------------------------------------------
#
# Engine-side belt-and-suspenders. When bittensor/vllm clobber the root
# logger AFTER engine.py is imported (e.g. during a checkpoint reload
# that rebuilds the vLLM ``LLM`` and triggers another dictConfig),
# logger.info(...) silently goes nowhere. We additionally write the same
# formatted string straight to ``sys.stderr`` so the operator never
# loses sight of PICK / GEN / SUB / OOZ / SUMMARY / window-banner lines.
#
# The duplicate when both channels work is acceptable: the stderr path
# has no timestamp prefix, so the timestamped logger line and the raw
# stderr line are visually distinguishable. Set ``RELIQUARY_NO_STDERR=1``
# to disable the stderr fallback once you've verified your deployment.

_STDERR_FALLBACK_ENABLED = os.environ.get("RELIQUARY_NO_STDERR", "") == ""


def _emit(level: int, fmt: str, *args) -> None:
    """Emit a structured line via logger AND raw stderr.

    Use for operator-critical events whose visibility must survive a
    handler clobber: per-attempt PICK/GEN/SUB/OOZ/SKIP/ERR, window
    banners, periodic SUMMARY. Routine debug/trace logs continue using
    plain ``logger.info`` / ``logger.debug``.
    """
    if args:
        try:
            msg = fmt % args
        except Exception:
            msg = f"{fmt} args={args!r}"
    else:
        msg = fmt
    logger.log(level, msg)
    if _STDERR_FALLBACK_ENABLED:
        try:
            # Use ``sys.__stderr__`` — the ORIGINAL, never-wrapped file
            # object — to bypass any tee installed by bittensor's
            # ``btlogging`` (which mirrors sys.stderr writes back into
            # the logging framework). Without this, every PICK/SUB
            # line would print twice once bittensor has been imported.
            stream = getattr(sys, "__stderr__", None) or sys.stderr
            print(f"[engine] {msg}", file=stream, flush=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Zone math — closed form for binary rewards over M_ROLLOUTS=8
# ---------------------------------------------------------------------------

# Binomial coefficients C(M_ROLLOUTS, k). Module-constant so the hot path
# is just multiplications/adds.
_BINOM_M: tuple[int, ...] = tuple(
    math.comb(M_ROLLOUTS, k) for k in range(M_ROLLOUTS + 1)
)

# σ ≥ SIGMA_MIN ⇔ k_solved ∈ [K_LO, K_HI] for binary rewards at M=8.
# (σ(k=1)≈0.331 < 0.43 ≤ σ(k=2)≈0.433.)
_K_LO: int = 2
_K_HI: int = 6


def _zone_probability(p: float) -> float:
    """``P(K_LO ≤ Binomial(M_ROLLOUTS, p) ≤ K_HI)``.

    The probability a group of M_ROLLOUTS rollouts at per-rollout solve
    probability ``p`` will pass the validator's σ ≥ SIGMA_MIN gate.
    """
    p = max(0.0, min(1.0, p))
    q = 1.0 - p
    total = 0.0
    for k in range(_K_LO, min(_K_HI, M_ROLLOUTS) + 1):
        total += _BINOM_M[k] * (p ** k) * (q ** (M_ROLLOUTS - k))
    return total


def _zone_status(rewards: list[float]) -> tuple[float, int, bool]:
    """Replicate the validator's ``rewards_std`` + ``is_in_zone`` checks.

    Returns ``(sigma, k_solved, in_zone)``. ``k_solved`` counts rewards
    ≥ ``_SOLVED_THRESHOLD`` — for MATH this is the only thing that
    matters; for a continuous env σ alone drives the gate.
    """
    n = len(rewards)
    if n < 2:
        return 0.0, 0, False
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    sigma = variance ** 0.5
    k_solved = sum(1 for r in rewards if r >= _SOLVED_THRESHOLD)
    return sigma, k_solved, sigma >= SIGMA_MIN


# ---------------------------------------------------------------------------
# Metrics — single-line summary the operator can grep
# ---------------------------------------------------------------------------

@dataclass
class _MinerMetrics:
    """Rolling counters surfaced to logs.

    Reports the picker's quality (in_zone_rate), the validator's
    accept rate, dominant reject reason, cohort warmup, and the
    batched-proof health channel (if it had to fall back, the operator
    needs to see it immediately).
    """

    submitted: int = 0
    accepted: int = 0
    rejected: int = 0
    network_errors: int = 0
    generated: int = 0
    local_out_of_zone: int = 0
    superseded_in_session: int = 0
    batched_proof_active: bool = True
    batched_proof_consecutive_fails: int = 0
    k_histogram: list[int] = field(default_factory=lambda: [0] * (M_ROLLOUTS + 1))
    reasons: dict[str, int] = field(default_factory=dict)

    def record_generation(self, k_solved: int) -> None:
        self.generated += 1
        if 0 <= k_solved <= M_ROLLOUTS:
            self.k_histogram[k_solved] += 1

    def record(self, accepted: bool, reason: str | None) -> None:
        self.submitted += 1
        if accepted:
            self.accepted += 1
            self.batched_proof_consecutive_fails = 0
        else:
            self.rejected += 1
            if reason == "superseded":
                self.superseded_in_session += 1
            if reason == "grail_fail" and self.batched_proof_active:
                self.batched_proof_consecutive_fails += 1
        if reason is not None:
            self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def record_local_oos(self) -> None:
        self.local_out_of_zone += 1

    def record_network_error(self) -> None:
        self.network_errors += 1

    def note_proof_fallback(self) -> None:
        self.batched_proof_active = False

    @property
    def in_zone_rate(self) -> float:
        return (
            (self.generated - self.local_out_of_zone) / self.generated * 100.0
            if self.generated else 0.0
        )

    def summary(self, stats: "_PromptStats | None" = None) -> str:
        rate = (self.accepted / self.submitted * 100.0) if self.submitted else 0.0
        top = sorted(self.reasons.items(), key=lambda kv: -kv[1])[:4]
        top_str = ",".join(f"{r}:{c}" for r, c in top) or "-"
        warm_str = ""
        if stats is not None:
            warmed, observed = stats.warmed_count()
            cohort_obs, cohort_cells = stats.cohort_observations()
            warm_str = (
                f" warmed={warmed}/{observed}"
                f" cohorts={cohort_cells}({cohort_obs}obs)"
            )
        proof_str = (
            f" proof=batched"
            if self.batched_proof_active
            else f" proof=per-rollout(fallback)"
        )
        k_str = "/".join(str(c) for c in self.k_histogram)
        return (
            f"generated={self.generated} in_zone={self.in_zone_rate:.1f}% "
            f"submitted={self.submitted} accepted={self.accepted} "
            f"({rate:.1f}%) rejected={self.rejected} "
            f"local_oos={self.local_out_of_zone} net_err={self.network_errors}"
            f"{warm_str}{proof_str} k_hist=[{k_str}] top=[{top_str}]"
        )


# ---------------------------------------------------------------------------
# Checkpoint pull (carried from v3 — same logic, same interface)
# ---------------------------------------------------------------------------

async def maybe_pull_checkpoint(
    state,
    local_n: int,
    local_hash: str,
    local_model,
    *,
    download_fn,
    load_fn,
):
    """If remote checkpoint_n > local, download via HF and load.

    Returns ``(new_local_n, new_local_hash, new_model)``. If no update
    is needed (remote ≤ local, or remote has no repo/revision yet),
    returns inputs unchanged.
    """
    if state.checkpoint_n <= local_n:
        return local_n, local_hash, local_model
    if state.checkpoint_repo_id is None or state.checkpoint_revision is None:
        return local_n, local_hash, local_model
    local_path = await download_fn(state.checkpoint_repo_id, state.checkpoint_revision)
    new_model = load_fn(local_path)
    return state.checkpoint_n, state.checkpoint_revision, new_model


async def _hf_download(repo_id: str, revision: str) -> str:
    """Download a snapshot into the local HF cache and return the model folder path."""
    import asyncio as _asyncio
    from huggingface_hub import snapshot_download

    return await _asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["model.safetensors", "config.json", "tokenizer*"],
    )


# ---------------------------------------------------------------------------
# Posterior + cohort statistics (schema v2)
# ---------------------------------------------------------------------------

class _PromptStats:
    """Beta-Bernoulli posterior with cohort priors and checkpoint versioning.

    Six mappings, all atomically persistable to a single JSON blob:

      _counts[idx]              = (solves, attempts)              per-prompt counts
      _last_checkpoint[idx]     = int                              checkpoint these counts grew on
      _cohort_counts[(L, S)]    = (solves, attempts)              MATH (level, subject) cell
      _cohort_last_checkpoint[(L, S)] = int                        cohort checkpoint tag
      _lengths[idx]             = (mean_len, n_obs)                completion-length tiebreak
      _arrival_ema[idx]         = float ∈ [0, 1]                   "popular among smart miners"
      _superseded_lifetime[idx] = int                              forensic: cumulative SUPERSEDED

    The checkpoint decay is applied **lazily** — posterior() blends the
    stored counts toward the cohort prior on read. This avoids walking
    every entry on every checkpoint bump (could be 10k+ entries).
    """

    # NOTE: alpha_prior / beta_prior must be slots, not bare class
    # attributes, because load_from() overwrites them per-instance and
    # __slots__ otherwise forbids new instance attributes.
    __slots__ = (
        "alpha_prior",
        "beta_prior",
        "_counts",
        "_last_checkpoint",
        "_cohort_counts",
        "_cohort_last_checkpoint",
        "_lengths",
        "_arrival_ema",
        "_superseded_lifetime",
        "_last_cooldown_set",
        "_cohort_cache",
    )

    def __init__(self) -> None:
        self.alpha_prior: float = 1.0
        self.beta_prior: float = 1.0
        self._counts: dict[int, tuple[int, int]] = {}
        self._last_checkpoint: dict[int, int] = {}
        self._cohort_counts: dict[tuple[str, str], tuple[int, int]] = {}
        self._cohort_last_checkpoint: dict[tuple[str, str], int] = {}
        self._lengths: dict[int, tuple[float, int]] = {}
        self._arrival_ema: dict[int, float] = {}
        self._superseded_lifetime: dict[int, int] = {}
        self._last_cooldown_set: frozenset[int] = frozenset()
        # Per-session cache of (level, subject) per prompt_idx so the
        # picker doesn't rebuild the dict on every candidate evaluation.
        self._cohort_cache: dict[int, tuple[str, str]] = {}

    # ------------------------- cohort helpers -------------------------

    @staticmethod
    def _cohort_key(level: str, subject: str) -> tuple[str, str]:
        return (str(level), str(subject))

    def _decayed_cohort(
        self,
        level: str,
        subject: str,
        current_checkpoint_n: int,
    ) -> tuple[float, float]:
        """Cohort counts after lazy checkpoint decay, returned as raw
        (solves_eff, attempts_eff) — both float because decay is fractional.
        """
        key = self._cohort_key(level, subject)
        s, n = self._cohort_counts.get(key, (0, 0))
        if n == 0:
            return (0.0, 0.0)
        last = self._cohort_last_checkpoint.get(key, current_checkpoint_n)
        bumps = max(0, current_checkpoint_n - last)
        if bumps == 0:
            return (float(s), float(n))
        decay = _COHORT_DECAY ** bumps
        return (s * decay, n * decay)

    def _cohort_prior(
        self,
        level: str,
        subject: str,
        current_checkpoint_n: int,
    ) -> tuple[float, float]:
        """Beta prior to use for a fresh prompt in this cohort.

        Below ``_COHORT_MIN_OBS`` observations: uninformative Beta(1, 1).
        Otherwise: cohort empirical rate (Laplace-smoothed) with an ESS
        capped at ``_COHORT_ESS_CAP`` so individual prompts can still
        override the cohort.
        """
        s_eff, n_eff = self._decayed_cohort(level, subject, current_checkpoint_n)
        if n_eff < _COHORT_MIN_OBS:
            return (self.alpha_prior, self.beta_prior)
        # Confidence scaled with log(n). 8 obs → ess=12, 16 → 16, 64+ → 20.
        ess = min(_COHORT_ESS_CAP, math.log2(max(2.0, n_eff)) * 4.0)
        p_hat = (s_eff + 1.0) / (n_eff + 2.0)
        alpha = ess * p_hat + self.alpha_prior
        beta = ess * (1.0 - p_hat) + self.beta_prior
        return (alpha, beta)

    # ------------------------- posterior API -------------------------

    def posterior(
        self,
        idx: int,
        current_checkpoint_n: int,
        level: str,
        subject: str,
    ) -> tuple[float, float]:
        """Beta posterior with cohort prior + lazy checkpoint decay.

        - Unseen prompt → cohort prior directly.
        - Seen prompt, same checkpoint → standard Beta posterior.
        - Seen prompt, K checkpoint bumps ago → blend toward cohort prior
          with factor ``_CHECKPOINT_DECAY ** K``.
        """
        s, n = self._counts.get(idx, (0, 0))
        if n == 0:
            return self._cohort_prior(level, subject, current_checkpoint_n)

        last_ckpt = self._last_checkpoint.get(idx, current_checkpoint_n)
        bumps = max(0, current_checkpoint_n - last_ckpt)
        alpha_cur = self.alpha_prior + s
        beta_cur = self.beta_prior + (n - s)
        if bumps == 0:
            return (alpha_cur, beta_cur)

        decay = _CHECKPOINT_DECAY ** bumps
        alpha_p, beta_p = self._cohort_prior(level, subject, current_checkpoint_n)
        return (
            alpha_p + decay * (alpha_cur - alpha_p),
            beta_p + decay * (beta_cur - beta_p),
        )

    def sample_p(
        self,
        idx: int,
        current_checkpoint_n: int,
        level: str,
        subject: str,
        rng: _random.Random,
    ) -> float:
        a, b = self.posterior(idx, current_checkpoint_n, level, subject)
        return rng.betavariate(a, b)

    def mean_p(
        self,
        idx: int,
        current_checkpoint_n: int,
        level: str,
        subject: str,
    ) -> float:
        a, b = self.posterior(idx, current_checkpoint_n, level, subject)
        return a / (a + b)

    def attempts(self, idx: int) -> int:
        _, n = self._counts.get(idx, (0, 0))
        return n

    def arrival_rate(self, idx: int) -> float:
        return self._arrival_ema.get(idx, 0.0)

    def avg_completion_len(self, idx: int) -> float | None:
        v = self._lengths.get(idx)
        return v[0] if v else None

    # ------------------------- updates -------------------------

    def record_group(
        self,
        prompt_idx: int,
        rewards: list[float],
        *,
        level: str,
        subject: str,
        checkpoint_n: int,
        completion_lens: list[int] | None = None,
    ) -> None:
        """Update per-prompt counts, the cohort cell, and length stats.

        Critical: BOTH the prompt and the cohort are tagged with the
        current ``checkpoint_n`` so the lazy decay starts from "now"
        rather than retroactively decaying just-collected evidence.
        """
        attempts = len(rewards)
        if attempts == 0:
            return
        solves = sum(1 for r in rewards if r >= _SOLVED_THRESHOLD)

        # Per-prompt
        s_prev, n_prev = self._counts.get(prompt_idx, (0, 0))
        self._counts[prompt_idx] = (s_prev + solves, n_prev + attempts)
        self._last_checkpoint[prompt_idx] = checkpoint_n

        # Cohort
        key = self._cohort_key(level, subject)
        cs, cn = self._cohort_counts.get(key, (0, 0))
        self._cohort_counts[key] = (cs + solves, cn + attempts)
        self._cohort_last_checkpoint[key] = checkpoint_n

        # Cohort cache for picker hot path
        self._cohort_cache[prompt_idx] = (str(level), str(subject))

        # Lengths
        if completion_lens:
            prev_mean, prev_n = self._lengths.get(prompt_idx, (0.0, 0))
            total_n = prev_n + len(completion_lens)
            new_mean = (prev_mean * prev_n + sum(completion_lens)) / total_n
            self._lengths[prompt_idx] = (new_mean, total_n)

    def record_cooldown_diff(self, new_cooldown_set: set[int] | list[int]) -> None:
        """Update arrival_ema from prompts that newly entered cooldown.

        Newly entered → bump EMA toward 1.0 (someone won this prompt
        recently).  All other tracked prompts get a slow background
        decay so quiescent ones return to zero and stop influencing
        the picker. Pruned below ``_ARRIVAL_PRUNE_BELOW`` to bound dict
        size for 12 500-prompt envs.
        """
        new = frozenset(new_cooldown_set)
        newly_entered = new - self._last_cooldown_set
        for idx in newly_entered:
            old = self._arrival_ema.get(idx, 0.0)
            self._arrival_ema[idx] = old + _ARRIVAL_EMA_ALPHA * (1.0 - old)

        # Background decay; iterate over a snapshot of keys because we
        # delete from the dict in-loop.
        for idx in list(self._arrival_ema.keys()):
            if idx in newly_entered:
                continue
            self._arrival_ema[idx] *= _ARRIVAL_QUIESCENT_DECAY
            if self._arrival_ema[idx] < _ARRIVAL_PRUNE_BELOW:
                del self._arrival_ema[idx]

        self._last_cooldown_set = new

    def record_superseded(self, prompt_idx: int) -> None:
        self._superseded_lifetime[prompt_idx] = (
            self._superseded_lifetime.get(prompt_idx, 0) + 1
        )

    def cache_cohort(self, prompt_idx: int, level: str, subject: str) -> None:
        """Fill the per-session ``(level, subject)`` cache without recording
        a group. Called by the picker so cohort lookups stay O(1).
        """
        self._cohort_cache[prompt_idx] = (str(level), str(subject))

    def cached_cohort(self, prompt_idx: int) -> tuple[str, str] | None:
        return self._cohort_cache.get(prompt_idx)

    # ------------------------- diagnostics -------------------------

    def warmed_count(self, min_attempts: int = 5) -> tuple[int, int]:
        observed = len(self._counts)
        warmed = sum(
            1 for _solves, n in self._counts.values() if n >= min_attempts
        )
        return warmed, observed

    def cohort_observations(self) -> tuple[int, int]:
        total = sum(n for _, n in self._cohort_counts.values())
        return total, len(self._cohort_counts)

    # ------------------------- persistence -------------------------

    def save_to(self, path: str) -> None:
        """Atomically persist all maps as schema-v2 JSON.

        Atomic-write pattern (``write to .tmp`` → ``os.replace``)
        guarantees no torn file on a hard kill. Cohort cache is NOT
        persisted — it's session-scoped and trivially rebuilds.
        """
        import json
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "schema": _STATS_SCHEMA,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "counts": {str(k): list(v) for k, v in self._counts.items()},
            "last_checkpoint": {str(k): v for k, v in self._last_checkpoint.items()},
            "cohort_counts": {
                f"{level}\t{subject}": list(v)
                for (level, subject), v in self._cohort_counts.items()
            },
            "cohort_last_checkpoint": {
                f"{level}\t{subject}": v
                for (level, subject), v in self._cohort_last_checkpoint.items()
            },
            "lengths": {str(k): list(v) for k, v in self._lengths.items()},
            "arrival_ema": {str(k): v for k, v in self._arrival_ema.items()},
            "superseded_lifetime": {
                str(k): v for k, v in self._superseded_lifetime.items()
            },
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    def load_from(self, path: str) -> bool:
        """Load schema v1 or v2; silently accept missing/corrupt files."""
        import json
        import os

        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, ValueError):
            return False

        # Shared fields
        self.alpha_prior = float(payload.get("alpha_prior", self.alpha_prior))
        self.beta_prior = float(payload.get("beta_prior", self.beta_prior))
        self._counts = {
            int(k): (int(v[0]), int(v[1]))
            for k, v in (payload.get("counts") or {}).items()
        }
        self._lengths = {
            int(k): (float(v[0]), int(v[1]))
            for k, v in (payload.get("lengths") or {}).items()
        }

        schema = int(payload.get("schema", 1))
        if schema >= 2:
            self._last_checkpoint = {
                int(k): int(v)
                for k, v in (payload.get("last_checkpoint") or {}).items()
            }

            def _parse_cohort(d):
                out = {}
                for k, v in (d or {}).items():
                    if "\t" in k:
                        level, subject = k.split("\t", 1)
                    else:
                        level, subject = k, ""
                    out[(level, subject)] = v
                return out

            raw_cc = _parse_cohort(payload.get("cohort_counts"))
            self._cohort_counts = {
                key: (int(v[0]), int(v[1])) for key, v in raw_cc.items()
            }
            raw_clc = _parse_cohort(payload.get("cohort_last_checkpoint"))
            self._cohort_last_checkpoint = {
                key: int(v) for key, v in raw_clc.items()
            }
            self._arrival_ema = {
                int(k): float(v)
                for k, v in (payload.get("arrival_ema") or {}).items()
            }
            self._superseded_lifetime = {
                int(k): int(v)
                for k, v in (payload.get("superseded_lifetime") or {}).items()
            }
        else:
            # v1 → v2 migration: keep prompt counts, zero everything else.
            # last_checkpoint defaults to 0 so the lazy decay treats the
            # data as already a few checkpoints stale (conservative).
            self._last_checkpoint = {k: 0 for k in self._counts}
        return True


# ---------------------------------------------------------------------------
# Prompt picker — Thompson with cohort priors, congestion, blacklist
# ---------------------------------------------------------------------------

def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    stats: _PromptStats | None = None,
    current_checkpoint_n: int = 0,
    slots_filled: int = 0,
    superseded_in_window: set[int] | None = None,
    candidates: int = _CANDIDATES_DEFAULT,
    max_attempts: int = 1000,
) -> int:
    """Best-of-K Thompson-sampled prompt selection.

    Scoring per candidate ``idx``::

        a, b      = stats.posterior(idx, ckpt_n, level, subject)
                    # cohort prior used when prompt is unseen
                    # or counts are from a stale checkpoint
        p_sample  = rng.betavariate(a, b)
        zone_p    = P(K_LO ≤ Binomial(M, p_sample) ≤ K_HI)
        arrival_z = stats.arrival_rate(idx)       ∈ [0, 1]
        score     = zone_p * (1 - _CONGESTION_WEIGHT * arrival_z)

    Tiebreak: shorter average completion length (faster generation →
    earlier TCP arrival → more SUPERSEDED wins).

    Race mode (slots_filled ≥ _RACE_MODE_SLOTS_THRESHOLD): K halves.

    Hard blacklist: any prompt in ``superseded_in_window`` is skipped
    — the race for that slot is over.

    Falls back to uniform-random with cooldown skip when ``stats`` is
    None (test path / first-ever pick before posterior loads).
    """
    rng = rng or _random
    superseded_in_window = superseded_in_window or set()
    n = len(env)
    blocked = cooldown_prompts | superseded_in_window
    over_half_blocked = len(blocked) >= n / 2

    eligible_list: list[int] | None = None
    if over_half_blocked:
        eligible_list = [i for i in range(n) if i not in blocked]
        if not eligible_list:
            raise RuntimeError("no eligible prompt — env fully blocked")

    def _draw_one() -> int | None:
        if eligible_list is not None:
            return rng.choice(eligible_list)
        for _ in range(max_attempts):
            idx = rng.randrange(n)
            if idx not in blocked:
                return idx
        return None

    if stats is None:
        idx = _draw_one()
        if idx is None:
            raise RuntimeError("no eligible prompt — env fully blocked")
        return idx

    K = (
        _CANDIDATES_RACE
        if slots_filled >= _RACE_MODE_SLOTS_THRESHOLD
        else candidates
    )

    seen: set[int] = set()
    best_idx: int | None = None
    best_score: float = -1.0
    best_len: float = float("inf")

    for _ in range(K):
        idx = _draw_one()
        if idx is None:
            break
        if idx in seen:
            continue
        seen.add(idx)

        cohort = stats.cached_cohort(idx)
        if cohort is None:
            try:
                problem = env.get_problem(idx)
                level = problem.get("level", "")
                subject = problem.get("subject", "")
            except Exception:
                level = subject = ""
            stats.cache_cohort(idx, level, subject)
        else:
            level, subject = cohort

        p_sampled = stats.sample_p(idx, current_checkpoint_n, level, subject, rng)
        zone_p = _zone_probability(p_sampled)
        arrival_z = stats.arrival_rate(idx)
        score = zone_p * (1.0 - _CONGESTION_WEIGHT * arrival_z)

        if score > best_score:
            best_idx = idx
            best_score = score
            best_len = stats.avg_completion_len(idx) or float("inf")
        elif score == best_score:
            alt_len = stats.avg_completion_len(idx) or float("inf")
            if alt_len < best_len:
                best_idx = idx
                best_len = alt_len

    if best_idx is None:
        raise RuntimeError("no eligible prompt")
    return best_idx


# ---------------------------------------------------------------------------
# Merkle root (carried verbatim from v3)
# ---------------------------------------------------------------------------

def _compute_merkle_root(rollouts) -> str:
    """SHA-256 Merkle root over (idx, tokens, reward, commit) leaves.

    Canonical JSON (``sort_keys=True, separators=(",", ":")``) makes the
    root deterministic across Python implementations and refactor-stable
    against dict-construction-order changes.
    """
    import hashlib
    import json

    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(json.dumps(r.tokens, separators=(",", ":")).encode())
        h.update(json.dumps(r.reward).encode())
        h.update(json.dumps(r.commit, sort_keys=True, separators=(",", ":")).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        new = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            new.append(hashlib.sha256(left + right).digest())
        leaves = new
    return leaves[0].hex()


# ---------------------------------------------------------------------------
# MiningEngine v4
# ---------------------------------------------------------------------------

class MiningEngine:
    """v4: serial pipeline + cohort-aware picker + batched GRAIL proof.

    Constructor signature is identical to v3 (and v2, modulo the
    ``stats_path`` kwarg). Pass either an HF ``AutoModelForCausalLM`` or
    a ``VLLMAdapter`` as ``vllm_model``; the engine detects the adapter
    via the ``_is_vllm_adapter`` sentinel in ``_load_checkpoint``.

    ``batched_proof=True`` enables the single-pass GRAIL forward. If two
    consecutive batched submissions return GRAIL_FAIL, the engine
    automatically falls back to the per-rollout path and stays there
    for the rest of the session.
    """

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment",
        *,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override: str | None = None,
        stats_path: str | None = _DEFAULT_STATS_PATH,
        batched_proof: bool = True,
    ) -> None:
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.env = env
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self.validator_url_override = validator_url_override

        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

        # Per-window randomness cache — re-derived only when window_n
        # changes. The HTTP /state poll runs faster than chain calls so
        # this saves ~80% of the chain round-trips.
        self._cached_window_n: int | None = None
        self._cached_randomness: str = ""

        # Posterior + cohort + arrival/superseded stats.
        self._prompt_stats = _PromptStats()
        self._stats_path = stats_path
        if self._stats_path:
            try:
                if self._prompt_stats.load_from(self._stats_path):
                    warmed, observed = self._prompt_stats.warmed_count()
                    cohort_obs, cohort_cells = (
                        self._prompt_stats.cohort_observations()
                    )
                    logger.info(
                        "loaded prompt stats from %s: observed=%d warmed=%d "
                        "cohorts=%d (obs=%d)",
                        self._stats_path, observed, warmed,
                        cohort_cells, cohort_obs,
                    )
            except Exception:
                logger.exception(
                    "failed to load prompt stats from %s; starting fresh",
                    self._stats_path,
                )
        self._save_counter: int = 0

        self._metrics = _MinerMetrics()
        self._metrics.batched_proof_active = batched_proof
        # Track which prompts already failed SUPERSEDED in the current
        # window — never retry them in the same window.
        self._superseded_in_window: set[int] = set()

        # Resolve the FULL set of EOS / stop tokens the model may emit.
        # ``tokenizer.eos_token_id`` is a single int but Qwen3 chat
        # models stop on EITHER ``<|im_end|>`` (151645) OR
        # ``<|endoftext|>`` (151643). vLLM honors all entries in
        # ``generation_config.eos_token_id`` (which is a list for Qwen3);
        # we MUST match that, otherwise rollouts ending on the secondary
        # EOS never get truncated and trailing pad tokens (==
        # ``<|endoftext|>`` == 151643 for Qwen3) survive into the
        # submission. That's the silent root cause of the
        # ``clen=max/max/max`` symptom + window_mismatch rejections.
        self._eos_ids: set[int] = self._resolve_eos_ids()
        # Flag: in the legacy single-EOS configuration we also strip
        # trailing pad tokens defensively. Computed once.
        self._pad_id: int | None = getattr(
            self.tokenizer, "pad_token_id", None,
        )
        logger.info(
            "engine eos resolution: eos_ids=%s pad_id=%s",
            sorted(self._eos_ids), self._pad_id,
        )

        # One-time logger reseat done lazily on first ``mine_window``
        # iteration after all transitive imports (bittensor / submitter)
        # have completed. See ``_one_time_logger_reseat``.
        self._logger_reseated: bool = False

    def _resolve_eos_ids(self) -> set[int]:
        """Collect every token ID the model treats as a stop token.

        Sources, in order of authority:

        1. ``tokenizer.eos_token_id`` — the canonical primary EOS.
        2. ``hf_model.generation_config.eos_token_id`` — the full list
           the model was trained to stop on. For Qwen3 chat models this
           contains both ``<|im_end|>`` and ``<|endoftext|>``.
        3. ``vllm_model.generation_config.eos_token_id`` — vLLM caches
           the same value but exposing through a different attribute
           path on the adapter; checked as a fallback.

        Returns a non-empty set. If everything fails (no tokenizer, no
        generation_config), falls back to ``{tokenizer.eos_token_id}``
        and logs a warning so the operator knows the truncation may
        miss multi-EOS models.
        """
        ids: set[int] = set()
        eid = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eid, int):
            ids.add(eid)
        for src in (self.hf_model, self.vllm_model):
            gc = getattr(src, "generation_config", None)
            if gc is None:
                continue
            gc_eos = getattr(gc, "eos_token_id", None)
            if isinstance(gc_eos, (list, tuple)):
                for x in gc_eos:
                    if isinstance(x, int):
                        ids.add(x)
            elif isinstance(gc_eos, int):
                ids.add(gc_eos)
        if not ids:
            logger.warning(
                "could not resolve any EOS token from tokenizer or "
                "generation_config; EOS-based truncation will likely "
                "fail (you'll see clen=max/max/max in GEN logs)",
            )
        return ids

    def _one_time_logger_reseat(self) -> None:
        """Strip handlers from the loggers we care about, once per process.

        Called from ``mine_window`` AFTER all transitive lazy imports
        (httpx, reliquary.miner.submitter — which transitively imports
        bittensor.btlogging) have completed. Any of those imports can:

          1. Reset the root handler list (bittensor's btlogging
             reconfigures the root logger on import).
          2. Attach a fresh StreamHandler directly to a NAMED logger
             (e.g. ``"bittensor"`` or ``"reliquary.miner.submitter"``).

        Case (2) is the silent cause of the "logs appear 2x then 3x
        over a long session" pattern: with ``propagate=True`` the
        emission fires once on the named handler AND once again via
        the root handler. Each subsequent clobber-recover cycle adds
        another handler.

        Idempotent — guarded by ``self._logger_reseated`` so we only
        do it once per ``mine_window`` lifecycle. The expensive part
        (handler enumeration) is microseconds, but keeping it once
        avoids ever-so-slightly mutating the global logging state on
        every poll loop tick.
        """
        if self._logger_reseated:
            return
        self._logger_reseated = True

        # Mirror of main-v4's _PINNED_RELIQUARY_LOGGERS plus a couple of
        # extras we know third-party imports may have polluted. Kept
        # local to engine so we don't introduce a circular import.
        pinned = (
            "reliquary",
            "reliquary.miner.engine",
            "reliquary.miner.submitter",
            "reliquary.infrastructure.chain",
            "reliquary.infrastructure.drand",
            "reliquary.cli",
            "vllm_adapter",
            "bittensor",
            "bittensor.core",
            "btlogging",
        )
        stripped = 0
        for name in pinned:
            _lg = logging.getLogger(name)
            n = len(_lg.handlers)
            if n:
                for h in list(_lg.handlers):
                    _lg.removeHandler(h)
                stripped += n
            _lg.propagate = True
            _lg.disabled = False
        root = logging.getLogger()
        _emit(
            logging.INFO,
            "[engine.v4] logger reseat: stripped=%d named-logger handlers, "
            "root_handlers=%d",
            stripped, len(root.handlers),
        )

    def _maybe_persist_stats(self) -> None:
        """Flush stats every ``_SUMMARY_EVERY`` writes (atomic-replace)."""
        if not self._stats_path:
            return
        self._save_counter += 1
        if self._save_counter % _SUMMARY_EVERY != 0:
            return
        try:
            self._prompt_stats.save_to(self._stats_path)
        except Exception:
            logger.exception(
                "failed to persist prompt stats to %s; continuing",
                self._stats_path,
            )

    # ------------------------------------------------------------------
    # Public API — same shape as v3.mine_window
    # ------------------------------------------------------------------

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,
        use_drand: bool = True,
    ) -> list:
        """Continuous poll-submit loop. Cancels on asyncio cancel or env-empty.

        The outer try/except scaffolding mirrors v3 so existing
        ``main.py`` / ``launcher.py`` callers don't need to change.
        """
        import httpx
        import random

        from reliquary.constants import POLL_INTERVAL_SECONDS
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
            get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import (
            BatchSubmissionRequest, WindowState,
        )

        # One-time logger reseat AFTER the submitter / submission /
        # bittensor-transitive imports above complete. Any of these
        # imports can re-clobber the root handler or attach a fresh
        # handler to a named logger; reseating here re-takes ownership
        # and strips per-logger handlers (which is what produces the
        # 2× / 3× duplicate log lines observed in long sessions).
        self._one_time_logger_reseat()

        # Early sentinel via both channels so the operator sees we
        # entered the main loop even if the logger is clobbered later.
        _emit(
            logging.INFO,
            "[engine.v4] mine_window entered backend=%s wallet=%s M=%d T=%.2f "
            "batched_proof=%s pid=%d",
            type(self.vllm_model).__name__,
            self.wallet.hotkey.ss58_address[:12],
            M_ROLLOUTS, T_PROTO,
            self._metrics.batched_proof_active,
            os.getpid(),
        )

        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)
        _emit(
            logging.INFO,
            "[engine.v4] validator url=%s — entering poll/submit loop",
            url,
        )

        rng = random.Random()
        results = []
        local_n = 0
        local_hash = ""
        last_window_n: int | None = None

        # Generous HTTP timeouts. Reasoning per leg:
        #   - connect=10s: long enough to absorb transient DNS / TCP
        #     handshake hiccups but quickly surfaces a dead validator.
        #   - read=90s: validator's GRAIL verification can take 30-60 s
        #     under heavy load (batched_proof_active=True ships 8
        #     rollouts × up to 8192 tokens). The default 30s timeout
        #     was producing window_mismatch rejections on perfectly
        #     good submissions when the validator was slow.
        #   - write=30s: upload of the ~MB-size proof payload.
        #   - pool=10s: connection-pool wait.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0, read=90.0, write=30.0, pool=10.0,
            ),
        ) as client:
            while True:
                try:
                    state = await get_window_state_v2(url, client=client)
                except SubmissionError:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                except Exception as e:
                    logger.debug("state fetch failed: %s", e)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # Checkpoint pull (no-op when remote ≤ local).
                if state.checkpoint_n > local_n and state.checkpoint_revision:
                    logger.info(
                        "checkpoint pull: local_n=%d → remote_n=%d revision=%s",
                        local_n, state.checkpoint_n,
                        (state.checkpoint_revision or "")[:12],
                    )
                try:
                    local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
                        state=state, local_n=local_n, local_hash=local_hash,
                        local_model=self.hf_model,
                        download_fn=_hf_download,
                        load_fn=self._load_checkpoint,
                    )
                except Exception:
                    logger.exception("checkpoint pull failed; keeping local")

                if state.state != WindowState.OPEN:
                    await asyncio.sleep(1)
                    continue

                # Window-edge bookkeeping.
                if last_window_n != state.window_n:
                    if last_window_n is not None:
                        _emit(
                            logging.INFO,
                            "=== window %d -> %d === | %s",
                            last_window_n, state.window_n,
                            self._metrics.summary(self._prompt_stats),
                        )
                    else:
                        _emit(
                            logging.INFO,
                            "=== window %d (first OPEN) === valid=%d/8 "
                            "cooldown=%d ckpt=%d",
                            state.window_n, state.valid_submissions,
                            len(state.cooldown_prompts), state.checkpoint_n,
                        )
                    last_window_n = state.window_n
                    self._superseded_in_window = set()
                    # Update arrival-rate EMA from the new cooldown delta.
                    self._prompt_stats.record_cooldown_diff(
                        set(state.cooldown_prompts)
                    )

                try:
                    randomness = await self._randomness_for_window(
                        subtensor, state.window_n, use_drand
                    )
                except Exception:
                    logger.exception(
                        "failed to derive randomness for window %d; retrying",
                        state.window_n,
                    )
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # Pick.
                cooldown_set = set(state.cooldown_prompts)
                try:
                    prompt_idx = pick_prompt_idx(
                        self.env, cooldown_set,
                        rng=rng, stats=self._prompt_stats,
                        current_checkpoint_n=state.checkpoint_n,
                        slots_filled=state.valid_submissions,
                        superseded_in_window=self._superseded_in_window,
                    )
                except RuntimeError:
                    _emit(
                        logging.INFO,
                        "[W=%d] WAIT env fully blocked cooldown=%d superseded=%d "
                        "env=%d; sleeping 5s",
                        state.window_n, len(cooldown_set),
                        len(self._superseded_in_window), len(self.env),
                    )
                    await asyncio.sleep(5)
                    continue

                problem = self.env.get_problem(prompt_idx)
                level = problem.get("level", "")
                subject = problem.get("subject", "")
                a, b = self._prompt_stats.posterior(
                    prompt_idx, state.checkpoint_n, level, subject,
                )
                p_hat = a / (a + b)
                attempts = self._prompt_stats.attempts(prompt_idx)
                arr = self._prompt_stats.arrival_rate(prompt_idx)
                _emit(
                    logging.INFO,
                    "[W=%d] PICK prompt=%-4d cohort=(%s,%s) phat=%.2f "
                    "attempts=%d arr=%.2f zone_p=%.2f slots=%d/8",
                    state.window_n, prompt_idx,
                    _short_level(level), _short_subject(subject),
                    p_hat, attempts, arr, _zone_probability(p_hat),
                    state.valid_submissions,
                )
                logger.debug(
                    "[W=%d] PICK detail prompt=%d posterior=Beta(%.2f,%.2f) "
                    "level=%r subject=%r",
                    state.window_n, prompt_idx, a, b, level, subject,
                )

                # Generate.
                t_gen = time.monotonic()
                generations = self._generate_n_rollouts(problem, M_ROLLOUTS)
                gen_ms = (time.monotonic() - t_gen) * 1000.0
                if len(generations) < M_ROLLOUTS:
                    _emit(
                        logging.WARNING,
                        "[W=%d] GEN  prompt=%-4d short: only %d/%d rollouts "
                        "(gen=%.1fs) -> skip",
                        state.window_n, prompt_idx,
                        len(generations), M_ROLLOUTS, gen_ms / 1000.0,
                    )
                    continue

                # Score (decode + reward) without building proofs yet.
                t_score = time.monotonic()
                scored = [self._score_rollout(g, problem) for g in generations]
                score_ms = (time.monotonic() - t_score) * 1000.0
                rewards = [r for r, _ in scored]
                completion_lens = [length for _, length in scored]

                # Update posterior + cohort BEFORE the proof/submit round
                # so the signal lands even if the submit fails.
                self._prompt_stats.record_group(
                    prompt_idx, rewards,
                    level=level, subject=subject,
                    checkpoint_n=state.checkpoint_n,
                    completion_lens=completion_lens,
                )
                sigma, k_solved, in_zone = _zone_status(rewards)
                self._metrics.record_generation(k_solved)
                self._maybe_persist_stats()
                _emit(
                    logging.INFO,
                    "[W=%d] GEN  prompt=%-4d rewards=%s k=%d/%d sigma=%.3f "
                    "in_zone=%s gen=%.1fs score=%.2fs clen=%d/%d/%d",
                    state.window_n, prompt_idx,
                    _fmt_rewards(rewards), k_solved, M_ROLLOUTS, sigma,
                    in_zone, gen_ms / 1000.0, score_ms / 1000.0,
                    min(completion_lens),
                    sorted(completion_lens)[len(completion_lens) // 2],
                    max(completion_lens),
                )

                # Local OUT_OF_ZONE short-circuit.
                if not in_zone:
                    self._metrics.record_local_oos()
                    _emit(
                        logging.INFO,
                        "[W=%d] OOZ  prompt=%-4d sigma=%.3f<%.2f k=%d/%d -> skip submit",
                        state.window_n, prompt_idx, sigma, SIGMA_MIN,
                        k_solved, M_ROLLOUTS,
                    )
                    if self._metrics.generated % _SUMMARY_EVERY == 0:
                        _emit(
                            logging.INFO,
                            "=== SUMMARY === %s",
                            self._metrics.summary(self._prompt_stats),
                        )
                    continue

                # Build GRAIL proofs.
                t_proof = time.monotonic()
                try:
                    rollout_submissions = self._build_rollout_submissions(
                        generations, rewards, randomness,
                    )
                except Exception as proof_exc:
                    _emit(
                        logging.ERROR,
                        "[W=%d] ERR  prompt=%-4d proof construction failed: %s — "
                        "skipping submit",
                        state.window_n, prompt_idx, proof_exc,
                    )
                    logger.exception("proof construction traceback:")
                    continue
                proof_ms = (time.monotonic() - t_proof) * 1000.0
                logger.debug(
                    "[W=%d] proof done prompt=%d proof_ms=%.0f mode=%s",
                    state.window_n, prompt_idx, proof_ms,
                    "batched" if self._metrics.batched_proof_active else "per-rollout",
                )

                merkle_root = _compute_merkle_root(rollout_submissions)

                # Pre-submit /state recheck — abort doomed submits when
                # the window rolled over during generation/proof.
                try:
                    fresh_state = await get_window_state_v2(url, client=client)
                    if fresh_state.window_n != state.window_n:
                        _emit(
                            logging.WARNING,
                            "[W=%d] SKIP prompt=%-4d window advanced to %d "
                            "during build (gen=%.1fs proof=%.1fs)",
                            state.window_n, prompt_idx, fresh_state.window_n,
                            gen_ms / 1000.0, proof_ms / 1000.0,
                        )
                        continue
                except SubmissionError as e:
                    logger.debug(
                        "pre-submit state check failed: %s; proceeding",
                        e,
                    )

                request = BatchSubmissionRequest(
                    miner_hotkey=self.wallet.hotkey.ss58_address,
                    prompt_idx=prompt_idx,
                    window_start=state.window_n,
                    merkle_root=merkle_root,
                    rollouts=rollout_submissions,
                    checkpoint_hash=local_hash,
                )
                # GPU mem snapshot BEFORE submit. After-snapshot is
                # embedded in the SUB line so the operator can compare
                # `gpu=A/B` (pre) to `gpu=C/B` (post). A C > A trend
                # across many submissions = the memory leak we
                # suspected when vLLM KV cache stayed allocated for
                # over-long padded sequences.
                gpu_pre = _gpu_mem_compact(self.vllm_gpu)
                t_http = time.monotonic()
                try:
                    resp = await submit_batch_v2(url, request, client=client)
                    http_ms = (time.monotonic() - t_http) * 1000.0
                    gpu_post = _gpu_mem_compact(self.vllm_gpu)
                    reason_str = (
                        resp.reason.value if hasattr(resp.reason, "value")
                        else str(resp.reason)
                    )
                    self._metrics.record(resp.accepted, reason_str)
                    if reason_str == "superseded":
                        self._superseded_in_window.add(prompt_idx)
                        self._prompt_stats.record_superseded(prompt_idx)
                    self._maybe_check_proof_fallback()
                    status = "ACCEPTED" if resp.accepted else "REJECTED"
                    proof_mode = (
                        "batched" if self._metrics.batched_proof_active
                        else "per-rollout"
                    )
                    sub_level = logging.INFO if resp.accepted else logging.WARNING
                    _emit(
                        sub_level,
                        "[W=%d] SUB  prompt=%-4d rewards=%s k=%d/%d "
                        "sigma=%.3f merkle=%s proof=%s gen=%.1fs proof=%.1fs "
                        "http=%.2fs gpu=%s->%s -> %s reason=%s",
                        state.window_n, prompt_idx,
                        _fmt_rewards(rewards), k_solved, M_ROLLOUTS, sigma,
                        merkle_root[:8], proof_mode,
                        gen_ms / 1000.0, proof_ms / 1000.0, http_ms / 1000.0,
                        gpu_pre, gpu_post,
                        status, reason_str,
                    )
                    results.append(resp)
                except SubmissionError as exc:
                    self._metrics.record_network_error()
                    _emit(
                        logging.ERROR,
                        "[W=%d] ERR  prompt=%-4d submit failed: %s",
                        state.window_n, prompt_idx, exc,
                    )

                if self._metrics.submitted and self._metrics.submitted % _SUMMARY_EVERY == 0:
                    _emit(
                        logging.INFO,
                        "=== SUMMARY === %s",
                        self._metrics.summary(self._prompt_stats),
                    )

        return results

    # ------------------------------------------------------------------
    # Proof-mode self-healing
    # ------------------------------------------------------------------

    def _maybe_check_proof_fallback(self) -> None:
        """Switch to per-rollout proofs after consecutive batched failures.

        Triggered by ``_metrics.batched_proof_consecutive_fails`` crossing
        ``_BATCHED_PROOF_FAIL_THRESHOLD``. Once tripped, never re-enabled
        in the same session — too risky to flip back and lose more
        submissions to a transient batched-vs-validator divergence.
        """
        if not self._metrics.batched_proof_active:
            return
        if (
            self._metrics.batched_proof_consecutive_fails
            >= _BATCHED_PROOF_FAIL_THRESHOLD
        ):
            logger.warning(
                "BATCHED-PROOF FALLBACK TRIGGERED: %d consecutive GRAIL_FAIL "
                "after batched submissions. Switching to per-rollout proofs "
                "for the rest of this session. Check sketch_diff vs "
                "PROOF_SKETCH_TOLERANCE_BASE / cross-GPU drift.",
                self._metrics.batched_proof_consecutive_fails,
            )
            self._metrics.note_proof_fallback()

    # ------------------------------------------------------------------
    # Checkpoint reload (carried verbatim from v3; vLLM adapter aware)
    # ------------------------------------------------------------------

    def _load_checkpoint(self, local_path: str):
        """Reload both hf_model and vllm_model from *local_path*.

        Adapter path (``vllm_model._is_vllm_adapter``): the adapter's own
        ``reload()`` rebuilds the vLLM ``LLM`` in-place. HF fallback path:
        ``AutoModelForCausalLM.from_pretrained`` rebuilds an HF generator.
        Both paths reload the proof model identically.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from reliquary.constants import ATTN_IMPLEMENTATION

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        try:
            new_hf = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.proof_gpu}").eval()
        except Exception:
            logger.exception(
                "Failed to reload hf_model from %s; keeping old model",
                local_path,
            )
            return self.hf_model

        old_hf = self.hf_model
        self.hf_model = new_hf
        del old_hf
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        if getattr(self.vllm_model, "_is_vllm_adapter", False):
            try:
                self.vllm_model.reload(local_path)
            except Exception as e:
                logger.exception(
                    "vLLM reload failed for %s; falling back to HF generation. Err: %s",
                    local_path, e,
                )
                try:
                    new_gen = AutoModelForCausalLM.from_pretrained(
                        local_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=ATTN_IMPLEMENTATION,
                    ).to(f"cuda:{self.vllm_gpu}").eval()
                    self.vllm_model = new_gen
                    logger.warning("vLLM fallback complete; using HF generation (slower)")
                except Exception as e2:
                    logger.exception(
                        "vLLM fallback also failed: %s. Miner generation is BROKEN.",
                        e2,
                    )
                    raise RuntimeError(
                        f"Both vLLM reload and HF fallback failed for {local_path}"
                    ) from e
                self._loaded_checkpoint_path = local_path
                return self.hf_model
        else:
            try:
                new_gen = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=ATTN_IMPLEMENTATION,
                ).to(f"cuda:{self.vllm_gpu}").eval()
            except Exception:
                logger.exception(
                    "Failed to reload vllm_model from %s; miner generation is "
                    "BROKEN until the next successful pull.",
                    local_path,
                )
                self.vllm_model = None
                self._loaded_checkpoint_path = None
                return self.hf_model

            old_gen = self.vllm_model
            self.vllm_model = new_gen
            del old_gen
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        self._loaded_checkpoint_path = local_path
        logger.info("Checkpoint %s loaded into both models", local_path)
        return self.hf_model

    # ------------------------------------------------------------------
    # Generation (carried from v3)
    # ------------------------------------------------------------------

    def _generate_n_rollouts(self, problem, n: int) -> list[dict]:
        """Run ``n`` independent samples in one batched ``.generate()`` call.

        Output rows are truncated at the first post-prompt EOS so trailing
        pad tokens (HF pads with ``pad_token_id = eos_token_id`` by default)
        don't show up in the submission.
        """
        import torch

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        # Dynamic max_new_tokens — verifier accepts max-length termination
        # iff prompt_length + completion_length >= MAX_NEW_TOKENS_PROTOCOL_CAP.
        budget = max(
            512,
            MAX_NEW_TOKENS_PROTOCOL_CAP - prompt_length,
        )
        effective_max_new = min(self.max_new_tokens, budget)

        with torch.no_grad():
            device_str = getattr(self.vllm_model, "device", "cpu")
            device = torch.device(device_str)
            input_tensor = torch.tensor(
                [prompt_tokens] * n,
                dtype=torch.long,
                device=device,
            )
            outputs = self.vllm_model.generate(
                input_tensor,
                max_new_tokens=effective_max_new,
                do_sample=True,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=TOP_K_PROTO,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Truncate each row at the first occurrence of ANY known stop
        # token. Walking from the front (not stripping pads from the
        # back) handles both:
        #   - pad_id != any eos_id: pads are skipped past until the real
        #     eos token shows up; everything from eos onward is dropped.
        #   - pad_id == an eos_id (Qwen3: pad_id=151643 IS an EOS):
        #     stripping pads from the back would eat the real eos; but
        #     scanning from the front finds the model's real eos first
        #     (since vLLM emits it before any padding starts).
        # For finish_reason='length' rollouts (no eos in real tokens),
        # ``cut`` is None and we keep the full gen sequence — which is
        # already exactly the right length (max_new_tokens of real
        # tokens, no trailing pads because that row WAS max_len).
        eos_ids = self._eos_ids
        rollouts: list[dict] = []
        for i in range(n):
            seq = outputs[i].tolist()
            gen = seq[prompt_length:]
            cut: int | None = next(
                (j for j, t in enumerate(gen) if t in eos_ids),
                None,
            )
            if cut is not None:
                gen = gen[: cut + 1]
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
            })
        return rollouts

    def _score_rollout(self, generation: dict, problem: dict) -> tuple[float, int]:
        """Decode + score one generation; returns (reward, completion_length)."""
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        reward = self.env.compute_reward(problem, completion_text)
        return reward, len(completion_tokens)

    # ------------------------------------------------------------------
    # Submission construction (batched or per-rollout)
    # ------------------------------------------------------------------

    def _build_rollout_submissions(
        self,
        generations: list[dict],
        rewards: list[float],
        randomness: str,
    ) -> list[RolloutSubmission]:
        """Return a list of ``RolloutSubmission`` for the given generations.

        Dispatches on ``_metrics.batched_proof_active``: the fast batched
        path when the flag is set (default), the per-rollout v3-compat
        path after a fallback has been triggered.
        """
        if self._metrics.batched_proof_active:
            return self._build_rollout_submissions_batched(
                generations, rewards, randomness,
            )
        return [
            self._build_rollout_submission_single(g, rew, randomness)
            for g, rew in zip(generations, rewards)
        ]

    def _build_rollout_submission_single(
        self,
        generation: dict,
        reward: float,
        randomness: str,
    ) -> RolloutSubmission:
        """Per-rollout (v3-compat) RolloutSubmission builder."""
        all_tokens = generation["tokens"]
        commit = self._build_grail_commit_single(generation, randomness)
        return RolloutSubmission(
            tokens=all_tokens,
            reward=reward,
            commit=commit,
        )

    def _build_grail_commit_single(self, generation: dict, randomness: str) -> dict:
        """Single-rollout GRAIL forward + commitments + log-probs + signature.

        Bit-identical to v3's ``_build_grail_commit`` — kept under a new
        name because v4 also has a batched sibling. Validator-side
        ``verify_commitment_proofs`` does the same single-sequence forward,
        which is why this path is the safe fallback for the batched one.
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]

        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs: list[float] = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")
        signature = sign_commit_binding(
            all_tokens, randomness, model_name, LAYER_INDEX,
            commitments, self.wallet,
        )

        return {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": GRAIL_PROOF_VERSION,
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": {
                "prompt_length": prompt_length,
                "completion_length": len(all_tokens) - prompt_length,
                "success": True,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": token_logprobs,
            },
        }

    def _build_rollout_submissions_batched(
        self,
        generations: list[dict],
        rewards: list[float],
        randomness: str,
    ) -> list[RolloutSubmission]:
        """Batched-GRAIL builder: ONE padded HF forward for all M rollouts.

        How it stays bit-compatible with the validator's per-rollout
        forward:

        - All M rollouts share the prompt prefix (same prompt_idx → same
          ``problem["prompt"]`` → same canonical tokenization). We assert
          this and bail to the per-rollout path if violated.
        - Right-pad rollouts with ``pad_token_id`` to ``max_seq_len``.
          Build an ``attention_mask`` that's 1 for real tokens, 0 for pads.
        - With this mask, FA2/SDPA computes attention over real tokens
          only — real positions never attend to or are attended-from
          pad positions. LayerNorm is per-token, projections are per-token,
          attention softmax is per-query-row. There are no cross-row
          reductions, so the activations at real positions in row i are
          mathematically identical to the activations a single-sequence
          ``[1, real_len_i]`` forward would produce — which is exactly
          what the validator runs in ``verify_commitment_proofs``.
        - Kernel-level differences (tile scheduling for the matmuls)
          can introduce tiny FP drift between batch sizes; that drift
          stays comfortably inside ``PROOF_SKETCH_TOLERANCE_BASE = 5000``
          plus the per-position sqrt growth. Belt-and-suspenders: if two
          consecutive batched submissions return GRAIL_FAIL,
          ``_maybe_check_proof_fallback`` flips the engine to the
          per-rollout path for the rest of the session.

        Slicing per-rollout outputs:

        - ``hidden_states_batch[i, :real_len_i]`` → matches the
          single-sequence forward's full hidden state tensor.
        - ``logits_batch[i, :real_len_i]`` → same.
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        if not generations:
            return []

        # All M rollouts must share the same prompt prefix. The .generate()
        # call uses the same prompt tokens for every row, so this is the
        # expected case — but if it's ever violated, bail to per-rollout
        # so the validator path never sees an out-of-band proof.
        prompt_length = generations[0]["prompt_length"]
        first_prompt = generations[0]["tokens"][:prompt_length]
        for g in generations[1:]:
            if (
                g["prompt_length"] != prompt_length
                or g["tokens"][:prompt_length] != first_prompt
            ):
                logger.warning(
                    "batched-proof: prompt prefix mismatch across rollouts; "
                    "falling back to per-rollout for this submission",
                )
                return [
                    self._build_rollout_submission_single(g, rew, randomness)
                    for g, rew in zip(generations, rewards)
                ]

        token_lists: list[list[int]] = [g["tokens"] for g in generations]
        real_lens: list[int] = [len(t) for t in token_lists]
        max_len: int = max(real_lens)
        M = len(generations)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        # Build padded tokens + mask on CPU first (fast), then ship to GPU.
        padded_rows: list[list[int]] = []
        for tokens, rl in zip(token_lists, real_lens):
            padded_rows.append(tokens + [pad_id] * (max_len - rl))

        device = f"cuda:{self.proof_gpu}"
        proof_input = torch.tensor(padded_rows, dtype=torch.long, device=device)
        attention_mask = torch.zeros(
            (M, max_len), dtype=torch.long, device=device,
        )
        for i, rl in enumerate(real_lens):
            attention_mask[i, :rl] = 1

        with torch.no_grad():
            hidden_states_batch, logits_batch = forward_single_layer(
                self.hf_model, proof_input, attention_mask, LAYER_INDEX
            )

        r_vec = self._verifier.generate_r_vec(randomness)
        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")

        submissions: list[RolloutSubmission] = []
        for i, (g, rew) in enumerate(zip(generations, rewards)):
            real_len = real_lens[i]
            all_tokens = token_lists[i]

            # Slice the real-token portion only — the padded suffix is
            # numerically meaningless under the attention mask.
            hidden_states_i = hidden_states_batch[i, :real_len]
            commitments = self._verifier.create_commitments_batch(
                hidden_states_i, r_vec,
            )

            log_probs = torch.log_softmax(
                logits_batch[i, :real_len].float(), dim=-1,
            )
            token_logprobs: list[float] = []
            for j in range(prompt_length, real_len):
                token_logprobs.append(log_probs[j - 1, all_tokens[j]].item())

            signature = sign_commit_binding(
                all_tokens, randomness, model_name, LAYER_INDEX,
                commitments, self.wallet,
            )

            commit = {
                "tokens": all_tokens,
                "commitments": commitments,
                "proof_version": GRAIL_PROOF_VERSION,
                "model": {"name": model_name, "layer_index": LAYER_INDEX},
                "signature": signature.hex(),
                "beacon": {"randomness": randomness},
                "rollout": {
                    "prompt_length": prompt_length,
                    "completion_length": real_len - prompt_length,
                    "success": True,
                    "total_reward": 0.0,
                    "advantage": 0.0,
                    "token_logprobs": token_logprobs,
                },
            }
            submissions.append(
                RolloutSubmission(
                    tokens=all_tokens,
                    reward=rew,
                    commit=commit,
                )
            )

        return submissions

    # ------------------------------------------------------------------
    # Randomness (carried from v3)
    # ------------------------------------------------------------------

    async def _randomness_for_window(
        self, subtensor, window_n: int, use_drand: bool
    ) -> str:
        """Cache window randomness so polling iterations don't re-derive it."""
        if self._cached_window_n == window_n and self._cached_randomness:
            return self._cached_randomness
        t = time.monotonic()
        randomness = await self._compute_randomness(subtensor, window_n, use_drand)
        chain_ms = (time.monotonic() - t) * 1000.0
        self._cached_window_n = window_n
        self._cached_randomness = randomness
        logger.debug(
            "[W=%d] randomness=%s... chain_ms=%.0f use_drand=%s",
            window_n, randomness[:16], chain_ms, use_drand,
        )
        return randomness

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool
    ) -> str:
        """Window randomness = block_hash(window_start) [+ drand beacon]."""
        block_hash = await chain.get_block_hash(subtensor, window_start)
        if use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                block_hash, beacon["randomness"], drand_round=beacon["round"]
            )
        return chain.compute_window_randomness(block_hash)
