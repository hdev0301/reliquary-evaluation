"""Miner engine — accuracy picker + batched GRAIL + serial pipeline.

Replaces ``reliquary/miner/engine.py``. Requires the submitter overlay for
multi-validator broadcast (``discover_validator_urls`` and
``submit_batch_v2_multi``); falls back to single-validator submission with a
startup warning otherwise. Install ``orjson`` for faster /submit encoding.

Engine invariants:
- Per-window randomness from ``block_hash(state.window_n)`` (+ drand beacon).
- Bit-identical GRAIL forward path vs the validator (``forward_single_layer``).
- Pre- and post-proof ``/state`` rechecks skip doomed POSTs across window rolls.
- Local OUT_OF_ZONE short-circuit.
- Atomic posterior persistence (no torn writes on kill).
- Two-GPU topology: ``vllm_gpu`` for generation, ``proof_gpu`` for proofs.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import random as _random

from reliquary.constants import (
    B_BATCH,
    BLOCK_TIME_SECONDS,
    BOOTSTRAP_SIGMA_MIN,
    BOOTSTRAP_WINDOWS,
    CHALLENGE_K,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    SIGMA_MIN,
    SUBNET_START_BLOCK,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
    WINDOW_LENGTH,
)

# Empirical OPEN-phase upper bound. Protocol minimum is 60s
# (WINDOW_LENGTH*BLOCK_TIME_SECONDS); observed windows run 1.5-3.5 min,
# bounded by B_BATCH=8 fill rate. Picker's deadline math uses this ceiling.
_OPEN_PHASE_BUDGET_S: int = 240
_OPEN_PHASE_MIN_S: int = WINDOW_LENGTH * BLOCK_TIME_SECONDS  # = 60

# Rolling tracker for observed OPEN duration → _effective_open_budget_s().
_OBSERVED_OPEN_HISTORY: int = 20
_OBSERVED_OPEN_SAFETY_FACTOR: float = 0.85
_OBSERVED_OPEN_MIN_FLOOR_S: float = 75.0

# Per-request /state timeout — overrides the client read timeout so a
# wedged poll can't burn the OPEN window. Keep this tight for latency:
# competitors poll the same endpoint; faster probes → fresher window_n.
_STATE_PROBE_TIMEOUT_S: float = 3.0

# After /state transport failures we backoff briefly — still far below
# POLL_INTERVAL_SECONDS (10s in reliquary.constants) so we reconnect fast.
_STATE_POLL_BACKOFF_S: float = 3.0

# Hard pre-pick deadline uses a floor on assumed gen time. When pregen has
# queued in-zone candidates, consumption skips full generation (~30–40s),
# so assume a much smaller residual pipeline.
_MIN_GEN_HARD_GATE_S: float = 4.0
_MIN_GEN_HARD_GATE_PREGEN_S: float = 1.5

# Faster idle polls than 1s defaults — reduces latency to detect OPEN /
# next window vs competitors on the same validator.
_IDLE_POLL_NON_OPEN_S: float = 0.5
_IDLE_POLL_BATCH_FULL_S: float = 0.35
_WAIT_SPIN_S: float = 1.0

# Skip redundant GET /state after proof when pre-proof probe is fresh —
# saves one RTT on the critical path (~50–400ms in practice).
_SKIP_POST_CHECK_MAX_WALL_S: float = 14.0
_SKIP_POST_CHECK_MAX_PROOF_S: float = 12.0
from reliquary.infrastructure import chain
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)

# When set, _emit forwards only INFO messages containing the primary
# ``GEN`` or ``SUB`` markers. WARNING/ERROR always pass through.
_PIPELINE_LOGS_ONLY = os.environ.get(
    "RELIQUARY_PIPELINE_LOGS_ONLY", "1",
).strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Tunables — picker, posterior, deadline, over-generation
# ---------------------------------------------------------------------------

# Submission-counter cadence for SUMMARY log and stats flush.
_SUMMARY_EVERY = 10

# Default location for the on-disk posterior cache. Pass ``stats_path=None``
# to disable persistence.
_DEFAULT_STATS_PATH = ".reliquary_miner_stats.json"

# Posterior persistence schema version. v1 = counts+lengths; v2 adds
# cohort_counts, last_checkpoint, arrival_ema, superseded_lifetime.
_STATS_SCHEMA = 2

# Lazy decay factors applied via posterior() reads.
_CHECKPOINT_DECAY = 0.5
_COHORT_DECAY = 0.85

# Cohort prior gating: below MIN_OBS, fall back to Beta(1,1); ESS cap
# keeps a strong cohort from drowning out per-prompt evidence.
_COHORT_MIN_OBS = 8
_COHORT_ESS_CAP = 20.0

# Best-of-K candidates in the Thompson picker. K=64 gives ~84% chance
# of hitting any given cohort (vs 35 cohorts in MATH).
_CANDIDATES_DEFAULT = 64

# Second-chance expansion when the initial best score is below threshold —
# helps in the strong-model regime where most cohorts saturate at k=8.
_CANDIDATES_SECOND_CHANCE = 256
_SECOND_CHANCE_SCORE_THRESHOLD: float = 0.15

# Race mode: shrink K as valid_submissions approaches B_BATCH (TCP arrival
# order matters for SUPERSEDED).
_CANDIDATES_RACE = 16
_RACE_MODE_SLOTS_THRESHOLD = max(2, B_BATCH - 4)
_CANDIDATES_NEAR_FULL = 8
_NEAR_FULL_SLOTS_THRESHOLD = max(1, B_BATCH - 2)

# Congestion tilt: down-weight zone_p by arrival_rate (proxy for popular
# prompts that other miners are winning).
_CONGESTION_WEIGHT = 0.4
_ARRIVAL_EMA_ALPHA = 0.05
_ARRIVAL_QUIESCENT_DECAY = 1.0 - _ARRIVAL_EMA_ALPHA * 0.1
_ARRIVAL_PRUNE_BELOW = 1e-4

# Trigger fallback to per-rollout proofs after this many consecutive
# batched GRAIL_FAILs.
_BATCHED_PROOF_FAIL_THRESHOLD = 2

# Reward threshold for "solved" — works for {0,1} MATH rewards and
# continuous reward envs alike.
_SOLVED_THRESHOLD = 0.5

# Over-generation cherry-pick: when initial M=8 is OOZ, draw EXTRA more
# rollouts and assemble an in-zone subset of M. Each rollout remains an
# honest protocol-temperature sample; subset selection only biases group
# sigma (the gate we're trying to clear).
_OVERGEN_ENABLED: bool = True
_OVERGEN_EXTRA: int = 4

# Posterior-informed max_new_tokens cap. Tightens max_new_tokens to
# ``observed_mean * MULT`` (floored at FLOOR) so the slowest rollout in
# a batched gen can't drag the whole batch to the 8192 cap. Mean (not
# max) is used because max equals the protocol cap as soon as any one
# rollout saturates — useless as a tightening signal. Trade-off: the
# rare long-tail rollout gets truncated, likely returns reward=0; for
# k=8/8 saturated prompts that's neutral-to-positive, for in-zone prompts
# it can knock the group out of zone. Right call when latency > yield.
_PROMPT_BUDGET_MIN_OBS: int = 8  # one full group of M=8
_PROMPT_BUDGET_MEAN_MULT: float = 2.0
_PROMPT_BUDGET_FLOOR: int = 1500

# Skip over-gen past this open_age (would overshoot the window edge).
_OVERGEN_MAX_OPEN_AGE_S: float = 120.0

# Skip over-gen at saturated k (recovery odds ~15-20% vs ~40% at k ∈ [1,7]).
_OVERGEN_MIN_K: int = 1
_OVERGEN_MAX_K: int = M_ROLLOUTS - 1

# Picker score multiplier when prompt has ever produced a < CHALLENGE_K
# rollout (would silently fail validator LOGPROB_MISMATCH).
_SHORT_COMPLETION_PENALTY: float = 0.10

# Picker score multiplier when prompt has ever produced a rollout at the
# long-tail band. Originally 0.42 to save GPU time. Empirical finding
# (2026-05-13 from top miner UID 186 dataset): the WINNING strategy is
# max-length completions for nearly every rollout — the validator's
# reward extractor finds the boxed answer regardless of where in the
# completion it appears, and consistent max-length submissions dominate
# the score table. 1.0 = no penalty; preserves the threshold constant
# for diagnostics only.
_LONG_COMPLETION_THRESHOLD_TOKENS: int = 5600
_LONG_COMPLETION_PENALTY: float = 1.0

# Tier 3 (2026-05-13): soft max-length cap. The protocol allows up to
# MAX_NEW_TOKENS_PROTOCOL_CAP (8192), and the validator accepts max-length
# termination. But the slowest rollout in a batched gen dictates wall-
# clock — capping at ~6000 saves ~25% gen time at the cost of missing
# answers boxed past token 6000 (rare for Qwen3-4B on MATH). Env var
# override so operators can A/B without code change. 0 = disabled.
_SOFT_MAX_NEW_TOKENS_CAP: int = int(
    os.environ.get("RELIQUARY_SOFT_MAX_NEW_TOKENS", "6000") or 0
)

# Time-budget-aware picker penalties: estimate
# expected_pipeline_s = open_age + gen + proof + http and penalize
# prompts whose expected finish crosses the OPEN deadline.
# 200 tok/s is conservative for Qwen3-4B-Instruct on H200.
_TOKENS_PER_SEC_EST: float = 200.0
_DEADLINE_PROOF_DEFAULT_S: float = 3.0
_DEADLINE_HTTP_DEFAULT_S: float = 5.0
_DEADLINE_HARD_PENALTY: float = 0.10
_DEADLINE_SOFT_PENALTY: float = 0.50
_DEADLINE_HARD_SLACK_S: float = 5.0
_DEADLINE_SOFT_SLACK_S: float = 20.0

# Per-window OOZ-cohort penalty — multiplicative (not zero) so an
# all-saturated window still picks something rather than starving the loop.
_OOZ_COHORT_PENALTY: float = 0.20

# Reference completion length used to scale http_avg_s per-prompt
# (mid-point of observed Qwen3-4B-Instruct MATH rollout lengths).
_TYPICAL_AVG_LEN_FOR_UPLOAD: float = 4096.0

# Hard threshold above which overgen / SHRT rescue are skipped — the
# uplink is the bottleneck, no point burning more generation we can't ship.
_HIGH_HTTP_AVG_S: float = 45.0


# ---------------------------------------------------------------------------
# Log formatting helpers
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


def _fmt_submit_per_validator(
    per_url: dict[str, tuple[object | None, float, BaseException | None]],
) -> str:
    """One token per validator: host, accept/reject reason, http ms, or NET_ERR."""
    parts: list[str] = []
    for url in sorted(per_url.keys()):
        resp, http_ms, exc = per_url[url]
        host = urlparse(url).netloc or url[:56]
        if exc is not None:
            parts.append(f"{host}:NET_ERR({type(exc).__name__})")
            continue
        if resp is None:
            parts.append(f"{host}:NO_RESPONSE")
            continue
        r_reason = getattr(resp, "reason", None)
        rs = (
            r_reason.value
            if r_reason is not None and hasattr(r_reason, "value")
            else str(r_reason)
        )
        acc = bool(getattr(resp, "accepted", False))
        parts.append(
            f"{host}:{'accepted' if acc else 'rejected'}={rs}:{http_ms:.0f}ms"
        )
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# GPU memory snapshot (NVML) — cheap, robust, no torch dep
# ---------------------------------------------------------------------------

_NVML_INITIALIZED = False
_NVML_AVAILABLE = False


def _ensure_nvml() -> bool:
    """Lazy-init pynvml exactly once. Returns False if unavailable.

    Avoids torch.cuda.mem_get_info which can deadlock when called from
    a thread while vLLM holds the CUDA context.
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
    """Return ``"used/total GB"`` or ``"n/a"`` if NVML unavailable."""
    if not _ensure_nvml():
        return "n/a"
    try:
        import pynvml  # type: ignore
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return f"{info.used / 1e9:.1f}/{info.total / 1e9:.0f}GB"
    except Exception:
        return "err"


def _nvml_mem_fraction_used(gpu_id: int) -> float | None:
    """Return used/total in [0,1] or None if NVML can't read the GPU."""
    if not _ensure_nvml():
        return None
    try:
        import pynvml  # type: ignore
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / float(info.total)
    except Exception:
        return None


# Log occasionally when vLLM GPU is nearly full — high NVML % is often
# normal (reserved pools), but sustained pressure correlates with OOM/slowdown.
_VRAM_PRESSURE_WARN_FRAC: float = 0.92
_VRAM_PRESSURE_WARN_EVERY: int = 12


# ---------------------------------------------------------------------------
# Dual-emit: structured logger.info + optional raw stderr fallback
# ---------------------------------------------------------------------------

_STDERR_FALLBACK_ENABLED = os.environ.get(
    "RELIQUARY_RAW_STDERR", "",
).strip().lower() in ("1", "true", "yes")

# Captured at MODULE IMPORT — before any lib can dup2 over fd 2. Used by
# _raw_stderr_write so RELIQUARY_RAW_STDERR=1 still hits kernel-level
# stderr even when bittensor/btlogging has redirected sys.stderr.
try:
    _RAW_STDERR_FD: int | None = os.dup(2)
except OSError:
    _RAW_STDERR_FD = None

_PINNED_RELIQUARY_LOGGERS = (
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
# Periodic re-strip cadence — third-party libs can re-attach handlers
# after the one-time reseat, producing duplicate log lines.
_HANDLER_SANITY_EVERY = 10
_handler_sanity_counter = 0

# Backstop dedupe ring in case handler-strip AND raw-fd both fail to
# prevent duplication.
_DEDUPE_WINDOW_S = 0.5
_dedupe_last: dict[tuple[str, str], float] = {}


def _resanity_pinned_handlers() -> int:
    """Strip any newly-attached handlers off pinned named loggers."""
    stripped = 0
    for name in _PINNED_RELIQUARY_LOGGERS:
        lg = logging.getLogger(name)
        if lg.handlers:
            for h in list(lg.handlers):
                lg.removeHandler(h)
                stripped += 1
        if not lg.propagate:
            lg.propagate = True
        if lg.disabled:
            lg.disabled = False
    return stripped


def _raw_stderr_write(line: str) -> None:
    """Write directly to the kernel-level stderr fd, bypassing every
    Python-side stream wrap."""
    if not line.endswith("\n"):
        line = line + "\n"
    try:
        data = line.encode("utf-8", "replace")
        if _RAW_STDERR_FD is not None:
            os.write(_RAW_STDERR_FD, data)
        else:
            stream = getattr(sys, "__stderr__", None) or sys.stderr
            stream.write(line)
            stream.flush()
    except Exception:
        pass


def _emit(level: int, fmt: str, *args) -> None:
    """Emit a structured line via logger; optional raw-fd duplicate.

    With ``_PIPELINE_LOGS_ONLY`` (default): INFO-level calls are skipped
    unless the message contains the primary ``GEN`` or ``SUB`` marker.
    WARNING+ always pass through.

    Three defensive layers against duplicate-log pathology: periodic
    handler strip, opt-in raw-fd write, and a 500ms dedupe ring.
    """
    if args:
        try:
            msg = fmt % args
        except Exception:
            msg = f"{fmt} args={args!r}"
    else:
        msg = fmt

    if (
        _PIPELINE_LOGS_ONLY
        and level == logging.INFO
        and " GEN " not in msg
        and " SUB  " not in msg
        and "PREGEN" not in msg
        and "pregen=" not in msg
    ):
        return

    global _handler_sanity_counter
    _handler_sanity_counter += 1
    if _handler_sanity_counter % _HANDLER_SANITY_EVERY == 0:
        n = _resanity_pinned_handlers()
        if n:
            note = (
                f"[engine.v4] handler sanity check: re-stripped {n} "
                f"unauthorized handlers from pinned loggers"
            )
            logger.info(note)
            if _STDERR_FALLBACK_ENABLED:
                _raw_stderr_write(f"[engine] {note}")

    now = time.monotonic()
    dedupe_key = (logger.name, msg)
    last_t = _dedupe_last.get(dedupe_key)
    if last_t is not None and (now - last_t) < _DEDUPE_WINDOW_S:
        _dedupe_last[dedupe_key] = now
        return
    _dedupe_last[dedupe_key] = now
    if len(_dedupe_last) > 256:
        cutoff = now - 60.0
        for k in [k for k, t in _dedupe_last.items() if t < cutoff]:
            del _dedupe_last[k]

    logger.log(level, msg)
    if _STDERR_FALLBACK_ENABLED:
        _raw_stderr_write(f"[engine] {msg}")


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
# Bootstrap: σ ≥ BOOTSTRAP_SIGMA_MIN ⇔ k ∈ [1, M-1] for binary MATH rewards.
_K_LO_BOOTSTRAP: int = 1
_K_HI_BOOTSTRAP: int = M_ROLLOUTS - 1


def _is_bootstrap_window(window_n: int) -> bool:
    """Match ``reliquary.validator.service.is_bootstrap_window``."""
    if window_n < SUBNET_START_BLOCK:
        return False
    return window_n - SUBNET_START_BLOCK < BOOTSTRAP_WINDOWS


def _zone_k_bounds(*, bootstrap: bool) -> tuple[int, int]:
    if bootstrap:
        return _K_LO_BOOTSTRAP, _K_HI_BOOTSTRAP
    return _K_LO, _K_HI


def _zone_probability(p: float, *, bootstrap: bool = False) -> float:
    """``P(k_lo ≤ Binomial(M_ROLLOUTS, p) ≤ k_hi)`` for validator zone gate.

    Steady: k ∈ [2, 6] ⇔ σ ≥ SIGMA_MIN for binary rewards.
    Bootstrap: k ∈ [1, 7] ⇔ σ ≥ BOOTSTRAP_SIGMA_MIN.
    """
    p = max(0.0, min(1.0, p))
    q = 1.0 - p
    k_lo, k_hi = _zone_k_bounds(bootstrap=bootstrap)
    total = 0.0
    for k in range(k_lo, min(k_hi, M_ROLLOUTS) + 1):
        total += _BINOM_M[k] * (p ** k) * (q ** (M_ROLLOUTS - k))
    return total


def _zone_status(
    rewards: list[float],
    *,
    bootstrap: bool = False,
) -> tuple[float, int, bool]:
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
    if sigma < 1e-8:
        return sigma, k_solved, False
    thr = BOOTSTRAP_SIGMA_MIN if bootstrap else SIGMA_MIN
    return sigma, k_solved, sigma >= thr


def _find_in_zone_subset(
    pool_rewards: list[float],
    pool_lens: list[int],
    target_size: int = M_ROLLOUTS,
    *,
    bootstrap: bool = False,
) -> list[int] | None:
    """Cherry-pick ``target_size`` indices so the chosen subset's reward
    stddev clears the in-zone gate. Prefers k closer to ``target_size // 2``
    (max sigma) and shorter completions (smaller payload, faster HTTP).
    Returns None if no in-zone subset of ``target_size`` exists.
    """
    n = len(pool_rewards)
    if n < target_size or len(pool_lens) != n:
        return None

    solves_idx = [i for i in range(n) if pool_rewards[i] >= _SOLVED_THRESHOLD]
    fails_idx = [i for i in range(n) if pool_rewards[i] < _SOLVED_THRESHOLD]

    solves_idx.sort(key=lambda i: pool_lens[i])
    fails_idx.sort(key=lambda i: pool_lens[i])

    k_lo, k_hi = _zone_k_bounds(bootstrap=bootstrap)
    mid_k = target_size // 2
    target_ks = sorted(
        range(k_lo, min(k_hi, target_size) + 1),
        key=lambda k: (abs(k - mid_k), k),
    )

    for k in target_ks:
        n_fails = target_size - k
        if k <= len(solves_idx) and n_fails <= len(fails_idx):
            return solves_idx[:k] + fails_idx[:n_fails]
    return None


# ---------------------------------------------------------------------------
# Metrics — single-line summary the operator can grep
# ---------------------------------------------------------------------------

@dataclass
class _MinerMetrics:
    """Rolling counters surfaced to logs.

    Production validators return ``reason=submitted`` (queued for async
    GRAIL) → ``queued_provisional``. Only ``reason=accepted`` reflects
    the inline verification path (tests / sync servers).
    """

    submitted: int = 0
    queued_provisional: int = 0
    validated_accepted: int = 0
    rejected: int = 0
    network_errors: int = 0
    generated: int = 0
    local_out_of_zone: int = 0
    superseded_in_session: int = 0
    batched_proof_active: bool = True
    batched_proof_consecutive_fails: int = 0
    k_histogram: list[int] = field(default_factory=lambda: [0] * (M_ROLLOUTS + 1))
    reasons: dict[str, int] = field(default_factory=dict)
    http_ms_recent: list[float] = field(default_factory=list)
    _http_ms_window: int = 20
    proof_ms_recent: list[float] = field(default_factory=list)
    _proof_ms_window: int = 20
    overgen_attempts: int = 0
    overgen_recoveries: int = 0

    def record_generation(self, k_solved: int) -> None:
        self.generated += 1
        if 0 <= k_solved <= M_ROLLOUTS:
            self.k_histogram[k_solved] += 1

    def record(self, accepted: bool, reason: str | None) -> None:
        self.submitted += 1
        if accepted:
            r = (reason or "").lower()
            if r == "submitted":
                self.queued_provisional += 1
            elif r == "accepted":
                self.validated_accepted += 1
            else:
                self.queued_provisional += 1
            self.batched_proof_consecutive_fails = 0
        else:
            self.rejected += 1
            if reason == "superseded":
                self.superseded_in_session += 1
            if reason == "grail_fail" and self.batched_proof_active:
                self.batched_proof_consecutive_fails += 1
        if reason is not None:
            self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def record_http_latency(self, http_ms: float) -> None:
        """Append a successful submit_batch_v2 round-trip time."""
        self.http_ms_recent.append(http_ms)
        if len(self.http_ms_recent) > self._http_ms_window:
            self.http_ms_recent.pop(0)

    def recent_http_avg_s(self) -> float:
        """Average of the last ``_http_ms_window`` HTTP times, in seconds.
        Returns 0.0 if no samples yet.
        """
        if not self.http_ms_recent:
            return 0.0
        return (sum(self.http_ms_recent) / len(self.http_ms_recent)) / 1000.0

    def record_proof_latency(self, proof_ms: float) -> None:
        """Append a successful proof-construction time."""
        self.proof_ms_recent.append(proof_ms)
        if len(self.proof_ms_recent) > self._proof_ms_window:
            self.proof_ms_recent.pop(0)

    def recent_proof_avg_s(self) -> float:
        """Average of the last ``_proof_ms_window`` proof times, in seconds.
        Returns 0.0 if no samples yet.
        """
        if not self.proof_ms_recent:
            return 0.0
        return (sum(self.proof_ms_recent) / len(self.proof_ms_recent)) / 1000.0

    def record_overgen_attempt(self) -> None:
        self.overgen_attempts += 1

    def record_overgen_recovery(self) -> None:
        self.overgen_recoveries += 1

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
        q_pct = (
            (self.queued_provisional / self.submitted * 100.0)
            if self.submitted
            else 0.0
        )
        v_pct = (
            (self.validated_accepted / self.submitted * 100.0)
            if self.submitted
            else 0.0
        )
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
        http_avg = self.recent_http_avg_s()
        http_str = (
            f" http_avg={http_avg:.1f}s" if http_avg > 0 else ""
        )
        if self.overgen_attempts > 0:
            ogr_rate = (
                self.overgen_recoveries / self.overgen_attempts * 100.0
            )
            overgen_str = (
                f" overgen={self.overgen_recoveries}/{self.overgen_attempts}"
                f"({ogr_rate:.0f}%)"
            )
        else:
            overgen_str = ""
        return (
            f"generated={self.generated} in_zone={self.in_zone_rate:.1f}% "
            f"submitted={self.submitted} queued={self.queued_provisional} "
            f"({q_pct:.1f}% prov.) validated={self.validated_accepted} "
            f"({v_pct:.1f}% post-GRAIL) rejected={self.rejected} "
            f"local_oos={self.local_out_of_zone} net_err={self.network_errors}"
            f"{warm_str}{proof_str}{http_str}{overgen_str} "
            f"k_hist=[{k_str}] top=[{top_str}]"
        )


# ---------------------------------------------------------------------------
# Async submit context (captures everything needed to log/record a submit
# result that arrives after the generating loop has already moved on)
# ---------------------------------------------------------------------------

@dataclass
class _SubmitCtx:
    """Snapshot of generation metadata for deferred SUB logging — created
    when firing the submit task, consumed when draining the result.
    """

    prompt_idx: int
    window_n: int
    rewards: list
    k_solved: int
    sigma: float
    merkle_root: str
    proof_mode: str
    gen_ms: float
    proof_ms: float
    gpu_pre: str
    open_age_s: float
    tokens: list[int] = field(default_factory=list)
    checkpoint_n: int = 0


@dataclass
class _PregenCandidate:
    """A pre-generated rollout group produced during CLOSED/TRAINING/PUBLISHING
    so that the OPEN-phase critical path skips PICK + GEN + SCORE and goes
    straight to PROOF + SUBMIT. Cuts ~30s off post-OPEN latency, which is the
    dominant lever for winning the validator's FIFO race.
    """

    prompt_idx: int
    level: str
    subject: str
    generations: list[dict]
    rewards: list[float]
    completion_lens: list[int]
    sigma: float
    k_solved: int
    in_zone: bool
    gen_ms: float
    created_at_t: float
    created_at_window_n: int
    created_at_checkpoint_n: int


# Maximum staleness of a pregen candidate before it gets discarded. Older
# than this and the cohort signal / cooldown set has drifted too far.
_PREGEN_MAX_AGE_S: float = 180.0
# Max simultaneous queued pregen candidates. Bumped from 1 → 3 so that
# multiple OPEN submissions benefit (not just the first), and so idle
# GPU time during batch-full / WAIT phases gets converted into ready
# candidates for the next window.
_PREGEN_MAX_QUEUE: int = 3

# Saturation-aware picker: penalize prompts whose recent history is
# dominated by k=0/8 or k=M/8 results (guaranteed OOZ — cherry-pick
# can't help). Threshold tuned for binary MATH rewards where most
# prompts the model has fully learned or fully missed.
_SATURATION_MIN_OBS: int = 2  # need at least 2 groups before trusting the signal
_SATURATION_PENALTY_HIGH: float = 0.05  # ≥80% saturated → near-zero
_SATURATION_PENALTY_MED: float = 0.30   # 50-80% saturated → heavy discount
_SATURATION_THRESHOLD_HIGH: float = 0.80
_SATURATION_THRESHOLD_MED: float = 0.50

# Per-prompt superseded-EMA penalty: prompts that consistently lose the
# FIFO race are hot-contested; shift draws toward same-EV less-contested
# alternatives. Multiplicative (not zero) so the picker can still
# pick them if nothing better is available. Signal = lifetime
# superseded / lifetime attempts. ``MIN_OBS=5`` avoids penalizing
# prompts we've only attempted once or twice on bad luck.
_SUPERSEDED_MIN_OBS: int = 5
_SUPERSEDED_PENALTY_HIGH: float = 0.30
_SUPERSEDED_PENALTY_MED: float = 0.65
_SUPERSEDED_THRESHOLD_HIGH: float = 0.50
_SUPERSEDED_THRESHOLD_MED: float = 0.25

# Own-SUB pregen-mode threshold: once we've landed N successful submits
# in the current window, redirect GPU from live gen to pregen for the
# NEXT window. Fixes the "validator says submit too late" FIFO problem
# by building a warm queue so the next window's first SUB lands fast.
# Validators typically accept 1 SUB per miner per window, so 1 = correct.
_OWN_SUB_PREGEN_THRESHOLD: int = 1


# ---------------------------------------------------------------------------
# Checkpoint pull
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
    """Pull remote checkpoint via HF when remote > local. Returns
    ``(new_local_n, new_local_hash, new_model)`` or inputs unchanged.
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

    Persisted maps (schema v2):
      _counts[idx]              = (solves, attempts)
      _last_checkpoint[idx]     = checkpoint these counts grew on
      _cohort_counts[(L, S)]    = (solves, attempts) for MATH cell
      _cohort_last_checkpoint   = cohort checkpoint tag
      _cohort_inzone[(L, S)]    = (in-zone groups, total groups)
      _lengths[idx]             = (mean_len, n_obs)
      _min_lens / _max_lens     = observed completion length extrema
      _arrival_ema[idx]         = "popular among smart miners" signal
      _superseded_lifetime[idx] = cumulative SUPERSEDED counter

    Checkpoint decay is lazy — posterior() blends stored counts toward
    the cohort prior on read so we don't walk 10k+ entries per bump.
    """

    # alpha_prior / beta_prior MUST be slots, not class attrs — load_from
    # overwrites them per-instance.
    __slots__ = (
        "alpha_prior",
        "beta_prior",
        "_counts",
        "_last_checkpoint",
        "_cohort_counts",
        "_cohort_last_checkpoint",
        "_cohort_inzone",
        "_lengths",
        "_min_lens",
        "_max_lens",
        "_arrival_ema",
        "_superseded_lifetime",
        "_last_cooldown_set",
        "_cohort_cache",
        "_k_saturated",
    )

    def __init__(self) -> None:
        self.alpha_prior: float = 1.0
        self.beta_prior: float = 1.0
        self._counts: dict[int, tuple[int, int]] = {}
        self._last_checkpoint: dict[int, int] = {}
        self._cohort_counts: dict[tuple[str, str], tuple[int, int]] = {}
        self._cohort_last_checkpoint: dict[tuple[str, str], int] = {}
        self._cohort_inzone: dict[tuple[str, str], tuple[int, int]] = {}
        self._lengths: dict[int, tuple[float, int]] = {}
        self._min_lens: dict[int, int] = {}
        self._max_lens: dict[int, int] = {}
        self._arrival_ema: dict[int, float] = {}
        self._superseded_lifetime: dict[int, int] = {}
        self._last_cooldown_set: frozenset[int] = frozenset()
        self._cohort_cache: dict[int, tuple[str, str]] = {}
        # Per-prompt saturation tracker: (saturated_groups, total_groups)
        # where "saturated" = k_solved ∈ {0, M_ROLLOUTS}. Used by the
        # picker to hard-skip prompts the model has fully learned or
        # fully missed — these can't be rescued by cherry-pick.
        self._k_saturated: dict[int, tuple[int, int]] = {}

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

    def min_completion_len(self, idx: int) -> int | None:
        return self._min_lens.get(idx)

    def has_short_completion(self, idx: int, threshold: int) -> bool:
        m = self._min_lens.get(idx)
        return m is not None and m < threshold

    def max_completion_len(self, idx: int) -> int | None:
        return self._max_lens.get(idx)

    def has_long_completion(self, idx: int, threshold: int) -> bool:
        m = self._max_lens.get(idx)
        return m is not None and m >= threshold

    def saturation_rate(self, idx: int, min_obs: int) -> float | None:
        """Fraction of past full-M groups where k_solved was 0 or M
        (guaranteed OOZ, unrecoverable). Returns None when fewer than
        ``min_obs`` groups recorded — picker should fall back to
        cohort-level signals in that case.
        """
        s, n = self._k_saturated.get(idx, (0, 0))
        if n < min_obs:
            return None
        return s / n

    def superseded_rate(self, idx: int, min_attempts: int) -> float | None:
        """Lifetime ``superseded / attempts`` for this prompt.

        ``superseded`` fires when validator's batch was already full by
        the time our submission arrived — a pure FIFO-race loss
        signal. High rate ⇒ hot-contested prompt; picker should prefer
        same-EV less-contested alternatives.

        ``attempts`` counts generations (not just submissions), so this
        under-estimates the true contention rate, but the ordering
        across prompts is preserved.

        Returns None when fewer than ``min_attempts`` attempts logged.
        """
        _, n = self._counts.get(idx, (0, 0))
        if n < min_attempts:
            return None
        sup = self._superseded_lifetime.get(idx, 0)
        if sup == 0:
            return 0.0
        return min(1.0, sup / float(n))

    # ------------------------- updates -------------------------

    def record_group(
        self,
        prompt_idx: int,
        rewards: list[float],
        *,
        level: str,
        subject: str,
        checkpoint_n: int,
        window_n: int,
        completion_lens: list[int] | None = None,
        for_posterior: bool = True,
    ) -> None:
        """Update per-prompt counts, the cohort cell, and length stats.

        Tags both prompt and cohort with current checkpoint_n so the lazy
        decay starts from "now" rather than retroactively discounting
        just-collected evidence.

        ``for_posterior=False``: only update rolling length stats (used when
        rewards are not an unbiased random M-group, e.g. after assembly
        steps that would poison Beta / cohort-in-zone learning).
        """
        attempts = len(rewards)
        if attempts == 0:
            return
        solves = sum(1 for r in rewards if r >= _SOLVED_THRESHOLD)
        boot = _is_bootstrap_window(window_n)

        if for_posterior:
            s_prev, n_prev = self._counts.get(prompt_idx, (0, 0))
            self._counts[prompt_idx] = (s_prev + solves, n_prev + attempts)
            self._last_checkpoint[prompt_idx] = checkpoint_n

            key = self._cohort_key(level, subject)
            cs, cn = self._cohort_counts.get(key, (0, 0))
            self._cohort_counts[key] = (cs + solves, cn + attempts)
            self._cohort_last_checkpoint[key] = checkpoint_n

            _, group_k, group_inzone = _zone_status(rewards, bootstrap=boot)
            ciz, ctot = self._cohort_inzone.get(key, (0, 0))
            self._cohort_inzone[key] = (
                ciz + (1 if group_inzone else 0),
                ctot + 1,
            )

            # Per-prompt saturation count. Only tracked for full groups
            # (len == M_ROLLOUTS) — cherry-pick / SHRT-rescue extras would
            # bias the saturation rate downward because they're smaller M.
            if attempts == M_ROLLOUTS:
                sat_s, sat_n = self._k_saturated.get(prompt_idx, (0, 0))
                is_saturated = group_k == 0 or group_k == M_ROLLOUTS
                self._k_saturated[prompt_idx] = (
                    sat_s + (1 if is_saturated else 0),
                    sat_n + 1,
                )

        self._cohort_cache[prompt_idx] = (str(level), str(subject))

        if completion_lens:
            prev_mean, prev_n = self._lengths.get(prompt_idx, (0.0, 0))
            total_n = prev_n + len(completion_lens)
            new_mean = (prev_mean * prev_n + sum(completion_lens)) / total_n
            self._lengths[prompt_idx] = (new_mean, total_n)
            cur_min = min(completion_lens)
            stored_min = self._min_lens.get(prompt_idx)
            if stored_min is None or cur_min < stored_min:
                self._min_lens[prompt_idx] = cur_min
            cur_max = max(completion_lens)
            stored_max = self._max_lens.get(prompt_idx)
            if stored_max is None or cur_max > stored_max:
                self._max_lens[prompt_idx] = cur_max

    def completion_budget(self, prompt_idx: int) -> int | None:
        """Suggested max_new_tokens cap = mean_completion_len × MULT.
        Returns None until ``_PROMPT_BUDGET_MIN_OBS`` samples accumulated.
        """
        stored = self._lengths.get(prompt_idx)
        if stored is None:
            return None
        mean_len, n_obs = stored
        if n_obs < _PROMPT_BUDGET_MIN_OBS:
            return None
        cap = int(mean_len * _PROMPT_BUDGET_MEAN_MULT)
        return max(cap, _PROMPT_BUDGET_FLOOR)

    def record_cooldown_diff(self, new_cooldown_set: set[int] | list[int]) -> None:
        """Bump arrival_ema toward 1 for newly-cooled prompts; decay others.
        Prunes below ``_ARRIVAL_PRUNE_BELOW`` to bound dict size.
        """
        new = frozenset(new_cooldown_set)
        newly_entered = new - self._last_cooldown_set
        for idx in newly_entered:
            old = self._arrival_ema.get(idx, 0.0)
            self._arrival_ema[idx] = old + _ARRIVAL_EMA_ALPHA * (1.0 - old)

        # Iterate over a snapshot — we delete in-loop.
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

    def cohort_inzone_rate(
        self,
        level: str,
        subject: str,
        min_groups: int = 2,
    ) -> float | None:
        """Laplace-smoothed empirical in-zone rate for the cohort, or
        None when fewer than ``min_groups`` groups recorded.
        """
        key = self._cohort_key(level, subject)
        iz, tot = self._cohort_inzone.get(key, (0, 0))
        if tot < min_groups:
            return None
        return (iz + 0.5) / (tot + 1.0)

    # ------------------------- persistence -------------------------

    def save_to(self, path: str) -> None:
        """Atomically persist all maps as schema-v2 JSON (write-to-tmp +
        os.replace). Cohort cache is session-scoped, not persisted.
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
            "cohort_inzone": {
                f"{level}\t{subject}": list(v)
                for (level, subject), v in self._cohort_inzone.items()
            },
            "lengths": {str(k): list(v) for k, v in self._lengths.items()},
            "min_lens": {str(k): v for k, v in self._min_lens.items()},
            "max_lens": {str(k): v for k, v in self._max_lens.items()},
            "arrival_ema": {str(k): v for k, v in self._arrival_ema.items()},
            "superseded_lifetime": {
                str(k): v for k, v in self._superseded_lifetime.items()
            },
            "k_saturated": {
                str(k): list(v) for k, v in self._k_saturated.items()
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
        # min_lens / max_lens optional — additive, rebuild from new obs.
        self._min_lens = {
            int(k): int(v)
            for k, v in (payload.get("min_lens") or {}).items()
        }
        self._max_lens = {
            int(k): int(v)
            for k, v in (payload.get("max_lens") or {}).items()
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
            raw_ciz = _parse_cohort(payload.get("cohort_inzone") or {})
            self._cohort_inzone = {
                key: (int(v[0]), int(v[1])) for key, v in raw_ciz.items()
            }
            self._arrival_ema = {
                int(k): float(v)
                for k, v in (payload.get("arrival_ema") or {}).items()
            }
            self._superseded_lifetime = {
                int(k): int(v)
                for k, v in (payload.get("superseded_lifetime") or {}).items()
            }
            # k_saturated added post-v2; optional, rebuilds from new obs.
            self._k_saturated = {
                int(k): (int(v[0]), int(v[1]))
                for k, v in (payload.get("k_saturated") or {}).items()
            }
        else:
            # v1 → v2 migration: last_checkpoint=0 treats data as stale.
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
    window_n: int = 0,
    slots_filled: int = 0,
    superseded_in_window: set[int] | None = None,
    candidates: int = _CANDIDATES_DEFAULT,
    max_attempts: int = 1000,
    open_age_s: float = 0.0,
    open_budget_s: float = float("inf"),
    proof_avg_s: float = _DEADLINE_PROOF_DEFAULT_S,
    http_avg_s: float = _DEADLINE_HTTP_DEFAULT_S,
    short_completion_threshold: int = CHALLENGE_K,
    ooz_cohorts_in_window: set[tuple[str, str]] | None = None,
) -> int:
    """Best-of-K Thompson picker.

    Score = zone_p × (1 - CONGESTION_WEIGHT × arrival_z) × ciz_rate²
            × {short, long, deadline, length-boost} multipliers.
    Tiebreak: shorter avg completion length.

    K shrinks as slots_filled approaches B_BATCH (race / near-cap modes).
    Hard blacklist for ``superseded_in_window``. Falls back to uniform
    random when ``stats`` is None.

    ``window_n`` selects bootstrap vs steady zone prior (matches validator).
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

    if slots_filled >= _NEAR_FULL_SLOTS_THRESHOLD:
        K = _CANDIDATES_NEAR_FULL
    elif slots_filled >= _RACE_MODE_SLOTS_THRESHOLD:
        K = _CANDIDATES_RACE
    else:
        K = candidates

    seen: set[int] = set()
    best_idx: int | None = None
    best_score: float = -1.0
    best_len: float = float("inf")

    def _score_one(idx: int) -> tuple[float, float] | None:
        if idx in seen:
            return None
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
        _boot = _is_bootstrap_window(window_n)
        zone_p = _zone_probability(p_sampled, bootstrap=_boot)
        arrival_z = stats.arrival_rate(idx)
        score = zone_p * (1.0 - _CONGESTION_WEIGHT * arrival_z)

        # ciz² gives ~12-25x discrimination between balanced (~0.7) and
        # saturated (~0.2) cohorts vs ~3.5x under linear.
        ciz_rate = stats.cohort_inzone_rate(level, subject)
        if ciz_rate is not None:
            score *= ciz_rate * ciz_rate

        if (
            ooz_cohorts_in_window is not None
            and (level, subject) in ooz_cohorts_in_window
        ):
            score *= _OOZ_COHORT_PENALTY

        if stats.has_short_completion(idx, short_completion_threshold):
            score *= _SHORT_COMPLETION_PENALTY

        if stats.has_long_completion(idx, _LONG_COMPLETION_THRESHOLD_TOKENS):
            score *= _LONG_COMPLETION_PENALTY

        # Hard-discount prompts whose recent history is dominated by
        # k=0/8 or k=M/8 groups. These are unrecoverable (cherry-pick
        # can't flip a fully-saturated group) and burn ~32s of GPU per
        # pick to produce nothing. The picker still draws them via
        # Thompson sampling when alternatives are scarce — multiplicative,
        # not zero, so the loop never starves.
        sat_rate = stats.saturation_rate(idx, _SATURATION_MIN_OBS)
        if sat_rate is not None:
            if sat_rate >= _SATURATION_THRESHOLD_HIGH:
                score *= _SATURATION_PENALTY_HIGH
            elif sat_rate >= _SATURATION_THRESHOLD_MED:
                score *= _SATURATION_PENALTY_MED

        # Per-prompt superseded-EMA penalty — shift draws away from
        # prompts where we consistently lose the FIFO race. Multiplicative
        # (not zero) so the picker can still pick them when same-EV
        # alternatives are scarce.
        sup_rate = stats.superseded_rate(idx, _SUPERSEDED_MIN_OBS)
        if sup_rate is not None:
            if sup_rate >= _SUPERSEDED_THRESHOLD_HIGH:
                score *= _SUPERSEDED_PENALTY_HIGH
            elif sup_rate >= _SUPERSEDED_THRESHOLD_MED:
                score *= _SUPERSEDED_PENALTY_MED

        avg_len_val = stats.avg_completion_len(idx)
        if (
            open_budget_s != float("inf")
            and avg_len_val is not None
            and _TOKENS_PER_SEC_EST > 0
        ):
            expected_gen_s = float(avg_len_val) / _TOKENS_PER_SEC_EST
            _length_factor = max(
                0.5,
                min(2.5, float(avg_len_val) / _TYPICAL_AVG_LEN_FOR_UPLOAD),
            )
            expected_http_s = http_avg_s * _length_factor
            expected_finish_s = (
                open_age_s + expected_gen_s + proof_avg_s + expected_http_s
            )
            slack_s = open_budget_s - expected_finish_s
            if slack_s < _DEADLINE_HARD_SLACK_S:
                score *= _DEADLINE_HARD_PENALTY
            elif slack_s < _DEADLINE_SOFT_SLACK_S:
                score *= _DEADLINE_SOFT_PENALTY

        # Smooth ±10% length nudge — gradient even when deadline slack is fine.
        if avg_len_val is not None and avg_len_val > 0:
            _len_ratio = float(avg_len_val) / _TYPICAL_AVG_LEN_FOR_UPLOAD
            _len_boost = 1.0 + 0.10 * (1.0 - _len_ratio)
            score *= max(0.85, min(1.15, _len_boost))

        eff_len = float(avg_len_val) if avg_len_val is not None else float("inf")
        return score, eff_len

    def _update_best(idx: int, score: float, alen: float) -> None:
        nonlocal best_idx, best_score, best_len
        if score > best_score:
            best_idx = idx
            best_score = score
            best_len = alen
        elif score == best_score and alen < best_len:
            best_idx = idx
            best_len = alen

    def _scan(num: int) -> None:
        for _ in range(num):
            idx = _draw_one()
            if idx is None:
                return
            scored = _score_one(idx)
            if scored is None:
                continue
            score, alen = scored
            _update_best(idx, score, alen)

    _scan(K)

    # Expand draw when the best score is weak — most cohorts saturated.
    # Skipped in race mode (speed > optimality at that point).
    if (
        best_score < _SECOND_CHANCE_SCORE_THRESHOLD
        and slots_filled < _RACE_MODE_SLOTS_THRESHOLD
        and K < _CANDIDATES_SECOND_CHANCE
    ):
        _scan(_CANDIDATES_SECOND_CHANCE - K)

    if best_idx is None:
        raise RuntimeError("no eligible prompt")
    return best_idx


# ---------------------------------------------------------------------------
# Merkle root (carried verbatim from v3)
# ---------------------------------------------------------------------------

def _compute_merkle_root(rollouts) -> str:
    """SHA-256 Merkle root over (idx, tokens, reward, commit) leaves.
    Canonical JSON encoding for cross-impl determinism.
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
# Submission speed helpers — completion cache, multi-prompt, window detector
# ---------------------------------------------------------------------------

class CompletionCache:
    """Cache successful completions by (prompt_idx, checkpoint_n) — skips
    GEN+PROOF when the same prompt recurs at the same checkpoint.
    """

    def __init__(self, max_age_windows: int = 5):
        self.cache: dict[tuple[int, int], list[int]] = {}
        self.max_age_windows = max_age_windows
        self.checkpoint_history: collections.deque = collections.deque(
            maxlen=max_age_windows * 2
        )

    def get(self, prompt_idx: int, checkpoint_n: int) -> list[int] | None:
        """Return cached tokens, or None if cache miss."""
        key = (prompt_idx, checkpoint_n)
        return self.cache.get(key)

    def record_success(
        self, prompt_idx: int, checkpoint_n: int, tokens: list[int],
    ) -> None:
        """Record a successful completion for reuse."""
        key = (prompt_idx, checkpoint_n)
        self.cache[key] = list(tokens)
        self.checkpoint_history.append(checkpoint_n)

    def evict_old_checkpoint(self, old_checkpoint_n: int) -> None:
        """Drop entries for an old checkpoint to avoid stale hits."""
        to_del = [k for k, v in self.cache.items() if k[1] == old_checkpoint_n]
        for k in to_del:
            del self.cache[k]

    def size_mb(self) -> float:
        total_tokens = sum(len(t) for t in self.cache.values())
        return total_tokens * 4 / (1024 * 1024)


class MultiPromptResult:
    """Result from parallel multi-prompt generation."""

    def __init__(self):
        self.completions: list[list[int]] = []
        self.prompt_indices: list[int] = []
        self.gen_times_ms: list[float] = []
        self.first_good_idx: int | None = None


async def _generate_candidates_parallel(
    engine: "MiningEngine",
    env: "Environment",
    prompt_indices: list[int],
    problem_data: dict,
    k: int = 4,
) -> MultiPromptResult:
    """Generate K candidate prompts in parallel, results in arrival order."""
    import asyncio

    result = MultiPromptResult()
    result.prompt_indices = list(prompt_indices)

    tasks = []
    for idx in prompt_indices:
        task = asyncio.create_task(
            asyncio.to_thread(
                engine._generate_n_rollouts,
                env.get_problem(idx), M_ROLLOUTS,
                idx,
            )
        )
        tasks.append(task)

    for task in asyncio.as_completed(tasks):
        try:
            completions = await task
            result.completions.append(completions)
            result.gen_times_ms.append(0.0)
        except Exception:
            result.completions.append([])

    return result


class WindowRollDetector:
    """Track window state to detect OPEN transitions for early queueing."""

    def __init__(self):
        self.last_window_n: int | None = None
        self.last_state: str = "CLOSED"
        self.window_open_time: float = 0.0

    def check_and_update(self, current_window_n: int, current_state: str) -> bool:
        """Return True if we just transitioned into OPEN."""
        just_opened = (
            self.last_state != "OPEN" and current_state == "OPEN"
        )
        if just_opened:
            self.window_open_time = time.monotonic()
        self.last_window_n = current_window_n
        self.last_state = current_state
        return just_opened


# ---------------------------------------------------------------------------
# MiningEngine
# ---------------------------------------------------------------------------

class MiningEngine:
    """Serial PICK→GEN→PROOF→SUBMIT pipeline with cohort-aware picker and
    batched GRAIL proof. Pass an HF ``AutoModelForCausalLM`` or a
    ``VLLMAdapter`` as ``vllm_model`` (detected via ``_is_vllm_adapter``).
    Two consecutive GRAIL_FAILs auto-fall-back to per-rollout proofs.
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
        validator_urls_override: list[str] | None = None,
        max_validators: int = 5,
        http_timeout_s: float = 30.0,
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
        # Multi-validator broadcast: discover up to max_validators from
        # the metagraph and submit to all in parallel for per-validator
        # EMA contribution.
        if validator_urls_override:
            self.validator_urls_override: list[str] | None = list(validator_urls_override)
        elif validator_url_override:
            self.validator_urls_override = [validator_url_override]
        else:
            self.validator_urls_override = None
        self.max_validators = max_validators
        self.http_timeout_s = http_timeout_s
        self._validator_urls: list[str] = []

        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

        # Per-window randomness cache — re-derived only on window_n change.
        self._cached_window_n: int | None = None
        self._cached_randomness: str = ""

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

        self._completion_cache = CompletionCache(max_age_windows=5)
        self._window_roll_detector = WindowRollDetector()
        self._superseded_in_window: set[int] = set()

        # Qwen3 stops on EITHER <|im_end|> (151645) OR <|endoftext|>
        # (151643). vLLM honors all entries in generation_config.eos_token_id
        # — we must match that or rollouts ending on the secondary EOS
        # never get truncated and trailing pads survive into submission.
        self._eos_ids: set[int] = self._resolve_eos_ids()
        self._pad_id: int | None = getattr(
            self.tokenizer, "pad_token_id", None,
        )
        logger.info(
            "engine eos resolution: eos_ids=%s pad_id=%s",
            sorted(self._eos_ids), self._pad_id,
        )

        # Lazy one-time logger reseat — fires after all transitive
        # third-party imports complete (bittensor / submitter).
        self._logger_reseated: bool = False

        self._window_open_seen_at: float | None = None
        self._observed_open_durations_s: collections.deque[float] = (
            collections.deque(maxlen=_OBSERVED_OPEN_HISTORY)
        )

        # Per-window OOZ-cohort blacklist — reset at every window-roll boundary.
        self._ooz_cohorts_in_window: set[tuple[str, str]] = set()

        # Own successful submission counter for the current window —
        # incremented in _drain_submit, reset on window-roll. Drives the
        # "pregen-priority mode" switch once threshold is reached.
        self._own_subs_in_window: int = 0

        # Pre-generation pipeline state. Producer runs during non-OPEN
        # phases; consumer pulls in OPEN phase to skip live PICK+GEN+SCORE.
        self._pregen_queue: list[_PregenCandidate] = []
        self._pregen_task: asyncio.Task | None = None
        self._vram_pressure_submit_ctr: int = 0

    def _maybe_warn_vram_pressure(self) -> None:
        """Throttled hint when NVML reports very high utilization on vLLM GPU."""
        frac = _nvml_mem_fraction_used(self.vllm_gpu)
        if frac is None or frac < _VRAM_PRESSURE_WARN_FRAC:
            self._vram_pressure_submit_ctr = 0
            return
        self._vram_pressure_submit_ctr += 1
        h = self._vram_pressure_submit_ctr
        if h != 1 and (h - 1) % _VRAM_PRESSURE_WARN_EVERY != 0:
            return
        extra = ""
        if self.proof_gpu == self.vllm_gpu:
            extra = (
                " Same GPU for vLLM+proof: use lower --gpu-memory-utilization "
                "(~0.55-0.68) or --enforce-eager."
            )
        else:
            extra = (
                " If you see OOM or stalls, lower --gpu-memory-utilization "
                "(e.g. 0.72) or pass --enforce-eager."
            )
        _emit(
            logging.WARNING,
            "[engine.v4] VRAM %.0f%% on vllm gpu=%d — vLLM often sits >90%% "
            "by design; watch for CUDA OOM.%s",
            frac * 100.0,
            self.vllm_gpu,
            extra,
        )

    def _resolve_eos_ids(self) -> set[int]:
        """Collect every token ID the model treats as a stop token, from
        tokenizer.eos_token_id and both models' generation_config (Qwen3
        chat lists multiple).
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
        """Strip handlers from pinned loggers once per process, after
        third-party transitive imports (bittensor/btlogging) settle.
        Idempotent via ``self._logger_reseated``.
        """
        if self._logger_reseated:
            return
        self._logger_reseated = True

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

        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
            get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import (
            BatchSubmissionRequest, WindowState,
        )
        # Optional submitter-v4 helpers — fall back to single-validator if absent.
        try:
            from reliquary.miner.submitter import (
                discover_validator_urls as _discover_validator_urls,
                submit_batch_v2_multi as _submit_batch_v2_multi,
            )
            _MULTI_VALIDATOR_AVAILABLE = True
        except ImportError:
            _discover_validator_urls = None  # type: ignore[assignment]
            _submit_batch_v2_multi = None  # type: ignore[assignment]
            _MULTI_VALIDATOR_AVAILABLE = False
            logger.warning(
                "[engine.v4] submitter does not expose multi-validator helpers "
                "— falling back to single-validator submission. "
                "Overlay submitter-v4.py to enable broadcast."
            )
        try:
            from reliquary.miner.submitter import (
                prewarm_connections as _prewarm_connections,
            )
        except ImportError:
            _prewarm_connections = None  # type: ignore[assignment]

        # Reseat loggers AFTER all transitive imports above (they may
        # clobber the root handler or attach to a named logger).
        self._one_time_logger_reseat()

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

        # Primary URL drives /state polls; full list is used for /submit
        # broadcast so each validator scores us independently.
        if self.validator_urls_override:
            urls = list(self.validator_urls_override)
        elif self.validator_url_override:
            urls = [self.validator_url_override]
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            if _MULTI_VALIDATOR_AVAILABLE and _discover_validator_urls is not None:
                urls = _discover_validator_urls(
                    metagraph, max_n=self.max_validators,
                )
                if not urls:
                    raise RuntimeError(
                        "no permitted validator found on metagraph"
                    )
            else:
                urls = [discover_validator_url(metagraph)]
        self._validator_urls = urls
        url = urls[0]  # primary URL for /state polls (kept for log compat)
        _emit(
            logging.INFO,
            "[engine.v4] validator urls (primary=%s, total=%d): %s "
            "— entering poll/submit loop",
            url, len(urls),
            ",".join(urls) if len(urls) <= 8 else f"{','.join(urls[:8])},...",
        )

        rng = random.Random()
        results = []
        local_n = 0
        local_hash = ""
        last_window_n: int | None = None

        # Validator /submit enqueues async and returns immediately —
        # only meaningful HTTP cost is the body upload (~5-30s for the
        # multi-MB GRAIL payload). Short timeouts + connection pooling
        # so a slow validator can't wedge the OPEN window.
        _http_to_s = float(self.http_timeout_s)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,
                # Response after upload is tiny (SUBMITTED JSON); keep read
                # bounded so a hung socket fails faster than competitors wait.
                read=max(12.0, _http_to_s * 0.45),
                write=_http_to_s,
                pool=5.0,
            ),
            http2=False,  # vLLM-stack envs often lack h2
            limits=httpx.Limits(
                max_keepalive_connections=max(8, len(self._validator_urls) * 2),
                max_connections=max(16, len(self._validator_urls) * 4),
                keepalive_expiry=120.0,
            ),
        ) as client:
            # Pre-warm TCP/TLS so the first /submit lands on already-pooled sockets.
            if _prewarm_connections is not None and self._validator_urls:
                try:
                    _pw_t = time.monotonic()
                    _pw_result = await _prewarm_connections(
                        self._validator_urls,
                        client=client,
                        timeout=5.0,
                    )
                    _pw_ms = (time.monotonic() - _pw_t) * 1000.0
                    _pw_ok = sum(1 for v in _pw_result.values() if v)
                    _emit(
                        logging.INFO,
                        "[engine.v4] validator prewarm: %d/%d reachable "
                        "in %.0fms (TCP/TLS pool primed)",
                        _pw_ok, len(_pw_result), _pw_ms,
                    )
                except Exception as _pw_exc:
                    logger.debug("prewarm_connections failed: %s", _pw_exc)

            # Async submit pattern: fire HTTP as a task, loop back to
            # PICK→GEN, drain at the next iteration. At most one in-flight
            # submit (the validator only accepts one per miner per slot).

            _pending_task: asyncio.Task | None = None
            _pending_ctx: _SubmitCtx | None = None

            async def _fire_submit(req: "BatchSubmissionRequest") -> tuple:
                """Submit to all validators in parallel (or single-validator
                fallback). Returns (accepted, reason, http_ms, gpu_post,
                per_url_breakdown).
                """
                _t = time.monotonic()
                urls = self._validator_urls or [url]
                if _MULTI_VALIDATOR_AVAILABLE and _submit_batch_v2_multi is not None and len(urls) > 1:
                    multi = await _submit_batch_v2_multi(
                        urls, req, client=client,
                        timeout=self.http_timeout_s,
                    )
                    _http_ms = (time.monotonic() - _t) * 1000.0
                    _gpu_post = _gpu_mem_compact(self.vllm_gpu)
                    return (
                        multi.accepted,
                        (
                            multi.best_reason.value
                            if hasattr(multi.best_reason, "value")
                            else str(multi.best_reason)
                        ),
                        _http_ms,
                        _gpu_post,
                        multi.per_url,
                    )
                # Single-validator path (v3 compatibility OR explicit
                # single-URL override).
                _resp = await submit_batch_v2(
                    urls[0], req, client=client,
                    timeout=self.http_timeout_s,
                )
                _http_ms = (time.monotonic() - _t) * 1000.0
                _gpu_post = _gpu_mem_compact(self.vllm_gpu)
                _reason_str = (
                    _resp.reason.value
                    if hasattr(_resp.reason, "value")
                    else str(_resp.reason)
                ) if _resp.reason is not None else "submitted"
                return (
                    _resp.accepted, _reason_str, _http_ms, _gpu_post,
                    {urls[0]: (_resp, _http_ms, None)},
                )

            async def _drain_submit(
                task: "asyncio.Task",
                ctx: "_SubmitCtx",
            ) -> None:
                """Await a (possibly broadcast) submit task and handle metrics."""
                try:
                    accepted, reason_str, http_ms, gpu_post, per_url = await task
                    # All-network-error case: every validator threw rather
                    # than responding — track separately so net_err reflects
                    # connectivity, not validator rejection.
                    n_validators = len(per_url) if per_url else 0
                    n_responded = sum(
                        1 for (r, _, _) in (per_url.values() if per_url else [])
                        if r is not None
                    )
                    if n_validators > 0 and n_responded == 0:
                        self._metrics.record_network_error()
                    else:
                        self._metrics.record(accepted, reason_str)
                    self._metrics.record_http_latency(http_ms)

                    # Track our own per-window successful submissions for
                    # the pregen-priority mode switch (FIFO race fix).
                    if accepted and reason_str == "submitted":
                        self._own_subs_in_window += 1

                    if reason_str == "superseded":
                        # Race for this slot is over — blacklist for window.
                        self._superseded_in_window.add(ctx.prompt_idx)
                        self._prompt_stats.record_superseded(ctx.prompt_idx)
                    self._maybe_check_proof_fallback()
                    status = (
                        "QUEUED"
                        if accepted and reason_str == "submitted"
                        else ("ACCEPTED" if accepted else "REJECTED")
                    )
                    # Validator uvicorn path returns SUBMITTED once the HTTP body
                    # is queued; GRAIL runs later — final accept/reject is not this line.
                    sub_note = (
                        "GRAIL_async"
                        if accepted and reason_str == "submitted"
                        else "-"
                    )
                    n_validators = len(per_url)
                    n_accepted = sum(
                        1 for (r, _, _) in per_url.values()
                        if r is not None and r.accepted
                    )
                    n_errors = sum(
                        1 for (r, _, e) in per_url.values()
                        if r is None and e is not None
                    )
                    multi_str = (
                        f"v={n_accepted}/{n_validators}"
                        + (f" net_err={n_errors}" if n_errors else "")
                    )
                    _emit(
                        logging.INFO if accepted else logging.WARNING,
                        "[W=%d] SUB  prompt=%-4d rewards=%s k=%d/%d "
                        "sigma=%.3f merkle=%s proof=%s gen=%.1fs "
                        "proof=%.1fs http=%.2fs gpu=%s->%s "
                        "open_age=%.1fs %s accepted=%s status=%s reason=%s "
                        "note=%s",
                        ctx.window_n, ctx.prompt_idx,
                        _fmt_rewards(ctx.rewards),
                        ctx.k_solved, M_ROLLOUTS, ctx.sigma,
                        ctx.merkle_root[:8], ctx.proof_mode,
                        ctx.gen_ms / 1000.0, ctx.proof_ms / 1000.0,
                        http_ms / 1000.0,
                        ctx.gpu_pre, gpu_post, ctx.open_age_s,
                        multi_str, accepted, status, reason_str,
                        sub_note,
                    )
                    _need_resp_breakdown = False
                    if per_url:
                        if len(per_url) > 1:
                            _need_resp_breakdown = True
                        else:
                            _r0, _ms0, _e0 = next(iter(per_url.values()))
                            if _r0 is None or _e0 is not None:
                                _need_resp_breakdown = True
                    if _need_resp_breakdown:
                        _emit(
                            logging.INFO,
                            "[W=%d] SUB|r prompt=%-4d per_validator %s",
                            ctx.window_n, ctx.prompt_idx,
                            _fmt_submit_per_validator(per_url),
                        )
                    self._maybe_warn_vram_pressure()
                    results.append((accepted, reason_str))
                except asyncio.CancelledError:
                    _emit(
                        logging.INFO,
                        "[W=%d] CANCEL prompt=%-4d submit cancelled "
                        "(window rolled before response landed)",
                        ctx.window_n, ctx.prompt_idx,
                    )
                except SubmissionError as _sub_exc:
                    self._metrics.record_network_error()
                    _emit(
                        logging.ERROR,
                        "[W=%d] ERR  prompt=%-4d submit failed: %s",
                        ctx.window_n, ctx.prompt_idx, _sub_exc,
                    )
                except Exception as _sub_exc:
                    self._metrics.record_network_error()
                    _emit(
                        logging.ERROR,
                        "[W=%d] ERR  prompt=%-4d submit unexpected: %s",
                        ctx.window_n, ctx.prompt_idx, _sub_exc,
                    )
                if (
                    self._metrics.submitted
                    and self._metrics.submitted % _SUMMARY_EVERY == 0
                ):
                    _emit(
                        logging.INFO,
                        "=== SUMMARY === %s",
                        self._metrics.summary(self._prompt_stats),
                    )

            while True:
                try:
                    state = await get_window_state_v2(
                        url, client=client,
                        timeout=_STATE_PROBE_TIMEOUT_S,
                    )
                except SubmissionError:
                    await asyncio.sleep(_STATE_POLL_BACKOFF_S)
                    continue
                except Exception as e:
                    logger.debug("state fetch failed: %s", e)
                    await asyncio.sleep(_STATE_POLL_BACKOFF_S)
                    continue

                # Drain completed submit from prior iteration before
                # logging / updating superseded-in-window.
                if _pending_task is not None and _pending_task.done():
                    await _drain_submit(_pending_task, _pending_ctx)
                    _pending_task = None
                    _pending_ctx = None

                # Drain completed pregen producer task.
                self._drain_completed_pregen()

                # Window rolled while a submit was in flight: the response
                # will be window_mismatch / window_not_active anyway, so
                # cancel and free the network.
                if (
                    _pending_task is not None
                    and not _pending_task.done()
                    and _pending_ctx is not None
                    and (
                        state.window_n != _pending_ctx.window_n
                        or state.state != WindowState.OPEN
                    )
                ):
                    _emit(
                        logging.WARNING,
                        "[W=%d] CANCEL pending submit prompt=%-4d "
                        "(submitted at w=%d, now w=%d state=%s) — "
                        "freeing network for next attempt",
                        state.window_n, _pending_ctx.prompt_idx,
                        _pending_ctx.window_n, state.window_n,
                        getattr(state.state, "value", str(state.state)),
                    )
                    _pending_task.cancel()
                    await _drain_submit(_pending_task, _pending_ctx)
                    _pending_task = None
                    _pending_ctx = None

                # Checkpoint pull (no-op when remote ≤ local).
                if state.checkpoint_n > local_n and state.checkpoint_revision:
                    if not _PIPELINE_LOGS_ONLY:
                        logger.info(
                            "checkpoint pull: local_n=%d → remote_n=%d "
                            "revision=%s",
                            local_n, state.checkpoint_n,
                            (state.checkpoint_revision or "")[:12],
                        )
                old_local_n = local_n
                try:
                    local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
                        state=state, local_n=local_n, local_hash=local_hash,
                        local_model=self.hf_model,
                        download_fn=_hf_download,
                        load_fn=self._load_checkpoint,
                    )
                    if local_n > old_local_n:
                        for old_ckpt in range(old_local_n):
                            self._completion_cache.evict_old_checkpoint(old_ckpt)
                except Exception:
                    logger.exception("checkpoint pull failed; keeping local")

                # Cold-start: mirror remote revision so checkpoint_hash
                # isn't empty when the gate runs WRONG_CHECKPOINT checks.
                if (
                    local_n == state.checkpoint_n
                    and not local_hash
                    and state.checkpoint_revision
                ):
                    local_hash = state.checkpoint_revision

                if state.state != WindowState.OPEN:
                    # Idle phase — pre-generate a candidate so the next
                    # OPEN can skip live PICK+GEN and submit ~30s sooner.
                    self._maybe_spawn_pregen(state, rng)
                    await asyncio.sleep(_IDLE_POLL_NON_OPEN_S)
                    continue

                if last_window_n != state.window_n:
                    # Record prior window's OPEN duration before reset
                    # so the picker learns the validator's real delivery rate.
                    if (
                        last_window_n is not None
                        and self._window_open_seen_at is not None
                    ):
                        _prev_open_s = (
                            time.monotonic() - self._window_open_seen_at
                        )
                        self._record_observed_open_duration(_prev_open_s)
                    if last_window_n is not None:
                        _emit(
                            logging.INFO,
                            "=== window %d -> %d (observed_open=%.0fs, "
                            "eff_budget=%.0fs) === | %s",
                            last_window_n, state.window_n,
                            (
                                time.monotonic() - self._window_open_seen_at
                                if self._window_open_seen_at is not None
                                else 0.0
                            ),
                            self._effective_open_budget_s(),
                            self._metrics.summary(self._prompt_stats),
                        )
                    else:
                        _emit(
                            logging.INFO,
                            "=== window %d (first OPEN) === valid=%d/%d "
                            "cooldown=%d ckpt=%d",
                            state.window_n, state.valid_submissions,
                            B_BATCH,
                            len(state.cooldown_prompts), state.checkpoint_n,
                        )
                    last_window_n = state.window_n
                    self._window_open_seen_at = time.monotonic()
                    self._superseded_in_window = set()
                    self._ooz_cohorts_in_window = set()
                    self._own_subs_in_window = 0
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
                    await asyncio.sleep(_STATE_POLL_BACKOFF_S)
                    continue

                # Batch sealed — skip pick to save GPU until window rolls.
                if state.valid_submissions >= B_BATCH:
                    if getattr(self, "_batch_full_notice_window", None) != (
                        state.window_n
                    ):
                        self._batch_full_notice_window = state.window_n
                        _emit(
                            logging.INFO,
                            "[W=%d] SKIP pick valid=%d/%d — batch target "
                            "reached; waiting for next window",
                            state.window_n,
                            state.valid_submissions,
                            B_BATCH,
                        )
                    # GPU is idle while batch is sealed — convert that time
                    # into ready pregen candidates for the next window.
                    self._maybe_spawn_pregen(state, rng)
                    await asyncio.sleep(_IDLE_POLL_BATCH_FULL_S)
                    continue

                # Own-SUB pregen-mode switch: once we've landed our share
                # of submissions in this window, redirect GPU from more
                # live-gen attempts to building a warm queue for the NEXT
                # window. Fixes the FIFO race (`submit too late`) by
                # ensuring next-window's first SUB has a pregen ready at
                # OPEN start instead of paying ~32s of cold live gen.
                if self._own_subs_in_window >= _OWN_SUB_PREGEN_THRESHOLD:
                    if getattr(self, "_own_sub_notice_window", None) != (
                        state.window_n
                    ):
                        self._own_sub_notice_window = state.window_n
                        _emit(
                            logging.INFO,
                            "[W=%d] PREGEN-MODE own_subs=%d/%d — switching "
                            "to pregen-priority for rest of window "
                            "(building queue for W=%d)",
                            state.window_n, self._own_subs_in_window,
                            _OWN_SUB_PREGEN_THRESHOLD, state.window_n + 1,
                        )
                    self._maybe_spawn_pregen(state, rng)
                    await asyncio.sleep(_IDLE_POLL_BATCH_FULL_S)
                    continue

                # Hard pre-pick deadline gate: if even the cheapest possible
                # attempt would land past the OPEN budget, skip PICK entirely.
                _now_age_s = (
                    time.monotonic() - self._window_open_seen_at
                    if self._window_open_seen_at is not None
                    else 0.0
                )
                _hard_http_s = max(
                    _DEADLINE_HTTP_DEFAULT_S,
                    self._metrics.recent_http_avg_s(),
                )
                _hard_proof_s = max(
                    _DEADLINE_PROOF_DEFAULT_S,
                    self._metrics.recent_proof_avg_s(),
                )
                _hard_min_gen_s = (
                    _MIN_GEN_HARD_GATE_PREGEN_S
                    if self._pregen_queue
                    else _MIN_GEN_HARD_GATE_S
                )
                _min_pipeline_s = (
                    _hard_http_s + _hard_proof_s + _hard_min_gen_s
                )
                _eff_budget_s = self._effective_open_budget_s()
                _hard_deadline_s = _eff_budget_s - _min_pipeline_s
                if _now_age_s > _hard_deadline_s:
                    # Throttle WAIT: one line on first entry per window, then 30s heartbeat.
                    _wait_state = getattr(self, "_wait_log_state", None)
                    _wn = state.window_n
                    _emit_wait = False
                    if _wait_state is None or _wait_state[0] != _wn:
                        self._wait_log_state = (_wn, time.monotonic())
                        _emit_wait = True
                    else:
                        if time.monotonic() - _wait_state[1] >= 30.0:
                            self._wait_log_state = (_wn, time.monotonic())
                            _emit_wait = True
                    if _emit_wait:
                        _emit(
                            logging.INFO,
                            "[W=%d] WAIT open_age=%.1fs > hard_deadline=%.1fs "
                            "(eff_budget=%.0fs - http=%.1f - proof=%.1f - "
                            "min_gen=%.1f) — window already too old to land "
                            "a submit; idling until next window",
                            state.window_n, _now_age_s, _hard_deadline_s,
                            _eff_budget_s, _hard_http_s,
                            _hard_proof_s, _hard_min_gen_s,
                        )
                    # GPU is idle during WAIT — pregen for the next window
                    # instead of letting that time go to waste.
                    self._maybe_spawn_pregen(state, rng)
                    await asyncio.sleep(_WAIT_SPIN_S)
                    continue

                cooldown_set = set(state.cooldown_prompts)

                # Pregen fast path: skip live PICK+GEN+SCORE if a
                # pre-generated candidate is ready and still valid.
                pregen = self._try_consume_pregen(state, cooldown_set)

                if pregen is not None:
                    prompt_idx = pregen.prompt_idx
                    level = pregen.level
                    subject = pregen.subject
                    generations = pregen.generations
                    rewards = pregen.rewards
                    completion_lens = pregen.completion_lens
                    sigma = pregen.sigma
                    k_solved = pregen.k_solved
                    in_zone = pregen.in_zone
                    gen_ms = pregen.gen_ms
                    score_ms = 0.0
                    problem = self.env.get_problem(prompt_idx)
                    pregen_age_s = time.monotonic() - pregen.created_at_t
                    open_age_s = (
                        time.monotonic() - self._window_open_seen_at
                        if self._window_open_seen_at is not None
                        else 0.0
                    )
                    _emit(
                        logging.INFO,
                        "[W=%d] PICK prompt=%-4d cohort=(%s,%s) pregen=True "
                        "age=%.1fs k=%d/%d sigma=%.3f in_zone=%s slots=%d/%d "
                        "open_age=%.1fs",
                        state.window_n, prompt_idx,
                        _short_level(level), _short_subject(subject),
                        pregen_age_s, k_solved, M_ROLLOUTS, sigma, in_zone,
                        state.valid_submissions, B_BATCH, open_age_s,
                    )
                    _emit(
                        logging.INFO,
                        "[W=%d] GEN  prompt=%-4d rewards=%s k=%d/%d sigma=%.3f "
                        "in_zone=%s gen=%.1fs (pregen age=%.1fs) score=%.2fs "
                        "clen=%d/%d/%d",
                        state.window_n, prompt_idx,
                        _fmt_rewards(rewards), k_solved, M_ROLLOUTS, sigma,
                        in_zone, gen_ms / 1000.0, pregen_age_s,
                        score_ms / 1000.0,
                        min(completion_lens),
                        sorted(completion_lens)[len(completion_lens) // 2],
                        max(completion_lens),
                    )
                else:
                    try:
                        _open_age_s = _now_age_s
                        _http_avg_s = (
                            self._metrics.recent_http_avg_s()
                            or _DEADLINE_HTTP_DEFAULT_S
                        )
                        _proof_avg_s = (
                            self._metrics.recent_proof_avg_s()
                            or _DEADLINE_PROOF_DEFAULT_S
                        )
                        prompt_idx = pick_prompt_idx(
                            self.env, cooldown_set,
                            rng=rng, stats=self._prompt_stats,
                            current_checkpoint_n=state.checkpoint_n,
                            window_n=state.window_n,
                            slots_filled=state.valid_submissions,
                            superseded_in_window=self._superseded_in_window,
                            open_age_s=_open_age_s,
                            open_budget_s=_eff_budget_s,
                            proof_avg_s=_proof_avg_s,
                            http_avg_s=_http_avg_s,
                            ooz_cohorts_in_window=self._ooz_cohorts_in_window,
                        )
                    except RuntimeError:
                        _emit(
                            logging.INFO,
                            "[W=%d] WAIT env fully blocked cooldown=%d "
                            "superseded=%d env=%d; sleeping 5s",
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
                    open_age_s = (
                        time.monotonic() - self._window_open_seen_at
                        if getattr(self, "_window_open_seen_at", None) is not None
                        else 0.0
                    )
                    _pick_boot = _is_bootstrap_window(state.window_n)
                    _emit(
                        logging.INFO,
                        "[W=%d] PICK prompt=%-4d cohort=(%s,%s) phat=%.2f "
                        "attempts=%d arr=%.2f zone_p=%.2f slots=%d/%d "
                        "open_age=%.1fs (min=%ds, eff_budget=%.0fs) "
                        "ooz_blk=%d bootstrap=%s",
                        state.window_n, prompt_idx,
                        _short_level(level), _short_subject(subject),
                        p_hat, attempts, arr,
                        _zone_probability(p_hat, bootstrap=_pick_boot),
                        state.valid_submissions, B_BATCH,
                        open_age_s, _OPEN_PHASE_MIN_S, _eff_budget_s,
                        len(self._ooz_cohorts_in_window),
                        _pick_boot,
                    )
                    logger.debug(
                        "[W=%d] PICK detail prompt=%d posterior=Beta(%.2f,%.2f) "
                        "level=%r subject=%r",
                        state.window_n, prompt_idx, a, b, level, subject,
                    )

                    if self._window_roll_detector.check_and_update(
                        state.window_n,
                        getattr(state.state, "value", str(state.state)),
                    ):
                        _emit(
                            logging.INFO,
                            "[W=%d] WINDOW_OPEN detected — queuing early "
                            "submission for FIFO advantage",
                            state.window_n,
                        )

                    # GRPO requires M independent stochastic rollouts at T_PROTO.
                    # Replaying one cached completion × M gives σ=0 → OUT_OF_ZONE.
                    t_gen = time.monotonic()
                    generations = await asyncio.to_thread(
                        self._generate_n_rollouts,
                        problem, M_ROLLOUTS,
                        prompt_idx,
                    )
                    gen_ms = (time.monotonic() - t_gen) * 1000.0

                    if _pending_task is not None and _pending_task.done():
                        await _drain_submit(_pending_task, _pending_ctx)
                        _pending_task = None
                        _pending_ctx = None
                    if len(generations) < M_ROLLOUTS:
                        _emit(
                            logging.WARNING,
                            "[W=%d] GEN  prompt=%-4d short: only %d/%d rollouts "
                            "(gen=%.1fs) -> skip",
                            state.window_n, prompt_idx,
                            len(generations), M_ROLLOUTS, gen_ms / 1000.0,
                        )
                        continue

                    t_score = time.monotonic()
                    scored = [self._score_rollout(g, problem) for g in generations]
                    score_ms = (time.monotonic() - t_score) * 1000.0
                    rewards = [r for r, _ in scored]
                    completion_lens = [length for _, length in scored]

                    self._prompt_stats.record_group(
                        prompt_idx, rewards,
                        level=level, subject=subject,
                        checkpoint_n=state.checkpoint_n,
                        window_n=state.window_n,
                        completion_lens=completion_lens,
                    )
                    sigma, k_solved, in_zone = _zone_status(
                        rewards,
                        bootstrap=_is_bootstrap_window(state.window_n),
                    )
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

                # Cherry-pick over-generation: if M=8 is OOZ, draw
                # OVERGEN_EXTRA more honest rollouts at T_PROTO and try
                # to assemble an in-zone subset (each rollout an honest
                # sample, subset choice only biases group sigma).
                if (
                    not in_zone
                    and _OVERGEN_ENABLED
                    and _OVERGEN_EXTRA > 0
                ):
                    overgen_open_age_s = (
                        time.monotonic() - self._window_open_seen_at
                        if self._window_open_seen_at is not None
                        else 0.0
                    )
                    _http_avg_now_s = self._metrics.recent_http_avg_s()
                    if _http_avg_now_s >= _HIGH_HTTP_AVG_S:
                        _emit(
                            logging.INFO,
                            "[W=%d] CHRY prompt=%-4d skipped "
                            "(http_avg=%.1fs ≥ %.1fs — uplink saturated) "
                            "k=%d/%d",
                            state.window_n, prompt_idx,
                            _http_avg_now_s, _HIGH_HTTP_AVG_S,
                            k_solved, M_ROLLOUTS,
                        )
                    elif not (_OVERGEN_MIN_K <= k_solved <= _OVERGEN_MAX_K):
                        _emit(
                            logging.INFO,
                            "[W=%d] CHRY prompt=%-4d skipped "
                            "(k=%d/%d saturated) — over-gen unlikely "
                            "to flip outcome",
                            state.window_n, prompt_idx,
                            k_solved, M_ROLLOUTS,
                        )
                    elif overgen_open_age_s > (
                        # Dynamic deadline = eff_budget - http - proof
                        # - estimated_overgen_cost, floored at 45s.
                        _chry_deadline := max(
                            45.0,
                            min(
                                self._effective_open_budget_s() * 0.8,
                                self._effective_open_budget_s()
                                - max(
                                    _DEADLINE_HTTP_DEFAULT_S,
                                    self._metrics.recent_http_avg_s(),
                                )
                                - max(
                                    _DEADLINE_PROOF_DEFAULT_S,
                                    self._metrics.recent_proof_avg_s(),
                                )
                                - max(
                                    5.0,
                                    (_OVERGEN_EXTRA / M_ROLLOUTS)
                                    * (gen_ms / 1000.0),
                                ),
                            ),
                        )
                    ):
                        _emit(
                            logging.INFO,
                            "[W=%d] CHRY prompt=%-4d skipped "
                            "(open_age=%.1fs>%.1fs deadline) k=%d/%d",
                            state.window_n, prompt_idx,
                            overgen_open_age_s, _chry_deadline,
                            k_solved, M_ROLLOUTS,
                        )
                    else:
                        self._metrics.record_overgen_attempt()
                        t_overgen = time.monotonic()
                        try:
                            extra_gens = await asyncio.to_thread(
                                self._generate_n_rollouts,
                                problem, _OVERGEN_EXTRA,
                                prompt_idx,
                            )
                        except Exception as og_exc:
                            extra_gens = []
                            logger.exception(
                                "over-gen failed: %s", og_exc,
                            )
                        if len(extra_gens) >= 1:
                            extra_scored = [
                                self._score_rollout(g, problem)
                                for g in extra_gens
                            ]
                            extra_rewards = [r for r, _ in extra_scored]
                            extra_lens = [length for _, length in extra_scored]
                            # Record the extras — posterior learns from
                            # all honest draws, not just the shipped subset.
                            self._prompt_stats.record_group(
                                prompt_idx, extra_rewards,
                                level=level, subject=subject,
                                checkpoint_n=state.checkpoint_n,
                                window_n=state.window_n,
                                completion_lens=extra_lens,
                            )
                            pool_gens = generations + extra_gens
                            pool_rewards = rewards + extra_rewards
                            pool_lens = completion_lens + extra_lens
                            chosen_idxs = _find_in_zone_subset(
                                pool_rewards, pool_lens, M_ROLLOUTS,
                                bootstrap=_is_bootstrap_window(state.window_n),
                            )
                            overgen_ms = (time.monotonic() - t_overgen) * 1000.0
                            if chosen_idxs is not None:
                                generations = [pool_gens[i] for i in chosen_idxs]
                                rewards = [pool_rewards[i] for i in chosen_idxs]
                                completion_lens = [
                                    pool_lens[i] for i in chosen_idxs
                                ]
                                sigma, k_solved, in_zone = _zone_status(
                                    rewards,
                                    bootstrap=_is_bootstrap_window(
                                        state.window_n
                                    ),
                                )
                                self._metrics.record_overgen_recovery()
                                _emit(
                                    logging.INFO,
                                    "[W=%d] CHRY prompt=%-4d pool=%d "
                                    "k=%d/%d sigma=%.3f overgen=%.1fs "
                                    "clen=%d/%d/%d -> recovered in_zone",
                                    state.window_n, prompt_idx,
                                    len(pool_gens), k_solved, M_ROLLOUTS,
                                    sigma, overgen_ms / 1000.0,
                                    min(completion_lens),
                                    sorted(completion_lens)[
                                        len(completion_lens) // 2
                                    ],
                                    max(completion_lens),
                                )
                            else:
                                _emit(
                                    logging.INFO,
                                    "[W=%d] CHRY prompt=%-4d pool=%d "
                                    "k_pool=%d/%d overgen=%.1fs -> no "
                                    "in_zone subset",
                                    state.window_n, prompt_idx,
                                    len(pool_gens),
                                    sum(
                                        1 for r in pool_rewards
                                        if r >= _SOLVED_THRESHOLD
                                    ),
                                    len(pool_rewards),
                                    overgen_ms / 1000.0,
                                )

                # Local OUT_OF_ZONE short-circuit — also blacklist the
                # cohort for the rest of the window so the picker stops
                # drawing from saturated cells.
                if not in_zone:
                    self._metrics.record_local_oos()
                    self._ooz_cohorts_in_window.add((level, subject))
                    _sigma_thr = (
                        BOOTSTRAP_SIGMA_MIN
                        if _is_bootstrap_window(state.window_n)
                        else SIGMA_MIN
                    )
                    _emit(
                        logging.INFO,
                        "[W=%d] OOZ  prompt=%-4d sigma=%.3f<%.2f k=%d/%d "
                        "cohort=(%s,%s) -> skip submit (cohort blacklisted "
                        "for window)",
                        state.window_n, prompt_idx, sigma, _sigma_thr,
                        k_solved, M_ROLLOUTS,
                        _short_level(level), _short_subject(subject),
                    )
                    if self._metrics.generated % _SUMMARY_EVERY == 0:
                        _emit(
                            logging.INFO,
                            "=== SUMMARY === %s",
                            self._metrics.summary(self._prompt_stats),
                        )
                    continue

                # CHALLENGE_K gate — completion_length < 32 silently
                # fails LOGPROB_MISMATCH in the async validator worker.
                # SHRT rescue regenerates the short rollouts (honest
                # samples, so statistical validity preserved) when the
                # deadline permits.
                min_clen = min(completion_lens)
                if min_clen < CHALLENGE_K:
                    short_idxs = [
                        i for i, cl in enumerate(completion_lens)
                        if cl < CHALLENGE_K
                    ]
                    shrt_open_age_s = (
                        time.monotonic() - self._window_open_seen_at
                        if self._window_open_seen_at is not None
                        else float("inf")
                    )
                    _http_s_shrt = max(
                        _DEADLINE_HTTP_DEFAULT_S,
                        self._metrics.recent_http_avg_s(),
                    )
                    _proof_s_shrt = max(
                        _DEADLINE_PROOF_DEFAULT_S,
                        self._metrics.recent_proof_avg_s(),
                    )
                    _repl_est_s = max(
                        2.0,
                        (len(short_idxs) / M_ROLLOUTS) * (gen_ms / 1000.0),
                    )
                    _shrt_eff_budget = self._effective_open_budget_s()
                    _shrt_deadline = max(
                        45.0,
                        min(
                            _shrt_eff_budget * 0.8,
                            _shrt_eff_budget
                            - _http_s_shrt - _proof_s_shrt - _repl_est_s,
                        ),
                    )
                    rescued = False
                    _http_avg_now_shrt_s = self._metrics.recent_http_avg_s()
                    _shrt_uplink_skip = (
                        _http_avg_now_shrt_s >= _HIGH_HTTP_AVG_S
                    )
                    if (
                        not _shrt_uplink_skip
                        and len(short_idxs) <= 2
                        and shrt_open_age_s <= _shrt_deadline
                    ):
                        try:
                            repl_gens = await asyncio.to_thread(
                                self._generate_n_rollouts,
                                problem, len(short_idxs),
                                prompt_idx,
                            )
                        except Exception as _shrt_exc:
                            repl_gens = []
                            logger.debug("SHRT rescue gen failed: %s", _shrt_exc)
                        if len(repl_gens) == len(short_idxs):
                            repl_scored = [
                                self._score_rollout(g, problem)
                                for g in repl_gens
                            ]
                            repl_rewards = [r for r, _ in repl_scored]
                            repl_lens = [l for _, l in repl_scored]
                            self._prompt_stats.record_group(
                                prompt_idx, repl_rewards,
                                level=level, subject=subject,
                                checkpoint_n=state.checkpoint_n,
                                window_n=state.window_n,
                                completion_lens=repl_lens,
                            )
                            new_gens = list(generations)
                            new_rews = list(rewards)
                            new_clens = list(completion_lens)
                            for pos, (rg, rr, rl) in zip(
                                short_idxs,
                                zip(repl_gens, repl_rewards, repl_lens),
                            ):
                                new_gens[pos] = rg
                                new_rews[pos] = rr
                                new_clens[pos] = rl
                            new_min = min(new_clens)
                            new_sigma, new_k, new_inzone = _zone_status(
                                new_rews,
                                bootstrap=_is_bootstrap_window(state.window_n),
                            )
                            if new_min >= CHALLENGE_K and new_inzone:
                                generations = new_gens
                                rewards = new_rews
                                completion_lens = new_clens
                                sigma = new_sigma
                                k_solved = new_k
                                in_zone = True
                                rescued = True
                                _emit(
                                    logging.INFO,
                                    "[W=%d] SHRT prompt=%-4d rescued %d short "
                                    "rollout(s) k=%d/%d sigma=%.3f "
                                    "clen=%d/%d/%d",
                                    state.window_n, prompt_idx,
                                    len(short_idxs), k_solved, M_ROLLOUTS,
                                    sigma,
                                    min(completion_lens),
                                    sorted(completion_lens)[
                                        len(completion_lens) // 2
                                    ],
                                    max(completion_lens),
                                )
                            else:
                                _emit(
                                    logging.INFO,
                                    "[W=%d] SHRT prompt=%-4d rescue failed "
                                    "(new_min=%d in_zone=%s) -> skip",
                                    state.window_n, prompt_idx,
                                    new_min, new_inzone,
                                )
                    if not rescued:
                        _shrt_skip_reason = (
                            f" (http_avg={_http_avg_now_shrt_s:.1f}s "
                            f"≥ {_HIGH_HTTP_AVG_S:.1f}s — uplink saturated)"
                            if _shrt_uplink_skip
                            else ""
                        )
                        _emit(
                            logging.INFO,
                            "[W=%d] SHRT prompt=%-4d min_clen=%d<%d -> skip submit "
                            "(would fail validator LOGPROB_MISMATCH)%s",
                            state.window_n, prompt_idx, min_clen, CHALLENGE_K,
                            _shrt_skip_reason,
                        )
                        if self._metrics.generated % _SUMMARY_EVERY == 0:
                            _emit(
                                logging.INFO,
                                "=== SUMMARY === %s",
                                self._metrics.summary(self._prompt_stats),
                            )
                        continue

                # Pre-proof window guard — must check state != OPEN to
                # catch window_not_active (same window_n, post-OPEN phase).
                _pre_proof_check_t: float | None = None
                _pre_proof_probe_start = time.monotonic()
                try:
                    early_state = await get_window_state_v2(
                        url, client=client,
                        timeout=_STATE_PROBE_TIMEOUT_S,
                    )
                    _pre_proof_check_t = time.monotonic()
                    _early_closed = (
                        early_state.window_n != state.window_n
                        or early_state.state != WindowState.OPEN
                    )
                    if _early_closed:
                        _emit(
                            logging.WARNING,
                            "[W=%d] SKIP prompt=%-4d window no longer "
                            "accepting (w=%d state=%s) before proof "
                            "(gen=%.1fs) — saved proof build",
                            state.window_n, prompt_idx,
                            early_state.window_n,
                            getattr(early_state.state, "value",
                                    str(early_state.state)),
                            gen_ms / 1000.0,
                        )
                        continue
                except SubmissionError as _early_exc:
                    _pre_probe_ms = (
                        time.monotonic() - _pre_proof_probe_start
                    ) * 1000.0
                    _emit(
                        logging.INFO,
                        "[W=%d] STATE-PROBE pre-proof failed in %.1fs "
                        "(%s); proceeding to proof — post-proof check "
                        "will catch any roll",
                        state.window_n, _pre_probe_ms / 1000.0, _early_exc,
                    )

                # Build GRAIL proofs — dispatched off-loop so the event
                # loop can drain the previous submit's response in parallel.
                t_proof = time.monotonic()
                try:
                    rollout_submissions = await asyncio.to_thread(
                        self._build_rollout_submissions,
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
                self._metrics.record_proof_latency(proof_ms)
                logger.debug(
                    "[W=%d] proof done prompt=%d proof_ms=%.0f mode=%s",
                    state.window_n, prompt_idx, proof_ms,
                    "batched" if self._metrics.batched_proof_active else "per-rollout",
                )

                merkle_root = _compute_merkle_root(rollout_submissions)

                # Post-proof /state recheck. Fast-path skip when the
                # pre-proof check was recent AND proof was cheap — a
                # window transition in that interval is negligible.
                _can_skip_post_check = (
                    _pre_proof_check_t is not None
                    and (time.monotonic() - _pre_proof_check_t)
                    < _SKIP_POST_CHECK_MAX_WALL_S
                    and (proof_ms / 1000.0) < _SKIP_POST_CHECK_MAX_PROOF_S
                )
                if _can_skip_post_check:
                    logger.debug(
                        "[W=%d] post-proof state check skipped "
                        "(pre-check age=%.1fs, proof=%.1fs) — fast path",
                        state.window_n,
                        time.monotonic() - (_pre_proof_check_t or time.monotonic()),
                        proof_ms / 1000.0,
                    )
                else:
                    _post_proof_probe_start = time.monotonic()
                    try:
                        fresh_state = await get_window_state_v2(
                            url, client=client,
                            timeout=_STATE_PROBE_TIMEOUT_S,
                        )
                        _post_closed = (
                            fresh_state.window_n != state.window_n
                            or fresh_state.state != WindowState.OPEN
                        )
                        if _post_closed:
                            _emit(
                                logging.WARNING,
                                "[W=%d] SKIP prompt=%-4d window no longer "
                                "accepting (w=%d state=%s) after proof "
                                "(gen=%.1fs proof=%.1fs)",
                                state.window_n, prompt_idx,
                                fresh_state.window_n,
                                getattr(fresh_state.state, "value",
                                        str(fresh_state.state)),
                                gen_ms / 1000.0, proof_ms / 1000.0,
                            )
                            continue
                    except SubmissionError as e:
                        # Probe timed out — submit anyway and let the
                        # validator decide. Better than skipping a
                        # potentially-good submission.
                        _post_probe_ms = (
                            time.monotonic() - _post_proof_probe_start
                        ) * 1000.0
                        _emit(
                            logging.INFO,
                            "[W=%d] STATE-PROBE post-proof failed in %.1fs "
                            "(%s); firing submit anyway",
                            state.window_n, _post_probe_ms / 1000.0, e,
                        )

                if local_n < state.checkpoint_n:
                    _emit(
                        logging.WARNING,
                        "[W=%d] SKIP prompt=%-4d submit: local_n=%d < "
                        "checkpoint_n=%d (would WRONG_CHECKPOINT)",
                        state.window_n, prompt_idx, local_n,
                        state.checkpoint_n,
                    )
                    continue
                if (
                    local_hash
                    and state.checkpoint_revision
                    and local_hash != state.checkpoint_revision
                ):
                    _emit(
                        logging.WARNING,
                        "[W=%d] SKIP prompt=%-4d submit: checkpoint revision "
                        "mismatch local=%s remote=%s (would WRONG_CHECKPOINT)",
                        state.window_n, prompt_idx,
                        local_hash[:12],
                        state.checkpoint_revision[:12],
                    )
                    continue

                request = BatchSubmissionRequest(
                    miner_hotkey=self.wallet.hotkey.ss58_address,
                    prompt_idx=prompt_idx,
                    window_start=state.window_n,
                    merkle_root=merkle_root,
                    rollouts=rollout_submissions,
                    checkpoint_hash=local_hash,
                )

                # Never run two submits concurrently.
                if _pending_task is not None and not _pending_task.done():
                    await _drain_submit(_pending_task, _pending_ctx)
                    _pending_task = None
                    _pending_ctx = None

                # Snapshot open_age + gpu at FIRE time, not response time.
                _sub_open_age = (
                    time.monotonic() - self._window_open_seen_at
                    if self._window_open_seen_at is not None
                    else 0.0
                )
                _pending_ctx = _SubmitCtx(
                    prompt_idx=prompt_idx,
                    window_n=state.window_n,
                    rewards=rewards,
                    k_solved=k_solved,
                    sigma=sigma,
                    merkle_root=merkle_root,
                    proof_mode=(
                        "batched"
                        if self._metrics.batched_proof_active
                        else "per-rollout"
                    ),
                    gen_ms=gen_ms,
                    proof_ms=proof_ms,
                    gpu_pre=_gpu_mem_compact(self.vllm_gpu),
                    open_age_s=_sub_open_age,
                    tokens=list(rollout_submissions[0].tokens) if rollout_submissions else [],
                    checkpoint_n=state.checkpoint_n,
                )
                # Fire HTTP in background, drain at the next iteration
                # so GPU can start the next PICK → GEN immediately.
                _pending_task = asyncio.create_task(
                    _fire_submit(request),
                    name=f"submit-w{state.window_n}-p{prompt_idx}",
                )
                logger.debug(
                    "[W=%d] submit task launched prompt=%d open_age=%.1fs; "
                    "continuing to next pick immediately",
                    state.window_n, prompt_idx, _sub_open_age,
                )

        # Drain the in-flight submit so results + metrics are final.
        if _pending_task is not None:
            try:
                await _drain_submit(_pending_task, _pending_ctx)
            except asyncio.CancelledError:
                pass
            _pending_task = None

        # Cancel any in-flight pregen so it doesn't outlive the loop.
        if self._pregen_task is not None and not self._pregen_task.done():
            self._pregen_task.cancel()
            try:
                await self._pregen_task
            except (asyncio.CancelledError, Exception):
                pass
            self._pregen_task = None

        return results

    def _record_observed_open_duration(self, duration_s: float) -> None:
        """Record prior window's OPEN duration. Discards < 30s (miner
        restarted mid-window) and clamps at _OPEN_PHASE_BUDGET_S.
        """
        if duration_s <= 0 or duration_s < 30.0:
            return
        clamped = min(duration_s, float(_OPEN_PHASE_BUDGET_S))
        self._observed_open_durations_s.append(clamped)

    def _effective_open_budget_s(self) -> float:
        """Realistic OPEN budget for the picker.

        n<3 obs: fall back to _OPEN_PHASE_BUDGET_S constant.
        3 ≤ n < 5: use min (no percentile confidence).
        n ≥ 5: P25 (sorted[n//4]) × SAFETY_FACTOR, floored at
        MIN_FLOOR_S, capped at BUDGET_S. P25 (vs min) prevents a single
        short window from pinning the budget when the validator's
        actual range is much wider.
        """
        n = len(self._observed_open_durations_s)
        if n < 3:
            return float(_OPEN_PHASE_BUDGET_S)
        sorted_durations = sorted(self._observed_open_durations_s)
        if n < 5:
            anchor = sorted_durations[0]
        else:
            anchor = sorted_durations[n // 4]
        effective = anchor * _OBSERVED_OPEN_SAFETY_FACTOR
        effective = max(effective, _OBSERVED_OPEN_MIN_FLOOR_S)
        effective = min(effective, float(_OPEN_PHASE_BUDGET_S))
        return effective

    def _maybe_check_proof_fallback(self) -> None:
        """Switch to per-rollout proofs after consecutive batched
        GRAIL_FAILs. One-way — never re-enabled in-session.
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
    # Pre-generation pipeline — produce during idle phases, consume at OPEN
    # ------------------------------------------------------------------

    def _drain_completed_pregen(self) -> None:
        """If the pregen task finished, collect any exception and clear."""
        if self._pregen_task is None or not self._pregen_task.done():
            return
        exc = self._pregen_task.exception()
        if exc is not None and not isinstance(exc, asyncio.CancelledError):
            logger.exception(
                "pregen task raised: %s", exc, exc_info=exc,
            )
        self._pregen_task = None

    def _try_consume_pregen(
        self,
        state,
        cooldown_set: set[int],
    ) -> _PregenCandidate | None:
        """Pop a fresh, valid pregen candidate or return None.

        Discards entries that have aged out, were generated under a
        different checkpoint, or whose prompt has since entered cooldown
        or the per-window superseded blacklist. Prefers in_zone candidates
        so an OPEN slot isn't burned on a guaranteed-fail submission
        when an acceptable one is sitting in the queue.
        """
        if not self._pregen_queue:
            return None
        now = time.monotonic()
        self._pregen_queue = [
            c for c in self._pregen_queue
            if now - c.created_at_t < _PREGEN_MAX_AGE_S
            and c.created_at_checkpoint_n == state.checkpoint_n
            and c.prompt_idx not in cooldown_set
            and c.prompt_idx not in self._superseded_in_window
        ]
        if not self._pregen_queue:
            return None
        boot = _is_bootstrap_window(state.window_n)
        while self._pregen_queue:
            for i, c in enumerate(self._pregen_queue):
                sigma, k_solved, in_zone = _zone_status(
                    c.rewards, bootstrap=boot,
                )
                if in_zone:
                    c.sigma = sigma
                    c.k_solved = k_solved
                    c.in_zone = in_zone
                    return self._pregen_queue.pop(i)
            dropped = self._pregen_queue.pop(0)
            _emit(
                logging.INFO,
                "[W=%d] PREGEN drop prompt=%-4d (no longer in-zone under "
                "bootstrap=%s) queue_remaining=%d",
                state.window_n, dropped.prompt_idx, boot,
                len(self._pregen_queue),
            )
        return None

    def _maybe_spawn_pregen(
        self,
        state,
        rng: _random.Random,
    ) -> None:
        """Spawn a pregen producer task if eligible (no task running,
        queue has room). Fire-and-forget; result is drained next loop tick.
        """
        if self._pregen_task is not None and not self._pregen_task.done():
            return
        if len(self._pregen_queue) >= _PREGEN_MAX_QUEUE:
            return
        self._pregen_task = asyncio.create_task(
            self._produce_pregen(state, rng),
            name=f"pregen-w{state.window_n}",
        )
        _emit(
            logging.INFO,
            "[W=%d] PREGEN spawn (state=%s, queue=%d)",
            state.window_n,
            getattr(state.state, "value", str(state.state)),
            len(self._pregen_queue),
        )

    async def _produce_pregen(
        self,
        state,
        rng: _random.Random,
    ) -> None:
        """Pick + gen + score one candidate and push to the queue.

        Records posterior + metrics just like the live pipeline. Picker
        runs with conservative deadline defaults (full OPEN budget,
        open_age=0) since we're generating before OPEN even starts.
        """
        cooldown_set = set(state.cooldown_prompts)
        try:
            prompt_idx = pick_prompt_idx(
                self.env, cooldown_set,
                rng=rng, stats=self._prompt_stats,
                current_checkpoint_n=state.checkpoint_n,
                window_n=state.window_n,
                slots_filled=0,
                superseded_in_window=self._superseded_in_window,
                open_age_s=0.0,
                open_budget_s=float(_OPEN_PHASE_BUDGET_S),
                proof_avg_s=(
                    self._metrics.recent_proof_avg_s()
                    or _DEADLINE_PROOF_DEFAULT_S
                ),
                http_avg_s=(
                    self._metrics.recent_http_avg_s()
                    or _DEADLINE_HTTP_DEFAULT_S
                ),
                ooz_cohorts_in_window=self._ooz_cohorts_in_window,
            )
        except RuntimeError:
            return

        problem = self.env.get_problem(prompt_idx)
        level = problem.get("level", "")
        subject = problem.get("subject", "")

        t_gen = time.monotonic()
        try:
            generations = await asyncio.to_thread(
                self._generate_n_rollouts,
                problem, M_ROLLOUTS, prompt_idx,
            )
        except Exception:
            logger.exception("pregen generation failed for prompt=%d", prompt_idx)
            return
        gen_ms = (time.monotonic() - t_gen) * 1000.0

        if len(generations) < M_ROLLOUTS:
            return

        scored = [self._score_rollout(g, problem) for g in generations]
        rewards = [r for r, _ in scored]
        completion_lens = [length for _, length in scored]

        self._prompt_stats.record_group(
            prompt_idx, rewards,
            level=level, subject=subject,
            checkpoint_n=state.checkpoint_n,
            window_n=state.window_n,
            completion_lens=completion_lens,
        )
        sigma, k_solved, in_zone = _zone_status(
            rewards,
            bootstrap=_is_bootstrap_window(state.window_n),
        )
        self._metrics.record_generation(k_solved)
        self._maybe_persist_stats()

        # Drop out-of-zone pregens. Empirically (W=1221 incident) consuming
        # OOZ pregens burns OPEN slots on cherry-pick attempts that rarely
        # recover, blocks the queue from filling with usable candidates,
        # and prevents the live picker from running with fresh window state.
        # The posterior still benefits — record_group() above ran regardless.
        if not in_zone:
            _emit(
                logging.INFO,
                "[W=%d] PREGEN drop prompt=%-4d cohort=(%s,%s) k=%d/%d "
                "sigma=%.3f in_zone=False gen=%.1fs (out-of-zone — not queued)",
                state.window_n, prompt_idx,
                _short_level(level), _short_subject(subject),
                k_solved, M_ROLLOUTS, sigma, gen_ms / 1000.0,
            )
            return

        candidate = _PregenCandidate(
            prompt_idx=prompt_idx,
            level=level,
            subject=subject,
            generations=generations,
            rewards=rewards,
            completion_lens=completion_lens,
            sigma=sigma,
            k_solved=k_solved,
            in_zone=in_zone,
            gen_ms=gen_ms,
            created_at_t=time.monotonic(),
            created_at_window_n=state.window_n,
            created_at_checkpoint_n=state.checkpoint_n,
        )
        self._pregen_queue.append(candidate)

        _emit(
            logging.INFO,
            "[W=%d] PREGEN ready prompt=%-4d cohort=(%s,%s) k=%d/%d "
            "sigma=%.3f in_zone=True gen=%.1fs queue=%d",
            state.window_n, prompt_idx,
            _short_level(level), _short_subject(subject),
            k_solved, M_ROLLOUTS, sigma, gen_ms / 1000.0,
            len(self._pregen_queue),
        )

    def _load_checkpoint(self, local_path: str):
        """Reload both hf_model and vllm_model from local_path. vLLM
        adapter uses .reload(); HF fallback rebuilds from_pretrained.
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

    def _generate_n_rollouts(
        self,
        problem,
        n: int,
        prompt_idx: int | None = None,
    ) -> list[dict]:
        """Run ``n`` independent samples in one batched .generate() call.
        Truncates each row at the first post-prompt EOS. With prompt_idx
        and enough prior samples, tightens max_new_tokens via
        ``_prompt_stats.completion_budget`` to bound the slowest rollout
        (which dictates batched gen wall-clock).
        """
        import torch

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        # Verifier accepts max-length termination iff
        # prompt_length + completion_length >= MAX_NEW_TOKENS_PROTOCOL_CAP.
        budget = max(
            512,
            MAX_NEW_TOKENS_PROTOCOL_CAP - prompt_length,
        )
        effective_max_new = min(self.max_new_tokens, budget)

        # Posterior-budget shortening disabled (2026-05-13). Empirical
        # competitor analysis showed top miners consistently generate to
        # the protocol max (8192 tokens) — the validator extracts the
        # boxed answer from anywhere in the completion, and max-length
        # rollouts dominate the score table. Shortening max_new_tokens
        # to ~2× historical mean was a GPU-time optimization that costs
        # us in_zone hits whenever the model would have boxed the answer
        # past the shortened cap. Leaving the helper in place for future
        # diagnostics; just not consuming it here.

        # Tier 3 soft cap: bound the slowest rollout's wall-clock without
        # sacrificing the max-length strategy entirely. 6000 is the
        # default — past the 95th percentile of boxed-answer positions
        # observed on Qwen3-4B + MATH. Override with the
        # RELIQUARY_SOFT_MAX_NEW_TOKENS env var; set to 0 to disable.
        if _SOFT_MAX_NEW_TOKENS_CAP > 0:
            effective_max_new = min(effective_max_new, _SOFT_MAX_NEW_TOKENS_CAP)

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
        # Walk from the front to truncate at first stop token — handles
        # pad_id == eos_id (Qwen3) where back-stripping would eat the
        # real EOS. finish=length rollouts have cut=None and keep all
        # tokens (max_new of real tokens, no pads).
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

    def _build_rollout_submissions(
        self,
        generations: list[dict],
        rewards: list[float],
        randomness: str,
    ) -> list[RolloutSubmission]:
        """Build RolloutSubmissions — batched path when active, per-rollout
        fallback after consecutive GRAIL_FAILs.
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
        """Single-rollout GRAIL forward + commitments + log-probs +
        signature. Bit-identical to the validator's single-sequence
        verify_commitment_proofs path.
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
        # Vectorized gather → ONE GPU→CPU sync via .tolist() instead of
        # one .item() per completion token. On a 6000-token rollout this
        # eliminates ~6000 individual cuda syncs (~0.5-1s) and matches
        # the per-rollout single .item() loop bit-for-bit.
        seq_len = len(all_tokens)
        if seq_len > prompt_length:
            row_idx = torch.arange(
                prompt_length - 1, seq_len - 1,
                device=log_probs.device,
            )
            col_idx = torch.tensor(
                all_tokens[prompt_length:seq_len],
                device=log_probs.device, dtype=torch.long,
            )
            token_logprobs: list[float] = log_probs[row_idx, col_idx].tolist()
        else:
            token_logprobs = []

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
        """ONE padded HF forward for all M rollouts.

        Bit-compat with the validator's single-sequence
        verify_commitment_proofs: attention_mask zeros out pad positions
        so per-token ops (layernorm, projections, softmax) produce the
        same activations at real positions as a [1, real_len] forward
        would. Tiny FP drift from kernel tile scheduling stays inside
        PROOF_SKETCH_TOLERANCE_BASE. Two consecutive GRAIL_FAILs trip
        the per-rollout fallback.

        Slicing: hidden_states_batch[i, :real_len_i] matches the
        single-sequence hidden state tensor.
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        if not generations:
            return []

        # Bail to per-rollout if the prompt prefix isn't identical
        # across rollouts — otherwise the validator sees an out-of-band proof.
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

        # Build padded tokens + mask on CPU, then ship to GPU.
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
            # Vectorized gather — see _build_grail_commit_single for
            # rationale. ~48,000 .item() syncs across M=8 rollouts collapse
            # to M tolist() round-trips, saving 0.5-3s of proof time on
            # long completions.
            if real_len > prompt_length:
                row_idx = torch.arange(
                    prompt_length - 1, real_len - 1,
                    device=log_probs.device,
                )
                col_idx = torch.tensor(
                    all_tokens[prompt_length:real_len],
                    device=log_probs.device, dtype=torch.long,
                )
                token_logprobs: list[float] = (
                    log_probs[row_idx, col_idx].tolist()
                )
            else:
                token_logprobs = []

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

    async def _randomness_for_window(
        self, subtensor, window_n: int, use_drand: bool
    ) -> str:
        """Cache window randomness across polling iterations."""
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
