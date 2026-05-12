"""Miner engine v4 — accuracy picker + batched GRAIL + serial pipeline.

DEPLOYMENT: this file replaces ``reliquary/miner/engine.py``. On the miner
box, alongside the math env patch AND the v4.2 submitter overlay:

    cp engine-v4.py    /root/reliquary/reliquary/miner/engine.py
    cp math-v4.py      /root/reliquary/reliquary/environment/math.py
    cp submitter-v4.py /root/reliquary/reliquary/miner/submitter.py

The submitter overlay is REQUIRED for multi-validator broadcast
(``discover_validator_urls`` and ``submit_batch_v2_multi``). Without
it, the engine falls back to single-validator submission and logs a
warning at startup — the engine still runs but you miss the ~N×
weight EMA multiplier.

OPTIONAL DEPENDENCY: ``pip install orjson`` to unlock 3-5× faster
JSON encoding on the multi-MB /submit payload. The submitter
auto-detects orjson and falls back to stdlib ``json`` if missing,
so it's a free speedup whenever you can install it.

What v4.2 adds (on top of v4)
=============================

1. **Multi-validator broadcast.** v4.0 picked ONE validator from the
   metagraph and submitted there. Every other permitted validator
   never saw our submissions → scored us 0 → our final on-chain
   weight (averaged across the validator set via consensus) collapsed
   by ~N. v4.2 discovers all permitted validators (up to
   ``max_validators``, default 5) and broadcasts each /submit to all
   of them in parallel. Each acceptor contributes one EMA tick to
   our weight — directly closing the gap to top miners.

2. **Window-roll cancellation.** When the OPEN window rolls (either
   to a new window_n or to TRAINING/PUBLISHING) while a submit is
   still in flight, the response is guaranteed to be window_mismatch
   / window_not_active. v4.2 detects this in the poll loop and
   cancels the doomed task, freeing the network for the next
   attempt.

3. **Hard pre-PICK deadline gate.** Below the picker's existing
   per-prompt deadline penalty, v4.2 adds a HARD gate that skips
   the pick entirely when ``open_age + http_avg + proof_avg + 5s``
   exceeds the OPEN-phase budget. v4.0 would still PICK / GEN /
   PROOF and only abort at the post-proof state recheck — wasting
   30-60 s of GPU time on a doomed pipeline.

4. **Fail-fast HTTP.** Pairs with submitter-v4's single-attempt
   submit and 30 s default timeout (was 60-120 s with 3 retries).
   A slow validator no longer wedges the OPEN window — we move on
   in 30 s max.

Time-efficiency additions (v4.2 polish)
=======================================

Generation, proof building, and HTTP submission must ALL fit inside
the ~60 s OPEN-window budget. Every saved second is one more
prompt slot we can claim. These changes attack each phase:

5. **One-shot payload encoding.** submitter-v4's
   ``submit_batch_v2_multi`` pays ``pydantic.model_dump(mode="json")``
   exactly once for the whole broadcast (was: once per validator URL,
   ~0.5-2 s × N). orjson is used when installed (``pip install orjson``)
   for a further 3-5× JSON-encode speedup on the >5 MB body.

6. **Validator-connection prewarm.** A cheap GET /state probes every
   discovered validator at startup so the TCP/TLS handshake (~50-300
   ms per host on the cold path) is amortized outside the OPEN
   window. The first /submit lands on already-keep-alived sockets.

7. **Off-loop GPU dispatch.** The blocking vLLM ``generate`` (~10-30 s)
   and HF proof forward (~3-5 s) now run via ``asyncio.to_thread``.
   Previously they froze httpx's reactor, silently delaying any
   in-flight submit's TCP send/recv by the full GPU-call duration.
   With the dispatch off-loop, the previous submit's response can
   land while the GPU is busy — so the metrics + window-roll
   detection stay current.

8. **Length-aware deadline scoring.** The picker now scales
   ``http_avg_s`` by ``avg_completion_len / 4096`` (clamped
   [0.5, 2.5]) so long-completion prompts pay a proportionally
   larger upload-time penalty than short-completion ones. Under
   tight budgets the picker steers toward fast-uploading prompts.

9. **Uplink-saturated overgen/SHRT skip.** When the rolling
   ``http_avg_s`` exceeds ``_HIGH_HTTP_AVG_S = 45 s``, both
   over-generation and short-rollout rescue are skipped up front:
   no point spending another 10-30 s of GPU time on a submission
   that can't fit through the uplink anyway.

10. **Fast-path post-proof state check.** When the pre-proof
    /state check was < 4 s ago AND proof took < 4 s, the post-proof
    recheck is skipped — saving a 50-300 ms HTTP round-trip with
    near-zero risk of a missed window transition in that window.

11. **Mid-gen drain.** After each ``await asyncio.to_thread(gen)``,
    we opportunistically drain any completed previous submit task.
    Lets accept/reject + SUMMARY logs surface within tens of ms of
    the network completing, not at the start of the NEXT iteration.

Window-rollover-miss fix (v4.2 post-mortem)
===========================================

Observation from a real W=954 trace: the engine picked an L5/Precalculus
prompt at open_age=43 s, took 41 s to generate, 2.6 s to proof, then
SKIPPED with "window no longer accepting" after a 66 s gap that didn't
show up in proof_ms. Root cause was three layered bugs:

a. ``_OPEN_PHASE_BUDGET_S = 240`` was a hard-coded over-estimate.
   The validator actually rolled OPEN → TRAINING at ~153 s, so the
   picker's deadline math thought it had 200 s of slack when it
   actually had ~110 s.

b. The pre- and post-proof ``/state`` probes used the AsyncClient's
   default 30 s timeout. When the validator hit a brief congestion
   spike at the window-roll boundary, each probe burned its full
   timeout (× 2 with the one-retry policy = up to 60 s combined).

c. The pre-proof state-probe exception path was logged at DEBUG,
   so the operator could not see that ``/state`` had wedged.

Fixes:

12. **Adaptive observed-OPEN budget.** ``_record_observed_open_duration``
    runs at every window-roll boundary; ``_effective_open_budget_s``
    returns ``min(_OPEN_PHASE_BUDGET_S, observed_min × 0.9)``
    floored at 90 s, so after ≥ 3 windows the picker uses what
    the validator actually delivers — not the optimistic ceiling.

13. **Tight ``/state`` per-request timeout.** ``_STATE_PROBE_TIMEOUT_S
    = 5 s`` is passed explicitly to every pre-proof, post-proof, and
    main-loop /state poll. Two attempts via the submitter's retry
    policy cap any single probe at ~10.5 s wall-clock even when the
    validator is at its slowest. (The /submit upload still uses the
    larger ``http_timeout_s`` default — that one moves real bytes.)

14. **Visible state-probe failures.** Pre-proof + post-proof
    ``SubmissionError`` is now INFO-level
    (``[W=…] STATE-PROBE pre/post-proof failed …``). The post-proof
    failure path also falls THROUGH to firing the submit rather than
    skipping — if /state is wedged we don't know the window state,
    so we submit and let the validator decide.

15. **Tighter slack thresholds in the picker.** Deadline penalties
    now require positive headroom: < 5 s slack → hard penalty,
    < 20 s slack → soft penalty (was: < -5 s / < 0 s, i.e. only
    prompts EXPECTED to bust the deadline got demoted). With an
    accurate ``_effective_open_budget_s``, this finally bites and
    long-completion prompts get correctly down-ranked when the
    window is already half-spent.

Saturated-cohort burn fix (W=957 post-mortem)
=============================================

A second log trace showed the picker drawing four high-phat (0.85-0.92)
prompts from saturated cohorts in a single window, all producing
k=8/8 → OOZ → no submit. The fundamental cause is the model being
strong enough that most MATH cohorts saturate at k=8, but the
picker had no mechanism to learn this in-window.

16. **Per-window OOZ-cohort blacklist.** Every OOZ skip records the
    ``(level, subject)`` of the wasted attempt into
    ``self._ooz_cohorts_in_window``. The picker multiplies any
    candidate from a blacklisted cohort by ``_OOZ_COHORT_PENALTY``
    (0.20 ≈ 5× preference for fresh cohorts). The set resets at
    every window-roll boundary. The OOZ log now prints
    ``cohort=(...) -> ... (cohort blacklisted for window)`` and the
    PICK log prints ``ooz_blk=N`` for visibility.

17. **Second-chance expanded candidate pool.** When the initial K=64
    draw yields a best score < ``_SECOND_CHANCE_SCORE_THRESHOLD = 0.15``
    (which is what happens when most cohorts in the random sample
    are saturated), the picker expands to K=256 candidates for a
    bigger lottery. Costs ~hundreds of µs of extra CPU; gains a
    real chance of hitting a non-saturated cohort. Race mode skips
    this — there, speed beats optimality.

Picker-sharpening fixes (W=954-W=964 post-mortem)
=================================================

A third trace (10 windows, 22 picks, 1 successful submit) showed the
picker drawing repeatedly from saturated cohorts even AFTER the OOZ
blacklist was active. Root cause: the discrimination signals were
too weak for the picker to skip the saturated cohorts in the first
half of a session.

18. **Faster cohort learning.** ``cohort_inzone_rate`` ``min_groups``
    drops from 5 → 2. With only 2 observations the Laplace smoothing
    still controls the variance, but the picker stops being
    completely blind to cohort quality in the first 2 windows.

19. **Squared ciz_rate penalty.** The picker multiplies ``score *=
    ciz_rate ** 2`` instead of ``ciz_rate``. A balanced cohort
    (ciz=0.7) scores 49 % under squared vs 70 % under linear; a
    saturated cohort (ciz=0.2) scores 4 % vs 20 %. The discrimination
    factor jumps from 3.5x → 12x so the picker reliably down-ranks
    saturated cohorts once any signal exists.

20. **Wider candidate pool.** ``_CANDIDATES_DEFAULT`` 32 → 64;
    ``_CANDIDATES_SECOND_CHANCE`` 128 → 256;
    ``_SECOND_CHANCE_SCORE_THRESHOLD`` 0.05 → 0.15. Triggering
    expansion at score < 0.15 (rather than < 0.05) means most picks
    now consider 256 candidates instead of 64, virtually guaranteeing
    the picker sees the best non-saturated cohort if one exists in
    the env. Combined CPU cost remains under 1 ms per pick.

21. **Tighter observed-OPEN floor.** ``_OBSERVED_OPEN_MIN_FLOOR_S``
    90 → 75 and ``_OBSERVED_OPEN_SAFETY_FACTOR`` 0.90 → 0.85. The
    W=954-964 trace had real OPEN windows as short as 86 s (W=955,
    W=960) — a 90 s floor over-estimated them. The lower numbers
    let the adaptive budget hug the actual edge so the picker's
    deadline math doesn't accept long-completion prompts that
    will overshoot.

23. **P25 anchor for eff_budget (W=973-989 post-mortem).** The
    original ``_effective_open_budget_s`` used ``min(observed)``
    which pinned eff_budget=91 s for 18 consecutive windows after
    a single 107 s observation, even though the validator actually
    ran 107-250 s windows (mean ≈ 180 s). The picker burned 50-150 s
    of WAIT per long window. ``min`` → ``sorted[n // 4]`` (25th
    percentile) at ``n ≥ 5`` observations so a single short outlier
    no longer dominates; the deque (20 windows) still holds enough
    history that a SUSTAINED short-window regime shifts the
    percentile down within ~5 windows. For ``n < 5`` we still use
    ``min`` because percentile estimates on tiny samples are noise.

24. **Throttled WAIT log.** Previously the hard-deadline gate
    logged a near-identical WAIT line every 2 s poll. Now we log
    once on first entry per window (with the reason), then a 30 s
    heartbeat — keeps the operator informed without drowning out
    the per-window PICK / GEN / SUB lines.

22. **Continuous length-aware boost.** Picker score gets a smooth
    ``× (1 + 0.10 × (1 - avg_len/4096))`` multiplier clamped to
    [0.85, 1.15]. The existing deadline penalty is binary (kicks
    in only below the slack threshold); the smooth boost adds a
    gentle gradient so the picker prefers short prompts even when
    slack is comfortable. Over many picks this shifts the
    expected GEN time downward, leaving more headroom for HTTP.

25. **Long-tail rollout penalty.** Persisted ``max_lens[prompt_idx]``
    tracks the longest completion length ever observed per prompt (same
    JSON file as ``min_lens``). If ``max_lens >= _LONG_COMPLETION_THRESHOLD_TOKENS``
    (default 5600), the picker multiplies score by
    ``_LONG_COMPLETION_PENALTY`` so prompts prone to runway generations
    are down-ranked without touching the env prompt prefix (prompt-binding
    safe).

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

   The picker is still best-of-K Thompson; ``K`` shrinks as
   ``slots_filled`` rises toward ``B_BATCH`` (race mode, then near-cap
   minimal draws) for faster pick and TCP arrival.

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
    CHALLENGE_K,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    SIGMA_MIN,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
    WINDOW_LENGTH,
)

# Expected maximum OPEN-phase duration in seconds. The protocol minimum
# is ``WINDOW_LENGTH * BLOCK_TIME_SECONDS = 5 * 12 = 60s``, but the
# validator's window stays OPEN until B_BATCH=8 distinct valid
# submissions land (or ``WINDOW_TIMEOUT_SECONDS=7200`` fires). In
# practice we observe windows of 1.5-3.5 minutes (slot fill is paced
# by miner arrival rate, not by chain blocks). 240s is the empirical
# upper end and is what the deadline-penalty picker uses to decide
# whether a slow-completion candidate can fit before the seal.
#
# Kept as a module constant rather than per-window dynamic estimate
# because (a) we don't have a clean signal for "seconds until seal"
# from /state, and (b) over-estimating is safer than under-estimating
# — an over-estimate means we MIGHT pick a slow prompt that misses
# the deadline; an under-estimate would force the picker to skip
# every candidate, starving the miner.
_OPEN_PHASE_BUDGET_S: int = 240
# Reference for the protocol minimum, kept for the PICK log line so
# the operator can see the "ceiling" the chain enforces independently
# of whatever the batcher's actual fill rate is.
_OPEN_PHASE_MIN_S: int = WINDOW_LENGTH * BLOCK_TIME_SECONDS  # = 60

# Observed-OPEN-duration tracker (v4.2 post-mortem fix).
# ``_OPEN_PHASE_BUDGET_S = 240`` is the optimistic upper bound; real
# validators frequently roll OPEN → TRAINING much earlier (e.g. 150 s
# in production logs). When the picker thinks it has 200 s of slack
# but actually has 110 s, it confidently picks long-completion
# prompts that miss the window every time. The fix: at every window
# roll, record ``now - _window_open_seen_at`` of the previous window
# into a rolling deque. The effective budget used by the picker is
# then ``min(_OPEN_PHASE_BUDGET_S, observed_min × _SAFETY_FACTOR)`` —
# realistic on the way down (we learn the validator's true OPEN
# duration), generous on the way up (we never trust beyond the
# configured ceiling).
_OBSERVED_OPEN_HISTORY: int = 20  # window count for the rolling min
_OBSERVED_OPEN_SAFETY_FACTOR: float = 0.85  # 15 % margin under the worst observed
_OBSERVED_OPEN_MIN_FLOOR_S: float = 75.0  # never go below this — would starve picker

# Tight per-request timeout for /state polls (v4.2 fix). The /state
# response body is < 1 KB and the validator's handler is a synchronous
# attribute read on its scheduler — there's no legitimate reason for a
# /state GET to take more than a couple of seconds. The httpx client's
# baseline ``read=15 s`` is generous so /submit body uploads can finish,
# but a 15-s-on-/state means one wedged poll during a window rollover
# burns half the OPEN window. 5 s here + one quick retry caps any
# single /state probe at ~10.5 s wall-clock even when the validator
# is at its slowest. The pre-/post-proof state checks pass this
# explicitly to override the client default.
_STATE_PROBE_TIMEOUT_S: float = 5.0
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

# Best-of-K candidates in the Thompson picker. Default 64 (was 32);
# bumped after W=954-W=964 log analysis showed the random draw was
# missing the rare-but-balanced cohort (L5/Precalculus) most windows.
# At K=64 vs an env with 35 cohorts, the chance of hitting any given
# cohort at least once is ≈ 1 - (34/35)^64 ≈ 84% — vs 60% at K=32.
_CANDIDATES_DEFAULT = 64

# Second-chance candidate pool size (v4.2). When the initial K=64 draw
# returns nothing better than ``_SECOND_CHANCE_SCORE_THRESHOLD``,
# expand the candidate pool to this size. Helps in the strong-model
# regime where most MATH cohorts saturate at k=8 and a random K=64
# draw is mostly saturated — a wider draw gives the picker more
# chances to hit a balanced cohort. Bumped from 128 → 256 (W=954-964
# log analysis): with 35 cohorts, K=256 hits any given cohort at
# ≈ 99.9% rate, virtually guaranteeing the picker sees the best
# non-saturated cohort if one exists. Still completes in single-digit
# ms — each evaluation is ~3 µs of cached cohort lookup + arithmetic.
_CANDIDATES_SECOND_CHANCE = 256
# A score below this is "weak". Raised from 0.05 → 0.15 after the
# W=954-964 log analysis: scores of 0.04-0.10 are still "the best in
# a saturated random sample, but not actually good" — they correspond
# to a marginal pick that will most likely produce k=0/8 or k=8/8.
# At 0.15 we trigger expansion whenever the initial K=64 didn't
# surface a clearly-balanced cohort × prompt combination. The extra
# CPU cost of one in three picks expanding is well under 1 ms.
_SECOND_CHANCE_SCORE_THRESHOLD: float = 0.15

# Race mode: when ``valid_submissions`` from /state is rising toward
# ``B_BATCH``, shrink Thompson draws so pick latency stays low — TCP
# arrival order matters for SUPERSEDED and the worker queue drains
# faster before ``active_batcher`` swaps.
_CANDIDATES_RACE = 16
_RACE_MODE_SLOTS_THRESHOLD = max(2, B_BATCH - 4)
# Near-cap: only ~2 distinct slots likely remain before seal — minimize
# CPU spent in pick_prompt_idx (microseconds matter vs GRAIL backlog).
_CANDIDATES_NEAR_FULL = 8
_NEAR_FULL_SLOTS_THRESHOLD = max(1, B_BATCH - 2)

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

# ───────────── Adaptive over-generation (cherry-pick) ─────────────
# The protocol mandates ``M=8`` rollouts per submission but does NOT
# mandate they be the FIRST 8 sampled. Each per-rollout validator
# check (GRAIL, signature, logprobs, distribution, termination) is
# computed against that single rollout's tokens — none of them
# inspect WHICH 8 of N draws we chose. So when the initial M=8 group
# is out-of-zone, we honestly draw ``_OVERGEN_EXTRA`` more rollouts
# at the protocol temperature and assemble an in-zone subset of 8
# from the combined pool (preferring rollouts with shorter completion
# length so the proof + payload + HTTP cost stays minimal).
#
# Per-rollout statistics (per-token chosen-prob distribution, median
# importance-sampling deviation, GRAIL sketches) are unchanged by
# subset selection because each rollout is an honest sample. Group
# sigma is the only statistic that subset-selection biases, and that's
# the gate we're trying to clear by definition.
#
# Cost: ``_OVERGEN_EXTRA / M_ROLLOUTS ≈ 50%`` extra generation on the
# ~40% of prompts whose first attempt was OOZ → ~+20% steady-state
# generation cost for a ~+30 pp lift in submission in-zone rate on
# marginal prompts. Set ``_OVERGEN_EXTRA = 0`` to disable.
_OVERGEN_ENABLED: bool = True
_OVERGEN_EXTRA: int = 4

# Gate over-generation by remaining OPEN budget. If we're already
# this many seconds into the OPEN phase, the extra rollouts will land
# the submission past the window edge → guaranteed window_mismatch
# even if we recover in-zone. Empirically observed window OPEN of
# 1.5-3.5 min → 120 s lets over-gen run in the first half of typical
# OPEN windows.
_OVERGEN_MAX_OPEN_AGE_S: float = 120.0

# Skip over-gen when the initial group's k_solved is at the saturated
# ends of the spectrum. Probability of recovery is ~18% at k=8 (model
# essentially solves at p>=0.95) and symmetric at k=0 — paying 15-30s
# of extra generation for that recovery rate is net-negative when
# k ∈ [1, 7] candidates exist with ~40% recovery. Tune wider (e.g.,
# 0 / 8 → True) only if logs show the model is genuinely on the edge
# at saturation (p ~ 0.9, not p ~ 1.0).
_OVERGEN_MIN_K: int = 1  # over-gen only if k_solved >= this
_OVERGEN_MAX_K: int = M_ROLLOUTS - 1  # ... and k_solved <= this (= 7)

# ───────────── Short-completion blacklist / penalty ─────────────
# Prompts that have ever produced a rollout with ``completion_length
# < CHALLENGE_K=32`` will trigger the validator's LOGPROB_MISMATCH
# pipeline silently if they reach /submit (the response is the
# provisional SUBMITTED sentinel — the real reject only surfaces in
# validator logs / R2 archive). Multiply the picker score by this
# factor so the prompt is heavily down-ranked but not permanently
# banned (a single one-off short gen could be a model fluke).
_SHORT_COMPLETION_PENALTY: float = 0.10

# Long-tail completion penalty (top-miner rollout alignment). Prompts whose
# rollouts have ever reached this many completion tokens correlate with groups
# that hit max-length runs (~8192) and churn GPU without reliable in-zone
# groups. Multiply picker score when ``max_seen >= threshold`` — milder than
# ``_SHORT_COMPLETION_PENALTY`` because long outputs are not a protocol fault.
_LONG_COMPLETION_THRESHOLD_TOKENS: int = 5600
_LONG_COMPLETION_PENALTY: float = 0.42

# ───────────── Time-budget-aware picker penalties ─────────────
# Picker estimates ``expected_pipeline_s = open_age + gen + proof +
# http`` and penalizes prompts whose expected_pipeline crosses the
# OPEN-phase deadline. ``_TOKENS_PER_SEC_EST`` converts a prompt's
# rolling ``avg_completion_len`` into a generation-time prediction;
# 200 tok/s is conservative for Qwen3-4B-Instruct on H200 at vLLM's
# default settings. _DEADLINE_PROOF_DEFAULT_S / _DEADLINE_HTTP_DEFAULT_S
# are used until the rolling averages have at least one sample.
_TOKENS_PER_SEC_EST: float = 200.0
_DEADLINE_PROOF_DEFAULT_S: float = 3.0
_DEADLINE_HTTP_DEFAULT_S: float = 5.0
_DEADLINE_HARD_PENALTY: float = 0.10  # very risky → score *= 0.10
_DEADLINE_SOFT_PENALTY: float = 0.50  # risky → score *= 0.50

# Slack thresholds (v4.2). ``slack_s = budget - expected_finish_s``;
# hard penalty when slack drops below 5 s (no buffer left for jitter
# or a longer-than-avg gen), soft penalty when below 20 s (risky but
# tolerable). Tightened from the previous "< -5 / < 0" because the
# old thresholds only flagged prompts EXPECTED to bust the budget,
# leaving zero headroom for the inevitable variance in gen time and
# /state-probe latency around window rollover — see the post-mortem
# of W=954 in the project history.
_DEADLINE_HARD_SLACK_S: float = 5.0
_DEADLINE_SOFT_SLACK_S: float = 20.0

# Per-window OOZ-cohort penalty (v4.2 picker realism).
# If a cohort (level, subject) already produced an OOZ skip during
# the current window, every subsequent prompt from that cohort gets
# its score multiplied by ``_OOZ_COHORT_PENALTY``. This addresses
# the W=957 pattern where the picker keeps drawing from the SAME
# saturated cohorts (L3/Prealgebra, L4/Prealgebra, L4/Counting...)
# burning multiple GEN cycles on guaranteed-k=8 prompts before the
# window closes. The penalty is multiplicative (not zero) so that
# in a window where ALL cohorts are saturated we still pick
# something rather than starving the loop. 0.20 ≈ 5× preference
# for a fresh cohort over a known-saturated one at equal raw
# scores.
_OOZ_COHORT_PENALTY: float = 0.20
# Reference completion length used to scale ``http_avg_s`` per-prompt.
# A candidate with avg_len = _TYPICAL_AVG_LEN_FOR_UPLOAD gets the
# unscaled http_avg estimate; longer prompts pay proportionally more.
# 4096 chosen as the rough mid-point of observed completion lengths
# on Qwen3-4B-Instruct MATH rollouts (range typically 800-8000 tok).
_TYPICAL_AVG_LEN_FOR_UPLOAD: float = 4096.0

# When http_avg climbs past this threshold the uplink is the dominant
# bottleneck; skip overgen / SHRT rescue rather than burning more
# generation time we can't afford to ship. The dynamic _chry_deadline
# already does this implicitly, but a hard threshold lets us log a
# clear "high_http_avg" reason instead of a generic deadline skip,
# and avoids a corner case where the rolling proof_avg drops fast
# enough to let _chry_deadline pass while http stays slow.
_HIGH_HTTP_AVG_S: float = 45.0


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
# Engine-side optional belt-and-suspenders. When bittensor/vllm clobber
# the root logger AFTER engine.py is imported, ``logger.info`` can go
# nowhere. Set ``RELIQUARY_RAW_STDERR=1`` to also mirror each ``_emit``
# line to the kernel stderr fd captured at import time.

_STDERR_FALLBACK_ENABLED = os.environ.get(
    "RELIQUARY_RAW_STDERR", "",
).strip().lower() in ("1", "true", "yes")

# Bulletproof direct-write fd, captured at MODULE IMPORT time — i.e.
# before bittensor / vllm / transformers had any chance to wrap or
# replace ``sys.stderr`` / ``sys.__stderr__``. ``os.dup(2)`` returns a
# NEW file descriptor pointing at the same underlying file as fd 2.
# After third-party libs do ``dup2(pipe_write, 2)`` to redirect stderr
# (which is what e.g. bittensor's btlogging does when it captures
# stderr to re-emit through its own logger), our captured fd STILL
# points at the original kernel-level stderr — so ``os.write(_FD2,
# ...)`` cannot be intercepted, recursed, or replayed by any Python
# layer. This is the only Python-level technique that survives
# arbitrary userspace stream wrapping. Raw duplicate is opt-in via
# ``RELIQUARY_RAW_STDERR=1`` (see ``_STDERR_FALLBACK_ENABLED``).
try:
    _RAW_STDERR_FD: int | None = os.dup(2)
except OSError:
    _RAW_STDERR_FD = None

# Pinned reliquary loggers that we keep handler-free (they propagate
# to root). Defended below by ``_resanity_pinned_handlers``.
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
# How often (in _emit calls) to re-strip handlers off named loggers.
# bittensor/vllm have shown a pattern of re-attaching handlers AFTER
# the one-time reseat fires (e.g. on each metagraph fetch, each
# checkpoint reload, each EngineCoreProc respawn), producing the
# "logs duplicate intermittently" symptom. _emit is hit ~5-10 times
# per generation (PICK + GEN + OOZ/SUB ± SUMMARY), so 10 means we
# self-heal within 1-2 generations of any new pollution.
_HANDLER_SANITY_EVERY = 10
_handler_sanity_counter = 0

# Deduplication ring: drop (logger_name, msg) tuples emitted within
# ``_DEDUPE_WINDOW_S`` of an identical prior emission. This is a
# defensive backstop in case BOTH the handler-strip AND the raw-fd
# write paths fail to prevent duplication (e.g. an upstream lib taps
# stderr at the C level via dup2 BEFORE we capture our own dup).
_DEDUPE_WINDOW_S = 0.5
_dedupe_last: dict[tuple[str, str], float] = {}


def _resanity_pinned_handlers() -> int:
    """Strip any newly-attached handlers off pinned named loggers.

    Also re-asserts ``propagate=True`` and ``disabled=False`` because
    some libs flip those when they install themselves.

    Cheap: most loggers have zero handlers most of the time, so this
    iterates a tuple of ~10 strings and does ``lg.handlers``-length
    check per call. Total cost ~5 µs per call on a typical box.

    Returns the number of handlers stripped this pass (0 in steady
    state).
    """
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
    """Write a line directly to the kernel-level stderr fd, bypassing
    every Python-side stream wrap (sys.stderr, sys.__stderr__, any tee
    installed by btlogging, etc.).

    Falls back to ``sys.__stderr__`` if the fd dup at import time
    failed (e.g. running in an embedded interpreter with no stderr).
    Always appends a newline if missing.
    """
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

    Use for operator-critical events: per-attempt PICK/GEN/SUB/OOZ/
    SKIP/ERR, window banners, periodic SUMMARY. Routine debug/trace
    logs continue using plain ``logger.info`` / ``logger.debug``.

    Three layers of defense against the duplicate-log pathology:

    1. **Periodic handler strip**: every ``_HANDLER_SANITY_EVERY``
       calls, scan pinned loggers and strip any handlers that got
       re-attached by third-party libs (bittensor/vllm) after the
       one-time reseat.
    2. **Raw-fd direct write** (opt-in ``RELIQUARY_RAW_STDERR=1``):
       bypass Python-level stream wrapping entirely.
    3. **Dedupe filter**: drop a (logger, msg) pair that's identical
       to one emitted in the last 500 ms. Catches the rare race where
       layers 1+2 both fail.
    """
    if args:
        try:
            msg = fmt % args
        except Exception:
            msg = f"{fmt} args={args!r}"
    else:
        msg = fmt

    # Layer 1: periodic handler sanity check.
    global _handler_sanity_counter
    _handler_sanity_counter += 1
    if _handler_sanity_counter % _HANDLER_SANITY_EVERY == 0:
        n = _resanity_pinned_handlers()
        if n:
            # Use _raw_stderr_write directly here — don't recurse
            # through _emit. Use the logger too so file handlers see
            # it.
            note = (
                f"[engine.v4] handler sanity check: re-stripped {n} "
                f"unauthorized handlers from pinned loggers"
            )
            logger.info(note)
            if _STDERR_FALLBACK_ENABLED:
                _raw_stderr_write(f"[engine] {note}")

    # Layer 3: dedupe by (logger_name, msg) within 500ms.
    now = time.monotonic()
    dedupe_key = (logger.name, msg)
    last_t = _dedupe_last.get(dedupe_key)
    if last_t is not None and (now - last_t) < _DEDUPE_WINDOW_S:
        # Identical message emitted very recently — skip silently. This
        # is the backstop for the duplicate-log pathology. In normal
        # operation no two operational messages are identical within
        # 500 ms, so this is safe.
        _dedupe_last[dedupe_key] = now
        return
    _dedupe_last[dedupe_key] = now
    # Periodically GC the dedupe map so it doesn't grow unbounded.
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


def _find_in_zone_subset(
    pool_rewards: list[float],
    pool_lens: list[int],
    target_size: int = M_ROLLOUTS,
) -> list[int] | None:
    """Cherry-pick ``target_size`` indices from the pool so the chosen
    subset's reward stddev clears the in-zone gate.

    Algorithm (binary-reward Bernoulli case — MATH env):

    1. Partition pool by reward into solves (``≥ _SOLVED_THRESHOLD``)
       and fails.
    2. The target k_solved range is ``[_K_LO, _K_HI]`` (= [2, 6] for
       M=8 at SIGMA_MIN=0.43). Inside that range, k closer to
       ``target_size // 2 = 4`` carries maximum sigma — try those first.
    3. Within each {solves, fails} bucket, prefer rollouts with the
       shortest ``completion_length``. Shorter completions → smaller
       GRAIL proof payload → smaller /submit body → faster HTTP →
       more headroom before the OPEN-window edge.

    For continuous-reward envs this still works correctly because the
    function only consumes ``_SOLVED_THRESHOLD``-thresholded counts to
    decide subset composition; the actual rewards in the chosen subset
    feed the validator's verbatim ``rewards_std`` check.

    Returns ``None`` if no in-zone subset of ``target_size`` exists.
    """
    n = len(pool_rewards)
    if n < target_size or len(pool_lens) != n:
        return None

    solves_idx = [i for i in range(n) if pool_rewards[i] >= _SOLVED_THRESHOLD]
    fails_idx = [i for i in range(n) if pool_rewards[i] < _SOLVED_THRESHOLD]

    solves_idx.sort(key=lambda i: pool_lens[i])
    fails_idx.sort(key=lambda i: pool_lens[i])

    mid_k = target_size // 2
    target_ks = sorted(
        range(_K_LO, min(_K_HI, target_size) + 1),
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

    Reports the picker's quality (in_zone_rate), submit outcomes,
    dominant reject reason, cohort warmup, and the batched-proof health
    channel (if it had to fall back, the operator needs to see it
    immediately).

    Production validators enqueue ``/submit`` and return ``reason=submitted``
    (queued for async GRAIL). That increments ``queued_provisional``, not
    ``validated_accepted``. Only ``reason=accepted`` reflects the inline
    verification path (tests / sync servers).
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
    # Rolling HTTP latency window (last N submit_batch_v2 calls).
    # Used to surface "validator is slow" in the SUMMARY line so the
    # operator immediately knows when their window_mismatch /
    # window_not_active rejection storm is caused by validator
    # latency rather than by their own miner code.
    http_ms_recent: list[float] = field(default_factory=list)
    _http_ms_window: int = 20
    # Rolling proof-construction time window (last N successful
    # _build_rollout_submissions calls). Fed into the picker's
    # deadline estimator alongside ``http_ms_recent`` so the score
    # multiplier reflects the actual pipeline cost rather than a
    # hard-coded guess.
    proof_ms_recent: list[float] = field(default_factory=list)
    _proof_ms_window: int = 20
    # Adaptive over-generation counters.
    #   overgen_attempts  — initial M=8 returned OOZ and we tried to
    #                       cherry-pick from an expanded pool.
    #   overgen_recoveries — cherry-pick found an in-zone subset.
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
    """Frozen snapshot of a generation's metadata for deferred SUB logging.

    Created just before ``asyncio.create_task(_fire_submit(...))`` and
    consumed by ``_drain_submit`` at the top of the next loop iteration
    when the HTTP task is done.
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

    Maps below (plus cohort in-zone rate) are persisted atomically to JSON:

      _counts[idx]              = (solves, attempts)              per-prompt counts
      _last_checkpoint[idx]     = int                              checkpoint these counts grew on
      _cohort_counts[(L, S)]    = (solves, attempts)              MATH (level, subject) cell
      _cohort_last_checkpoint[(L, S)] = int                        cohort checkpoint tag
      _cohort_inzone[(L, S)]    = (in-zone groups, total groups)
      _lengths[idx]             = (mean_len, n_obs)                completion-length tiebreak
      _min_lens[idx]           = int                               min observed completion length
      _max_lens[idx]           = int                               max observed completion length
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
        # Per-cohort empirical in-zone rate: (inzone_count, total_groups).
        # Tracks how often a cohort produces a group with k ∈ [K_LO, K_HI]
        # (the zone gate). Used by the picker to multiply zone_p by the
        # cohort's empirical hit rate, discounting cohorts that consistently
        # return k=0 or k=8 even when the per-prompt phat looks marginal.
        "_cohort_inzone",
        "_lengths",
        "_min_lens",
        "_max_lens",
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
        self._cohort_inzone: dict[tuple[str, str], tuple[int, int]] = {}
        self._lengths: dict[int, tuple[float, int]] = {}
        # Minimum observed completion length per prompt. Used by the
        # picker to down-rank prompts that have ever produced a rollout
        # < CHALLENGE_K (which the validator silently rejects with
        # LOGPROB_MISMATCH if it reaches /submit). Stored separately
        # from ``_lengths`` (which tracks the mean for tie-break) so a
        # one-off short gen leaves a permanent dent.
        self._min_lens: dict[int, int] = {}
        # Maximum observed completion length per prompt. Picker penalizes
        # prompts whose rollouts ever hit the long-tail band (cheap GPU churn).
        self._max_lens: dict[int, int] = {}
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

    def min_completion_len(self, idx: int) -> int | None:
        """Minimum observed completion length for ``idx``, or None if
        never observed. Picker uses this to detect prompts that have
        produced rollouts shorter than the validator's CHALLENGE_K gate.
        """
        return self._min_lens.get(idx)

    def has_short_completion(self, idx: int, threshold: int) -> bool:
        """True if any observed rollout for ``idx`` was shorter than
        ``threshold``. Picker uses this to apply the short-completion
        penalty multiplier (``_SHORT_COMPLETION_PENALTY``).
        """
        m = self._min_lens.get(idx)
        return m is not None and m < threshold

    def max_completion_len(self, idx: int) -> int | None:
        """Longest observed completion length for ``idx``, or None if never measured."""
        return self._max_lens.get(idx)

    def has_long_completion(self, idx: int, threshold: int) -> bool:
        """True if any observed rollout for ``idx`` reached at least ``threshold`` tokens."""
        m = self._max_lens.get(idx)
        return m is not None and m >= threshold

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

        # Cohort in-zone empirical rate. Track whether the submitted
        # *group* was in-zone. Each call to record_group represents one
        # generation attempt with M=8 rollouts; we record 1 group.
        _, _, group_inzone = _zone_status(rewards)
        ciz, ctot = self._cohort_inzone.get(key, (0, 0))
        self._cohort_inzone[key] = (
            ciz + (1 if group_inzone else 0),
            ctot + 1,
        )

        # Cohort cache for picker hot path
        self._cohort_cache[prompt_idx] = (str(level), str(subject))

        # Lengths
        if completion_lens:
            prev_mean, prev_n = self._lengths.get(prompt_idx, (0.0, 0))
            total_n = prev_n + len(completion_lens)
            new_mean = (prev_mean * prev_n + sum(completion_lens)) / total_n
            self._lengths[prompt_idx] = (new_mean, total_n)
            # Update minimum observed completion length. This is the
            # signal the picker uses to penalize prompts whose rollouts
            # have ever fallen below the validator's CHALLENGE_K=32 gate.
            cur_min = min(completion_lens)
            stored_min = self._min_lens.get(prompt_idx)
            if stored_min is None or cur_min < stored_min:
                self._min_lens[prompt_idx] = cur_min
            cur_max = max(completion_lens)
            stored_max = self._max_lens.get(prompt_idx)
            if stored_max is None or cur_max > stored_max:
                self._max_lens[prompt_idx] = cur_max

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

    def cohort_inzone_rate(
        self,
        level: str,
        subject: str,
        min_groups: int = 2,
    ) -> float | None:
        """Empirical in-zone rate for the (level, subject) cohort.

        Returns the fraction of groups from this cohort that landed in
        zone (k ∈ [K_LO, K_HI]). Returns ``None`` when fewer than
        ``min_groups`` groups have been attempted (not enough signal to
        trust). The picker multiplies ``zone_p`` by this value to
        discount cohorts that consistently produce saturated k=0/8 even
        when the per-prompt phat looks marginal.

        ``min_groups`` dropped from 5 → 2 (W=954-964 log analysis): the
        Laplace smoothing already absorbs small-sample noise, and the
        early picks of a session would otherwise see ciz_rate=None on
        most cohorts (in the trace, only ~28 groups had been recorded
        across 35 cohorts ≈ 0.8 obs/cohort, so the picker had zero
        cohort-level discrimination until the third or fourth window).
        """
        key = self._cohort_key(level, subject)
        iz, tot = self._cohort_inzone.get(key, (0, 0))
        if tot < min_groups:
            return None
        return (iz + 0.5) / (tot + 1.0)  # Laplace-smoothed

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
        # ``min_lens`` / ``max_lens`` are optional: older stats files omit
        # them; rebuild as new observations land. Additive — no schema bump.
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
    # Deadline-awareness context. The picker estimates each candidate's
    # expected pipeline time (open_age + gen + proof + http) and
    # penalizes prompts whose expected completion crosses the OPEN
    # window edge. Defaults preserve the legacy behaviour (no penalty).
    open_age_s: float = 0.0,
    open_budget_s: float = float("inf"),
    proof_avg_s: float = _DEADLINE_PROOF_DEFAULT_S,
    http_avg_s: float = _DEADLINE_HTTP_DEFAULT_S,
    # Short-completion gate. Prompts whose minimum-observed
    # completion length is below this threshold get the
    # ``_SHORT_COMPLETION_PENALTY`` multiplier applied to their score.
    # CHALLENGE_K=32 is the validator's logprob-claim minimum.
    short_completion_threshold: int = CHALLENGE_K,
    # Cohorts already proven OOZ-saturated in the current window;
    # prompts from these cohorts get _OOZ_COHORT_PENALTY applied
    # to their score so a single bad window doesn't keep generating
    # k=8 rollouts from the same cohort.
    ooz_cohorts_in_window: set[tuple[str, str]] | None = None,
) -> int:
    """Best-of-K Thompson-sampled prompt selection.

    Scoring per candidate ``idx``::

        a, b      = stats.posterior(idx, ckpt_n, level, subject)
                    # cohort prior used when prompt is unseen
                    # or counts are from a stale checkpoint
        p_sample  = rng.betavariate(a, b)
        zone_p    = P(K_LO ≤ Binomial(M, p_sample) ≤ K_HI)
        arrival_z = stats.arrival_rate(idx)       ∈ [0, 1]
        base      = zone_p * (1 - _CONGESTION_WEIGHT * arrival_z)
        short_mul = _SHORT_COMPLETION_PENALTY if stats has ever
                    seen ``idx`` produce a < CHALLENGE_K rollout
                    else 1.0
        long_mul  = _LONG_COMPLETION_PENALTY if stats has ever
                    seen ``idx`` produce a rollout >= threshold
                    (``_LONG_COMPLETION_THRESHOLD_TOKENS``)
                    else 1.0
        ddl_mul   = _DEADLINE_HARD_PENALTY  if expected pipeline
                    finishes > 5 s past OPEN edge
                    _DEADLINE_SOFT_PENALTY if 0-5 s past edge
                    1.0 otherwise
        score     = base * short_mul * long_mul * ddl_mul

    Tiebreak: shorter average completion length (faster generation →
    earlier TCP arrival → more SUPERSEDED wins).

    Near-cap mode (slots_filled ≥ _NEAR_FULL_SLOTS_THRESHOLD): ``K`` is
    minimal (``_CANDIDATES_NEAR_FULL``) — tail race before ``B_BATCH`` seal.

    Race mode (slots_filled ≥ _RACE_MODE_SLOTS_THRESHOLD but below
    near-cap): ``K`` is ``_CANDIDATES_RACE`` instead of the default pool.

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
        """Evaluate one candidate; returns ``(score, avg_len)`` or None
        if the candidate is unevaluable (e.g. already seen). Updates
        ``seen`` as a side effect.
        """
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
        zone_p = _zone_probability(p_sampled)
        arrival_z = stats.arrival_rate(idx)
        score = zone_p * (1.0 - _CONGESTION_WEIGHT * arrival_z)

        # Cohort empirical in-zone rate multiplier. Squared so the
        # picker discriminates ~12-25x between a balanced cohort (ciz
        # ≈ 0.7) and a saturated one (ciz ≈ 0.2) instead of only 3.5x
        # under the linear version. The W=954-964 trace showed the
        # linear penalty was insufficient: saturated cohorts (L4/Algebra,
        # L3/Counting, L4/Geometry) routinely outscored balanced cohorts
        # (L5/Precalculus) because their phat × zone_p baseline was
        # high enough that even a 3.5x ciz penalty left them competitive.
        # Squaring brings the saturated cohorts' effective score down by
        # the same factor again so the picker reliably prefers a
        # known-balanced cohort once 2+ observations exist.
        ciz_rate = stats.cohort_inzone_rate(level, subject)
        if ciz_rate is not None:
            score *= ciz_rate * ciz_rate

        # Per-window OOZ-cohort penalty.
        if (
            ooz_cohorts_in_window is not None
            and (level, subject) in ooz_cohorts_in_window
        ):
            score *= _OOZ_COHORT_PENALTY

        # Short-completion penalty.
        if stats.has_short_completion(idx, short_completion_threshold):
            score *= _SHORT_COMPLETION_PENALTY

        # Long-tail completion penalty (ever saw a very long rollout).
        if stats.has_long_completion(idx, _LONG_COMPLETION_THRESHOLD_TOKENS):
            score *= _LONG_COMPLETION_PENALTY

        # Time-budget penalty with length-scaled http estimate.
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

        # Continuous length-aware boost (v4.2 W=954-964 fix). The
        # deadline penalty above is binary (kicks in only when slack
        # drops below the threshold); when slack is moderate, the
        # picker is indifferent between a 1 000-token prompt and a
        # 7 000-token prompt that finish in nearly the same projected
        # time. Apply a smooth multiplier so a clearly-short prompt
        # gets a small bonus (≈ 1.10x at avg_len = 500) and a
        # clearly-long one a small penalty (≈ 0.92x at avg_len =
        # 8000). Effect is small per-pick (within ±10 %) but consistent
        # — over many picks the picker drifts toward fast-completing
        # prompts so the OPEN-window race is less prone to bad luck
        # on a single long generation.
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
        """Draw and evaluate up to ``num`` more candidates."""
        for _ in range(num):
            idx = _draw_one()
            if idx is None:
                return
            scored = _score_one(idx)
            if scored is None:
                continue
            score, alen = scored
            _update_best(idx, score, alen)

    # Initial draw.
    _scan(K)

    # Second-chance expansion. When the best score after the initial
    # K draws is weak (typical when the model is so strong that most
    # cohorts saturate at k=8), do another, larger round of draws so
    # the picker has a meaningfully better chance of finding a
    # balanced-cohort prompt. Costs a few hundred microseconds of
    # extra CPU work — negligible against the ~10-40 s GEN that
    # follows. Skipped in race mode (slots_filled tall already, the
    # priority is speed not optimality).
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
        # Multi-validator broadcast (v4.2). When ``validator_urls_override``
        # is None, the engine discovers up to ``max_validators`` permitted
        # validators from the metagraph and broadcasts every /submit to
        # all of them in parallel. This multiplies the effective EMA
        # weight contribution per submission by ~N (each validator scores
        # us independently). Backward compatibility: if only
        # ``validator_url_override`` is given, we treat it as a 1-element
        # list.
        if validator_urls_override:
            self.validator_urls_override: list[str] | None = list(validator_urls_override)
        elif validator_url_override:
            self.validator_urls_override = [validator_url_override]
        else:
            self.validator_urls_override = None
        self.max_validators = max_validators
        # Per-request HTTP timeout. Lowered from v3's 60 s to 30 s by
        # default so a single slow validator doesn't cost us the whole
        # OPEN window. The validator's /submit endpoint enqueues and
        # returns immediately once the body upload completes — so 30 s
        # is plenty for the upload itself on any reasonable uplink.
        self.http_timeout_s = http_timeout_s
        # Resolved per-mine-window list of validator URLs. Populated by
        # ``_resolve_validator_urls`` at the top of ``mine_window``;
        # re-discovered on every metagraph fetch.
        self._validator_urls: list[str] = []

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

        # Wall-clock at which we first observed the current window in
        # OPEN state. Used to surface ``open_age=N.Ns`` in PICK and SUB
        # logs so the operator can see whether submissions are racing
        # the ~60s OPEN-phase window edge.
        self._window_open_seen_at: float | None = None

        # Rolling history of observed OPEN-phase durations per window
        # (in seconds, computed at each window-roll boundary as
        # ``now - _window_open_seen_at``). Used by
        # ``_effective_open_budget_s`` to give the picker a realistic
        # budget rather than the optimistic ``_OPEN_PHASE_BUDGET_S``
        # constant.
        #
        # WHY: ``_OPEN_PHASE_BUDGET_S = 240`` is a generous upper bound
        # on the protocol's OPEN duration. Real validators commonly
        # roll OPEN→TRAINING much sooner (e.g. ~150 s in production
        # logs), so the picker's deadline math thought it had ~200 s
        # of headroom when it actually had ~110 s — picking a long-
        # completion prompt that misses the window every time. The
        # observed-min tracker below lets the engine learn the actual
        # OPEN duration this validator delivers and clamp the budget
        # to it (with a safety factor).
        self._observed_open_durations_s: collections.deque[float] = (
            collections.deque(maxlen=_OBSERVED_OPEN_HISTORY)
        )

        # Per-window OOZ-cohort blacklist (v4.2). Cohorts whose
        # current-window pick already produced an OOZ skip; the
        # picker uses this to apply ``_OOZ_COHORT_PENALTY`` to any
        # future candidate from the same cohort during the window.
        # Reset to ``set()`` at every window-roll boundary in the
        # main poll loop.
        self._ooz_cohorts_in_window: set[tuple[str, str]] = set()

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
        # Multi-validator broadcast support (v4.2). The submitter module
        # may have been overlaid with submitter-v4 which exposes the
        # plural ``discover_validator_urls`` and ``submit_batch_v2_multi``.
        # We import them lazily and fall back to single-validator if
        # the running submitter is upstream-v3 (no plural helpers).
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
                "Overlay submitter-v4.py onto /root/reliquary/reliquary/miner/submitter.py "
                "to enable broadcast and reduce window_mismatch rejections."
            )
        # prewarm_connections is a v4.2 optional helper. Older submitter
        # builds (no prewarm) → fall back to a no-op.
        try:
            from reliquary.miner.submitter import (
                prewarm_connections as _prewarm_connections,
            )
        except ImportError:
            _prewarm_connections = None  # type: ignore[assignment]

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

        # Resolve validator URLs once at startup. The PRIMARY url (the
        # first in the list) drives /state polls; the FULL list is used
        # for /submit broadcast so every validator that's online scores
        # our submission, multiplying our weight EMA contribution.
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

        # v4.2 tightened HTTP timeouts + HTTP/2 keepalive.
        #
        # CRITICAL INSIGHT (server.py /submit): the validator's
        # /submit endpoint enqueues the request on a background worker
        # and returns IMMEDIATELY with ``SUBMITTED`` (the queue is
        # unbounded, and the real GRAIL/logprob/distribution checks
        # run async). So the only meaningful HTTP cost is the upload
        # itself — for a 10-30 MB payload (8 rollouts × up to 8192
        # tokens × per-token GRAIL commitments + logprobs), that's
        # ~5-30 s on a typical box uplink.
        #
        # v3 used read=120s + 3 retries with 1s/2s/4s backoff. That
        # gave a single doomed submit up to 247 s before giving up —
        # the OPEN window has rolled at least three times by then
        # and every retry hits window_mismatch.
        #
        # v4.2: short timeouts + zero submit retries via
        # submitter-v4. Pool the connection (HTTP/2 keep-alive) so
        # back-to-back submits to the same validator skip the TCP
        # handshake.
        _http_to_s = float(self.http_timeout_s)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                # connect: dead-validator check (DNS + TCP). Short so
                # a dropped validator doesn't block our pipeline.
                connect=5.0,
                # read: response wait. The validator's /submit reply
                # is < 1 KB (just BatchSubmissionResponse). 15 s is
                # ample if the body uploaded — anything longer is
                # almost certainly a wedged validator.
                read=max(15.0, _http_to_s * 0.5),
                # write: body upload. The bulk of the latency lives
                # here for any non-LAN setup.
                write=_http_to_s,
                pool=5.0,
            ),
            # HTTP/2 keep-alive halves the per-submit TCP/TLS setup
            # cost when broadcasting to the same set of validators
            # window after window. Falls back to HTTP/1.1 transparently
            # if httpx wasn't built with h2 support (which most
            # default installs are).
            http2=False,  # vLLM-stack envs often lack h2; safer default
            limits=httpx.Limits(
                max_keepalive_connections=max(8, len(self._validator_urls) * 2),
                max_connections=max(16, len(self._validator_urls) * 4),
                keepalive_expiry=120.0,
            ),
        ) as client:
            # Pre-warm TCP/TLS to every discovered validator BEFORE the
            # first OPEN window's submit. The handshake cost
            # (DNS + TCP + TLS) is ~50-300 ms per host on the cold path
            # — paying that during a /submit eats into the OPEN-window
            # budget. Pre-warm here so the connection pool already
            # has live keep-alived sockets by the time PICK → GEN →
            # PROOF → SUBMIT fires the first batch.
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

            # ── Async HTTP submit helpers ───────────────────────────────────
            # The engine used to block on ``await submit_batch_v2(...)``
            # (20-70 s), keeping the GPU idle the whole time.  Now we fire
            # the HTTP call as an ``asyncio.Task`` and immediately loop back
            # to PICK → GEN, so generation overlaps the network round-trip.
            # The task result is drained at the top of the next iteration.
            #
            # Serialisation rule: at most one in-flight submit at a time
            # (the validator only accepts one submission per miner per window
            # slot anyway).  If the previous task is still running when we
            # finish the next GEN+PROOF, we await it before creating a new
            # one — but in practice GEN ≈ 36 s >> http_avg ≈ 20 s so the
            # task is almost always done by then.

            _pending_task: asyncio.Task | None = None
            _pending_ctx: _SubmitCtx | None = None

            async def _fire_submit(req: "BatchSubmissionRequest") -> tuple:
                """Submit to ALL validators in parallel (v4.2 broadcast).

                Returns ``(accepted, best_reason_str, max_http_ms,
                gpu_post, per_url_breakdown)``. Falls back to
                single-validator submission if submitter-v4 isn't
                deployed (``_MULTI_VALIDATOR_AVAILABLE = False``).
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
                    # All-network-error case: if EVERY validator threw a
                    # network exception (no validator actually responded
                    # with a structured BatchSubmissionResponse), this is a
                    # network-error event, not a validator rejection. Track
                    # separately so the SUMMARY net_err counter reflects
                    # real connectivity issues rather than burying them
                    # under the rejection bucket.
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
                    if reason_str == "superseded":
                        # SUPERSEDED is per-validator: another miner won
                        # this prompt at that validator. We blacklist
                        # the prompt for the current window regardless
                        # because retrying at the same validator(s) is
                        # a guaranteed loss.
                        self._superseded_in_window.add(ctx.prompt_idx)
                        self._prompt_stats.record_superseded(ctx.prompt_idx)
                    self._maybe_check_proof_fallback()
                    status = (
                        "QUEUED"
                        if accepted and reason_str == "submitted"
                        else ("ACCEPTED" if accepted else "REJECTED")
                    )
                    # Per-validator breakdown — folds N validators into
                    # a compact ``v=A/B`` (accepted/total) string so a
                    # broadcast SUB log stays single-line.
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
                        "open_age=%.1fs %s accepted=%s status=%s reason=%s",
                        ctx.window_n, ctx.prompt_idx,
                        _fmt_rewards(ctx.rewards),
                        ctx.k_solved, M_ROLLOUTS, ctx.sigma,
                        ctx.merkle_root[:8], ctx.proof_mode,
                        ctx.gen_ms / 1000.0, ctx.proof_ms / 1000.0,
                        http_ms / 1000.0,
                        ctx.gpu_pre, gpu_post, ctx.open_age_s,
                        multi_str, accepted, status, reason_str,
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
                    results.append((accepted, reason_str))
                except asyncio.CancelledError:
                    # Task cancelled by window-roll handler — log
                    # quietly so the operator can correlate window
                    # transitions with cancelled submits.
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
                # Periodic SUMMARY after any submit outcome.
                if (
                    self._metrics.submitted
                    and self._metrics.submitted % _SUMMARY_EVERY == 0
                ):
                    _emit(
                        logging.INFO,
                        "=== SUMMARY === %s",
                        self._metrics.summary(self._prompt_stats),
                    )

            # ───────────────────────────────────────────────────────────────

            while True:
                try:
                    # Use the short state-probe timeout here too: a
                    # stalled main-loop /state would block the engine
                    # from ever entering OPEN-window logic, and the
                    # default 30 s client timeout amplifies any
                    # validator hiccup into a full poll-interval miss.
                    state = await get_window_state_v2(
                        url, client=client,
                        timeout=_STATE_PROBE_TIMEOUT_S,
                    )
                except SubmissionError:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                except Exception as e:
                    logger.debug("state fetch failed: %s", e)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # Drain any completed async submit from the previous
                # iteration.  Done right after state fetch so `state` is
                # fresh before we log / update superseded-in-window.
                if _pending_task is not None and _pending_task.done():
                    await _drain_submit(_pending_task, _pending_ctx)
                    _pending_task = None
                    _pending_ctx = None

                # WINDOW-ROLL CANCELLATION (v4.2). If a submit is still
                # in flight but the window has already changed (or the
                # validator state moved out of OPEN), the response is
                # guaranteed to be ``window_mismatch`` /
                # ``window_not_active``. Cancel the doomed task now so
                # we free the network for the next submit ASAP. The
                # task's ``_fire_submit`` body catches CancelledError
                # via the surrounding ``_drain_submit`` handler — no
                # leak.
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
                    # Drain handles CancelledError gracefully.
                    await _drain_submit(_pending_task, _pending_ctx)
                    _pending_task = None
                    _pending_ctx = None

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

                # When local generation matches the validator gate but we never
                # persisted a revision string (e.g. cold start), mirror remote so
                # BatchSubmissionRequest.checkpoint_hash cannot be empty while
                # the gate expects WRONG_CHECKPOINT checks.
                if (
                    local_n == state.checkpoint_n
                    and not local_hash
                    and state.checkpoint_revision
                ):
                    local_hash = state.checkpoint_revision

                if state.state != WindowState.OPEN:
                    await asyncio.sleep(1)
                    continue

                # Window-edge bookkeeping.
                if last_window_n != state.window_n:
                    # Record the previous window's observed OPEN
                    # duration BEFORE we reset _window_open_seen_at,
                    # so the picker learns the validator's real
                    # delivery rate window-by-window.
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
                    # Track the wall-clock at which we first saw this
                    # window in OPEN state, so we can surface
                    # ``open_age=N.Ns`` to make pipeline-vs-window-edge
                    # races diagnosable. The protocol gives us
                    # ``_OPEN_PHASE_BUDGET_S`` (configured upper-bound)
                    # of OPEN per window, but the validator may roll
                    # earlier — see ``_effective_open_budget_s()``.
                    self._window_open_seen_at = time.monotonic()
                    self._superseded_in_window = set()
                    # Reset the per-window OOZ-cohort blacklist —
                    # cohorts saturated in window N-1 may behave
                    # differently in window N (different prompts get
                    # sampled), so re-evaluate fresh each window.
                    self._ooz_cohorts_in_window = set()
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

                # ``valid_submissions`` tracks accepted rows toward training.
                # Once it reaches ``B_BATCH``, the validator is sealing — any
                # new GEN is competing only for archive semantics / next
                # window; skip pick to save GPU until OPEN rolls.
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
                    await asyncio.sleep(1.0)
                    continue

                # HARD PRE-PICK DEADLINE GATE (v4.2). When the OPEN
                # window has been running long enough that the
                # CHEAPEST possible attempt (just http_avg + proof_avg)
                # would land past the budget, every PICK is doomed.
                # Skip the pick entirely and idle until the window
                # rolls — far better than burning a generate+proof
                # cycle on a submit that will return window_mismatch.
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
                # Floor on gen cost — even a length-cap-1 prompt
                # takes ~5 s of vLLM warmup + sampling. Most prompts
                # take 15-45 s, but we use a fast lower bound here
                # because the picker's per-prompt avg_len gating
                # already filters out slow prompts.
                _hard_min_gen_s = 5.0
                _min_pipeline_s = (
                    _hard_http_s + _hard_proof_s + _hard_min_gen_s
                )
                # Use the OBSERVED-OPEN-aware budget rather than the
                # optimistic constant. After ≥ 3 windows of evidence
                # this clamps the budget to what the validator
                # actually delivered, so the gate skips PICKs that
                # would otherwise burn 30-60 s of GPU on a doomed
                # pipeline. See ``_effective_open_budget_s``.
                _eff_budget_s = self._effective_open_budget_s()
                _hard_deadline_s = _eff_budget_s - _min_pipeline_s
                if _now_age_s > _hard_deadline_s:
                    # Throttle the WAIT log: emit once on first entry per
                    # window (so the operator sees WHY the picker stopped),
                    # then a heartbeat every 30 s. Previously every 2 s
                    # poll logged a near-identical line, drowning the rest
                    # of the engine's per-window output in repetitive
                    # WAIT lines (see W=975-989 traces).
                    _wait_state = getattr(self, "_wait_log_state", None)
                    _wn = state.window_n
                    _emit_wait = False
                    if _wait_state is None or _wait_state[0] != _wn:
                        # First WAIT entry of this window.
                        self._wait_log_state = (_wn, time.monotonic())
                        _emit_wait = True
                    else:
                        # Subsequent WAIT polls — log a 30 s heartbeat only.
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
                    await asyncio.sleep(2.0)
                    continue

                # Pick.
                cooldown_set = set(state.cooldown_prompts)
                try:
                    # Feed the deadline estimator with real rolling
                    # averages where available so the picker's
                    # ``score *= ddl_mul`` reflects current network +
                    # proof cost rather than hard-coded guesses.
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
                open_age_s = (
                    time.monotonic() - self._window_open_seen_at
                    if getattr(self, "_window_open_seen_at", None) is not None
                    else 0.0
                )
                _emit(
                    logging.INFO,
                    "[W=%d] PICK prompt=%-4d cohort=(%s,%s) phat=%.2f "
                    "attempts=%d arr=%.2f zone_p=%.2f slots=%d/%d "
                    "open_age=%.1fs (min=%ds, eff_budget=%.0fs) "
                    "ooz_blk=%d",
                    state.window_n, prompt_idx,
                    _short_level(level), _short_subject(subject),
                    p_hat, attempts, arr, _zone_probability(p_hat),
                    state.valid_submissions, B_BATCH,
                    open_age_s, _OPEN_PHASE_MIN_S, _eff_budget_s,
                    len(self._ooz_cohorts_in_window),
                )
                logger.debug(
                    "[W=%d] PICK detail prompt=%d posterior=Beta(%.2f,%.2f) "
                    "level=%r subject=%r",
                    state.window_n, prompt_idx, a, b, level, subject,
                )

                # Generate.
                #
                # v4.2: dispatch the (blocking) vLLM call to a worker
                # thread via ``asyncio.to_thread`` so the event loop
                # can service in-flight network I/O (the previous
                # submit's body upload + response, /state polls, etc.)
                # while the GPU is producing tokens. Without this, the
                # ~10-30 s vLLM call freezes httpx's reactor, which is
                # the silent cause of the previous submit completing
                # 10-30 s LATER than it actually finished on the wire.
                # vLLM's sync ``LLM.generate`` serializes through its
                # own scheduler so the off-thread dispatch is safe.
                t_gen = time.monotonic()
                generations = await asyncio.to_thread(
                    self._generate_n_rollouts, problem, M_ROLLOUTS,
                )
                gen_ms = (time.monotonic() - t_gen) * 1000.0
                # Opportunistic drain: if the previous submit task
                # completed during gen (likely now that to_thread frees
                # the loop), record its result BEFORE we burn proof
                # cycles. Lets metrics surface accept/reject sooner.
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

                # Adaptive over-generation + cherry-pick.
                # ---------------------------------------------------------
                # If the initial M=8 group is out-of-zone, draw
                # _OVERGEN_EXTRA more honest rollouts at T_PROTO and try
                # to assemble an in-zone subset of M from the combined
                # pool. Each rollout in the pool is a clean sample from
                # the model's protocol-temperature distribution, so the
                # validator's per-rollout statistical checks (GRAIL,
                # logprob median dev, distribution q10, termination)
                # all pass on whichever subset we ship. The protocol
                # nowhere requires us to ship the FIRST M draws.
                #
                # Gated by remaining OPEN budget: skip when we're
                # already past _OVERGEN_MAX_OPEN_AGE_S into the OPEN
                # phase, because the extra gen would push the submit
                # past the window edge and we'd lose to window_mismatch
                # even after recovering in-zone.
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
                    # High-uplink-latency hard gate (v4.2). When recent
                    # /submit traffic has been slow enough that adding
                    # any extra generation will almost certainly miss
                    # the OPEN-window edge, abandon over-gen up front
                    # so we can ship what we have (or fall through to
                    # the local OOZ short-circuit and pick a different
                    # prompt). Skipping here is strictly faster than
                    # letting the dynamic _chry_deadline catch it
                    # because we also avoid the deadline-formula CPU
                    # and the +4 generation queue setup cost.
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
                    # Skip over-gen on saturated groups (k=0 or k=8):
                    # at those extremes the model's per-rollout p is
                    # so close to {0, 1} that +4 more honest draws
                    # have a ~15-20% chance of producing the opposite
                    # outcome we need to assemble an in-zone subset.
                    # Pay for over-gen only when we're near the in-zone
                    # boundary (k ∈ [1, 7]) where the same +4 has a
                    # ~40% recovery rate.
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
                        # Dynamic CHRY deadline: how much of the OPEN
                        # budget remains after http + proof + overgen
                        # cost? This is tighter than the static
                        # _OVERGEN_MAX_OPEN_AGE_S = 120 s cap, which
                        # was leaving too little headroom when
                        # http_avg ≈ 70 s but cutting too aggressively
                        # when http_avg is low.
                        #
                        # Anchored on _effective_open_budget_s() so the
                        # deadline tightens automatically when the
                        # observed-OPEN tracker sees the validator is
                        # rolling early.
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
                                # Estimate overgen time proportional to
                                # the just-completed generation time.
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
                            # Stats update with the extras so the
                            # posterior learns from ALL honest draws,
                            # not just the submitted subset.
                            self._prompt_stats.record_group(
                                prompt_idx, extra_rewards,
                                level=level, subject=subject,
                                checkpoint_n=state.checkpoint_n,
                                completion_lens=extra_lens,
                            )
                            pool_gens = generations + extra_gens
                            pool_rewards = rewards + extra_rewards
                            pool_lens = completion_lens + extra_lens
                            chosen_idxs = _find_in_zone_subset(
                                pool_rewards, pool_lens, M_ROLLOUTS,
                            )
                            overgen_ms = (time.monotonic() - t_overgen) * 1000.0
                            if chosen_idxs is not None:
                                # Repaint the working group from the
                                # cherry-picked subset and re-evaluate.
                                generations = [pool_gens[i] for i in chosen_idxs]
                                rewards = [pool_rewards[i] for i in chosen_idxs]
                                completion_lens = [
                                    pool_lens[i] for i in chosen_idxs
                                ]
                                sigma, k_solved, in_zone = _zone_status(rewards)
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

                # Local OUT_OF_ZONE short-circuit.
                if not in_zone:
                    self._metrics.record_local_oos()
                    # Remember the cohort so the picker will down-rank
                    # any future candidate from it for the rest of
                    # this window. Without this, the picker keeps
                    # drawing from the same saturated cohort and
                    # burning successive GEN cycles on guaranteed
                    # k=8 outcomes (see W=957 trace: 3× picks from
                    # Prealgebra cohorts all OOZ before the window
                    # rolled).
                    self._ooz_cohorts_in_window.add((level, subject))
                    _emit(
                        logging.INFO,
                        "[W=%d] OOZ  prompt=%-4d sigma=%.3f<%.2f k=%d/%d "
                        "cohort=(%s,%s) -> skip submit (cohort blacklisted "
                        "for window)",
                        state.window_n, prompt_idx, sigma, SIGMA_MIN,
                        k_solved, M_ROLLOUTS,
                        _short_level(level), _short_subject(subject),
                    )
                    # Surface SUMMARY periodically even when skipping submit.
                    if self._metrics.generated % _SUMMARY_EVERY == 0:
                        _emit(
                            logging.INFO,
                            "=== SUMMARY === %s",
                            self._metrics.summary(self._prompt_stats),
                        )
                    continue

                # CHALLENGE_K gate. The validator's ``verify_logprobs_claim``
                # requires every non-truncated rollout to have
                # ``completion_length >= CHALLENGE_K=32``; anything shorter
                # silently fails with ``LOGPROB_MISMATCH`` deep in the
                # async worker (the miner never sees that verdict — the
                # /submit response is the provisional SUBMITTED sentinel).
                #
                # A rollout that ends with EOS at length < 32 is the
                # ONLY case this catches: those pass termination Path 2
                # and reach the logprob check. ``finish=length`` rollouts
                # are at completion_length = effective_max_new (always
                # >> 32), so they're never the short one.
                #
                # SHRT rescue: when ≤ 2 rollouts in an otherwise in-zone
                # group are short, generate replacement(s) rather than
                # discarding the whole attempt. The replacement is honest
                # (same protocol temperature), so statistical validity is
                # preserved. Gated by the same dynamic deadline used for
                # CHRY so we don't push the submit past the window edge.
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
                    # Anchored on _effective_open_budget_s() so the
                    # SHRT rescue deadline tightens automatically when
                    # the observed-OPEN tracker sees the validator
                    # rolling early — same fix as CHRY above.
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
                    # High-uplink-latency hard gate (v4.2). Same logic
                    # as the CHRY uplink gate: if the network is too
                    # slow to fit another generation cycle inside the
                    # OPEN window, skip the rescue. The rest of the
                    # short-rollout handling will log a single "SHRT
                    # skip" line below.
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
                            new_sigma, new_k, new_inzone = _zone_status(new_rews)
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

                # Fast pre-proof window guard. Check window state RIGHT
                # NOW — before spending 2-3 s building the proof —
                # so we can abort early when the window has already
                # closed or moved to a non-OPEN phase (TRAINING /
                # PUBLISHING) during generation or CHRY.
                # Checking `state != OPEN` is what prevents the
                # window_not_active rejection: the window_n may still
                # match (same window, different phase), but the
                # validator will refuse the submission.
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
                    # Promoted from DEBUG → INFO so the operator can see
                    # when the validator's /state is throwing during a
                    # rollover — previously these failures were silent
                    # in INFO logs and masked the picker's miss.
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

                # Build GRAIL proofs.
                #
                # v4.2: same off-loop dispatch as gen, for the same
                # reason. Proof is shorter (~3-5 s) but on a slow
                # uplink the previous submit's response may still be
                # arriving — letting the event loop service those
                # bytes during proof keeps the metrics current and
                # frees the network for the next /submit ASAP.
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

                # Post-proof /state recheck — abort doomed submits when
                # the window rolled over or left OPEN during proof build.
                # Mirrors the pre-proof guard: must check both window_n
                # AND WindowState.OPEN so we catch window_not_active
                # (same window_n, but now in TRAINING/PUBLISHING).
                #
                # Skip-fast optimisation (v4.2): when the pre-proof
                # /state check was recent AND the proof build was
                # cheap, the chance of a window transition in that
                # short interval is negligible. Skipping the recheck
                # saves a ~50-300 ms HTTP round-trip that would
                # otherwise eat into the OPEN-window submit budget.
                # v4.2: tuned in lockstep with _STATE_PROBE_TIMEOUT_S
                # = 5 s. A healthy /state poll takes ~50-300 ms; the
                # 8 s ceiling accommodates one slow probe (up to 5 s)
                # PLUS up to ~3 s of proof build before the fast-path
                # skip is invalidated. Slow probes are not a "window
                # rolled" signal — they're just network jitter — so
                # forcing a second probe would only waste another ~5 s.
                _skip_post_check_max_age_s = 8.0
                _skip_post_check_max_proof_s = 5.0
                _can_skip_post_check = (
                    _pre_proof_check_t is not None
                    and (time.monotonic() - _pre_proof_check_t)
                    < _skip_post_check_max_age_s
                    and (proof_ms / 1000.0) < _skip_post_check_max_proof_s
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
                        # Promoted from DEBUG → INFO. If the post-proof
                        # /state probe times out, we DON'T know whether
                        # the window rolled — fall through and try the
                        # submit anyway. Worst case the validator
                        # rejects with window_not_active and we record
                        # that explicitly; best case we got lucky and
                        # the window is still open. Either way, the
                        # operator gets to see the state-probe failure.
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

                # Serialise: if the previous submit task is still
                # in-flight (rare — GEN ≈ 36 s > typical http ≈ 20 s)
                # await it now so we never have two concurrent submits.
                if _pending_task is not None and not _pending_task.done():
                    logger.debug(
                        "[W=%d] awaiting previous submit task before "
                        "firing new one", state.window_n,
                    )
                    await _drain_submit(_pending_task, _pending_ctx)
                    _pending_task = None
                    _pending_ctx = None

                # Snapshot open_age and gpu at the moment we FIRE the
                # submit (not when the response arrives) so the SUB log
                # reflects when the request was actually sent.
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
                )
                # Fire HTTP in the background — the GPU is free to start
                # the next PICK → GEN immediately.  The SUB log line and
                # all metrics updates happen in _drain_submit at the top
                # of the next iteration (or during the serialise await
                # above if the next attempt also needs to submit).
                _pending_task = asyncio.create_task(
                    _fire_submit(request),
                    name=f"submit-w{state.window_n}-p{prompt_idx}",
                )
                logger.debug(
                    "[W=%d] submit task launched prompt=%d open_age=%.1fs; "
                    "continuing to next pick immediately",
                    state.window_n, prompt_idx, _sub_open_age,
                )

        # Drain any in-flight submit task before returning so its result
        # lands in ``results`` and metrics are up-to-date for the caller.
        if _pending_task is not None:
            try:
                await _drain_submit(_pending_task, _pending_ctx)
            except asyncio.CancelledError:
                pass
            _pending_task = None

        return results

    # ------------------------------------------------------------------
    # Observed-OPEN-duration tracking (v4.2 picker realism)
    # ------------------------------------------------------------------

    def _record_observed_open_duration(self, duration_s: float) -> None:
        """Record how long the previous window stayed in OPEN.

        Called at each window-roll boundary with
        ``now - _window_open_seen_at`` of the window that just closed.
        Bounded to a sensible range so a freak partial observation
        (e.g. miner started mid-window) doesn't poison the picker.
        """
        if duration_s <= 0:
            return
        # Discard absurdly short observations (< 30 s = miner restarted
        # mid-window and saw the tail of OPEN, not a full OPEN phase).
        # Cap at _OPEN_PHASE_BUDGET_S to keep arithmetic bounded — we
        # don't believe any window lasts longer than the configured
        # upper bound.
        if duration_s < 30.0:
            return
        clamped = min(duration_s, float(_OPEN_PHASE_BUDGET_S))
        self._observed_open_durations_s.append(clamped)

    def _effective_open_budget_s(self) -> float:
        """Return the realistic OPEN budget for the picker.

        Until we have ≥ 3 observations, fall back to the configured
        ``_OPEN_PHASE_BUDGET_S`` constant (we don't have enough data
        to clamp confidently). Once we do, return
        ``P25(observed) × safety_factor`` floored at
        ``_OBSERVED_OPEN_MIN_FLOOR_S`` and capped at the configured
        ``_OPEN_PHASE_BUDGET_S`` ceiling.

        Why P25 rather than ``min`` (v4.2 → v4.3 W=973-989 lesson):

        The original implementation used ``min(observed)``. That made
        a single 107 s window (W=973) pin eff_budget=91 s for every
        subsequent window — but the actual validator windows ranged
        107-250 s, mean ≈ 180 s. The picker was then idling in WAIT
        for 50-150 s per window because hard_deadline=78 s was way
        too tight against the real OPEN duration. We saw multiple
        late submits (open_age 80-118 s) succeed when the picker
        DID happen to push past, proving the budget was the
        bottleneck.

        P25 (25th percentile) drops the bottom 25 % of observations
        as outliers — a single short window no longer dominates,
        but a SUSTAINED run of short windows (e.g. validator under
        load) shifts P25 down within ~5 windows. The safety factor
        (0.85) keeps a 15 % margin against the P25 we're betting on,
        so the picker rarely commits to a pipeline that overshoots.

        For ``n < 5`` observations we use ``min`` (no statistical
        confidence in a percentile yet); for ``n ≥ 5`` we use
        ``sorted[n // 4]`` which is exactly the 25th-percentile
        index by the "lower" interpolation convention. The deque
        is bounded by ``_OBSERVED_OPEN_HISTORY = 20``.
        """
        n = len(self._observed_open_durations_s)
        if n < 3:
            return float(_OPEN_PHASE_BUDGET_S)
        sorted_durations = sorted(self._observed_open_durations_s)
        if n < 5:
            # Not enough samples for a confident percentile; use min.
            anchor = sorted_durations[0]
        else:
            # P25: the ``n // 4``-th element of the sorted deque.
            anchor = sorted_durations[n // 4]
        effective = anchor * _OBSERVED_OPEN_SAFETY_FACTOR
        effective = max(effective, _OBSERVED_OPEN_MIN_FLOOR_S)
        effective = min(effective, float(_OPEN_PHASE_BUDGET_S))
        return effective

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
