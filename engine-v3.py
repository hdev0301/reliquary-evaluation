"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

DEPLOYMENT: this file is the v3 of the reliquary-evaluation miner engine
and is intended to REPLACE the upstream ``reliquary/miner/engine.py`` in
the installed reliquary repo. Concretely on the miner box:

    cp engine-v3.py /root/reliquary/reliquary/miner/engine.py

The companion files (``launcher.py``, ``vllm_adapter.py``) live at the
reliquary repo root (``/root/reliquary/``). See ``launcher.py``'s
docstring for the full file layout and run command.

Both the stock ``reliquary mine`` CLI (which passes an HF model) and the
``launcher.py`` (which passes a ``VLLMAdapter``) continue to work — the
engine's ``_load_checkpoint`` detects the backend via the
``_is_vllm_adapter`` sentinel and dispatches accordingly. So copying
this file in does NOT break the HF path; it just adds the vLLM path
when the launcher is used.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.

**v3 adds real vLLM generation support** (the previous ``vllm_model`` was
actually an HF ``AutoModelForCausalLM`` despite the name). vLLM is 5–10×
faster than HF ``.generate()`` for the M_ROLLOUTS-sample workload, which
is the single biggest lever for resolving the ``window_mismatch`` race
on the busy 2–3 min validator window cycle.

The engine itself is generation-backend-agnostic — it just calls
``self.vllm_model.generate(...)`` with the HuggingFace signature. The
two backends are:

- **HuggingFace** (default, used by the stock ``reliquary mine`` CLI):
  ``vllm_model`` is an ``AutoModelForCausalLM`` instance; generation goes
  through HF kernels; ``_load_checkpoint`` rebuilds via
  ``from_pretrained``.

- **vLLM** (used by the ``launcher.py`` in this directory): ``vllm_model``
  is a ``VLLMAdapter`` (see ``vllm_adapter.py``) that exposes the same
  ``.generate()`` signature but dispatches to a vLLM ``LLM`` instance.
  ``_load_checkpoint`` detects the adapter via ``_is_vllm_adapter`` and
  calls its ``reload(local_path)`` method, which rebuilds the underlying
  vLLM engine in-place (vLLM has no hot weight-swap primitive).

All other paths — GRAIL proof construction, picker, probe, persistence —
are backend-agnostic and unchanged from v2.

Five corrections over the upstream reference miner that materially
improve acceptance rate (carried forward from v2):

  1. Per-window GRAIL randomness. The validator derives sketch randomness
     from ``block_hash(state.window_n)``; the miner must do the same for
     every window or every submission GRAIL-fails. Upstream computed it
     once at startup with ``window_start=0``.
  2. Bayesian zone-aware prompt selection. With M_ROLLOUTS=8 binary
     rewards, the validator's ``σ ≥ SIGMA_MIN`` gate is mathematically
     equivalent to ``2 ≤ k_solved ≤ 6``. The picker maintains a Beta
     posterior on each prompt's solve probability and Thompson-samples
     candidates, scoring each by ``P(2 ≤ Binomial(8, p) ≤ 6)``. This
     concentrates effort on prompts that are likely to be in-zone AND
     keeps an unseen-prompt exploration tail via the Beta(1,1) prior.
  3. Local OUT_OF_ZONE short-circuit. Even with the picker, some groups
     land out-of-zone; submitting them is a deterministic OUT_OF_ZONE
     reject and burns time in the SUPERSEDED race for the next prompt.
     We compute σ locally and skip the HTTP round-trip when σ < SIGMA_MIN.
  4. Probe-then-continue rollout generation with Bayesian abort. Instead
     of always paying for M_ROLLOUTS=8 rollouts up-front, we generate
     ``_PROBE_SIZE=3`` first and decide whether to fill out the group by
     computing the posterior-predictive in-zone probability ``P(k_total
     ∈ [2, 6] | k_probe, p̂_posterior, n_more=5)``. We abort when this
     drops below ``_PROBE_ABORT_THRESHOLD=0.30``. This generalizes the
     simpler ``k_probe ∈ {0, _PROBE_SIZE}`` rule with prior knowledge:
     unseen prompts that probe extreme abort (as before), but a warm
     mid-prompt that happens to roll k_probe=3/3 has a barely-shifted
     posterior and continues — the older rule would have wastefully
     aborted on the same observation.
  5. Posterior persistence. The Beta posterior is the only signal that
     accumulates across sessions; without persistence, a restart resets
     σ-yield to its initial uniform-random behaviour. Stats are saved
     atomically every ``_SUMMARY_EVERY`` attempts and loaded on startup.
"""

from __future__ import annotations

import asyncio
import logging
import math
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


# Emit a rolling counter summary every N submission attempts. Set high enough
# to avoid log spam, low enough to be a useful pulse during debugging. Also
# governs how often _PromptStats is flushed to disk.
_SUMMARY_EVERY = 10

# Cold-start probe size. We generate this many rollouts first, decide whether
# to continue based on k_probe and the prior posterior, and either abort
# (skipping the remaining M_ROLLOUTS - _PROBE_SIZE) or fill out a full group.
# _PROBE_SIZE=3 is the sweet spot for M=8 binary rewards: enough signal to
# update the posterior meaningfully, small enough to make the abort cheap.
_PROBE_SIZE = 3

# Abort threshold for the Bayesian probe: if posterior-predictive
# ``P(k_final ∈ [2, 6] | k_probe, posterior_mean, n_more)`` drops below
# this, the expected value of continuing the remaining ``M - _PROBE_SIZE``
# rollouts doesn't beat just trying another prompt. 0.30 reproduces the
# hardcoded ``k_probe ∈ {0, _PROBE_SIZE}`` rule on Beta(1,1) priors and
# generalizes correctly when prior history is informative — e.g. a
# historically-mid prompt that probes 3/3 doesn't abort because the
# posterior is barely shifted.
_PROBE_ABORT_THRESHOLD = 0.30

# Skip the probe entirely once a prompt has been attempted this many times.
# The picker chose a warm prompt because its posterior says it's mid-difficulty
# (high zone_p) — re-verifying with a probe is pure overhead because two
# sequential ``.generate()`` calls (3 then 5) take longer than one batched
# call of 8 due to poor GPU re-use across kernel launches. The probe is only
# valuable on cold prompts where it can short-circuit the abort path.
_PROBE_WARM_THRESHOLD = 5

# Default location for the on-disk posterior cache. Resolved at engine
# construction; the miner can override via the ``stats_path`` kwarg (set to
# ``None`` to disable persistence).
_DEFAULT_STATS_PATH = ".reliquary_miner_stats.json"


@dataclass
class _MinerMetrics:
    """Rolling counters surfaced to logs so an operator can see at a glance
    which rejection reason is dominating and how the local zone-prediction
    matches what the validator returns.
    """

    submitted: int = 0
    accepted: int = 0
    rejected: int = 0
    network_errors: int = 0
    # Total rollout groups generated (= submitted + local_out_of_zone).
    # Use ``in_zone_rate`` to track picker quality — the fraction of
    # generated groups that pass the σ ≥ SIGMA_MIN gate locally.
    generated: int = 0
    # Local short-circuits (we declined to submit because we predicted
    # OUT_OF_ZONE from the rewards distribution).
    local_out_of_zone: int = 0
    # Probe aborts (cold-start ``_PROBE_SIZE`` rollouts produced an extreme
    # k_probe; the remaining M_ROLLOUTS - _PROBE_SIZE rollouts were skipped).
    # Use ``probe_abort_rate`` to track how often the probe is paying off;
    # if abort_rate is very low (< 15%) the probe overhead dominates and
    # we should consider disabling it.
    probe_aborts: int = 0
    probe_attempts: int = 0
    # Histogram of k_solved over all generated groups — quick visual on
    # whether the model + picker is producing a healthy mid-difficulty
    # distribution or piling up at the extremes.
    k_histogram: list[int] = field(default_factory=lambda: [0] * (M_ROLLOUTS + 1))
    # Histogram of k_probe over all probe attempts (0..._PROBE_SIZE).
    probe_histogram: list[int] = field(default_factory=lambda: [0] * (_PROBE_SIZE + 1))
    reasons: dict[str, int] = field(default_factory=dict)

    def record_generation(self, k_solved: int) -> None:
        self.generated += 1
        if 0 <= k_solved <= M_ROLLOUTS:
            self.k_histogram[k_solved] += 1

    def record_probe(self, k_probe: int, aborted: bool) -> None:
        self.probe_attempts += 1
        if 0 <= k_probe <= _PROBE_SIZE:
            self.probe_histogram[k_probe] += 1
        if aborted:
            self.probe_aborts += 1

    def record(self, accepted: bool, reason: str | None) -> None:
        self.submitted += 1
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        if reason is not None:
            self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def record_local_oos(self) -> None:
        self.local_out_of_zone += 1

    def record_network_error(self) -> None:
        self.network_errors += 1

    @property
    def in_zone_rate(self) -> float:
        return (
            (self.generated - self.local_out_of_zone) / self.generated * 100.0
            if self.generated else 0.0
        )

    @property
    def probe_abort_rate(self) -> float:
        return (
            self.probe_aborts / self.probe_attempts * 100.0
            if self.probe_attempts else 0.0
        )

    def summary(self, stats: "_PromptStats | None" = None) -> str:
        # Compact one-liner with the picker's key signals:
        #   - in_zone_rate: % of generated groups that pass σ ≥ SIGMA_MIN
        #     locally. Climbs as the Beta posterior warms up.
        #   - probe_abort: how often the cold-start probe aborted before
        #     paying for the full 8-rollout batch. Higher early, lower as
        #     posteriors concentrate on mid-difficulty prompts.
        #   - warmed/observed: how many prompts have ≥5 attempts vs touched.
        #     Picker exploitation strengthens with warmed count.
        rate = (self.accepted / self.submitted * 100.0) if self.submitted else 0.0
        top = sorted(self.reasons.items(), key=lambda kv: -kv[1])[:4]
        top_str = ",".join(f"{r}:{c}" for r, c in top) or "-"
        warm_str = ""
        if stats is not None:
            warmed, observed = stats.warmed_count()
            warm_str = f" warmed={warmed}/{observed}"
        k_str = "/".join(str(c) for c in self.k_histogram)  # k=0..M
        probe_str = (
            f" probe_abort={self.probe_aborts}/{self.probe_attempts}"
            f"({self.probe_abort_rate:.0f}%)"
            if self.probe_attempts else ""
        )
        return (
            f"generated={self.generated} in_zone={self.in_zone_rate:.1f}% "
            f"submitted={self.submitted} accepted={self.accepted} "
            f"({rate:.1f}%) rejected={self.rejected} "
            f"local_oos={self.local_out_of_zone} net_err={self.network_errors}"
            f"{probe_str}{warm_str} k_hist=[{k_str}] top=[{top_str}]"
        )


def _zone_status(rewards: list[float]) -> tuple[float, int, bool]:
    """Compute (sigma, k_solved, in_zone) for a rollout group.

    ``k_solved`` is the number of rollouts with reward >= 0.5; under the
    math env's binary reward this is the only thing that matters for the
    validator's ``rewards_std`` and ``is_in_zone`` gates. We replicate the
    same population-stddev formula the validator uses (see
    ``validator/verifier.py::rewards_std``).
    """
    n = len(rewards)
    if n < 2:
        return 0.0, 0, False
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    sigma = variance ** 0.5
    k_solved = sum(1 for r in rewards if r >= 0.5)
    return sigma, k_solved, sigma >= SIGMA_MIN


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

    state.checkpoint_repo_id + state.checkpoint_revision identify the
    HF snapshot. download_fn/load_fn still injected for testability.

    Returns ``(new_local_n, new_local_hash, new_model)``. If no update is
    needed (remote ≤ local, or remote has no repo/revision yet), returns
    inputs unchanged.
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
    import asyncio
    from huggingface_hub import snapshot_download

    return await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["model.safetensors", "config.json", "tokenizer*"],
    )


# Binomial coefficients C(M_ROLLOUTS, k) for k=0..M_ROLLOUTS. Used by
# ``_zone_probability``. Computed at module load so the hot path is just
# multiplications and adds. The protocol fixes M_ROLLOUTS at 8 so this
# tuple is tiny and never recomputed.
_BINOM_M: tuple[int, ...] = tuple(math.comb(M_ROLLOUTS, k) for k in range(M_ROLLOUTS + 1))

# In-zone gate (binary reward): σ ≥ SIGMA_MIN ⇔ k_solved ∈ [K_LO, K_HI].
# For M=8, SIGMA_MIN=0.43:
#   σ(k=1) = 0.331  σ(k=2) = 0.433  σ(k=6) = 0.433  σ(k=7) = 0.331
# So the integer band is [2, 6].
_K_LO: int = 2
_K_HI: int = 6


def _zone_probability(p: float) -> float:
    """Return ``P(K_LO ≤ Binomial(M_ROLLOUTS, p) ≤ K_HI)``.

    This is the probability that a group of M_ROLLOUTS rollouts at a given
    per-rollout solve probability ``p`` will pass the validator's
    σ ≥ SIGMA_MIN gate. Used as the scoring function for prompt selection.

    Sample values (M=8, band=[2,6]):
        p=0.10 → 0.187    p=0.50 → 0.992
        p=0.20 → 0.703    p=0.60 → 0.989  (symmetric with p=0.4)
        p=0.30 → 0.940    p=0.80 → 0.703
        p=0.40 → 0.989    p=0.90 → 0.187
    """
    p = max(0.0, min(1.0, p))
    q = 1.0 - p
    total = 0.0
    for k in range(_K_LO, min(_K_HI, M_ROLLOUTS) + 1):
        total += _BINOM_M[k] * (p ** k) * (q ** (M_ROLLOUTS - k))
    return total


def _continuation_in_zone_probability(
    k_probe: int, p_hat: float, n_more: int,
) -> float:
    """Posterior-predictive ``P(k_total ∈ [K_LO, K_HI] | k_probe, p̂, n_more)``.

    After the probe observes ``k_probe`` solves out of ``_PROBE_SIZE``,
    the remaining ``n_more`` rollouts are modelled as
    ``Binomial(n_more, p̂)`` where ``p̂`` is the posterior mean of the
    solve rate (the Beta posterior updated by the probe's outcomes).
    The group lands in-zone iff ``k_total = k_probe + k_more ∈ [K_LO, K_HI]``,
    which means ``k_more ∈ [K_LO - k_probe, K_HI - k_probe]`` clamped to
    ``[0, n_more]``.

    Used by the probe abort gate. Lower than ``_PROBE_ABORT_THRESHOLD`` →
    skip the remaining rollouts. Properly Bayesian: it generalizes the
    hardcoded ``k_probe ∈ {0, _PROBE_SIZE}`` rule with prior knowledge.
    Worked example: warm prompt with historic p≈0.5 probes 3/3 →
    posterior barely shifts → continuation P ≈ 0.73 → continue (the old
    rule would have aborted).
    """
    k_more_lo = max(0, _K_LO - k_probe)
    k_more_hi = min(n_more, _K_HI - k_probe)
    if k_more_lo > k_more_hi:
        return 0.0
    p = max(0.0, min(1.0, p_hat))
    q = 1.0 - p
    total = 0.0
    for k in range(k_more_lo, k_more_hi + 1):
        total += math.comb(n_more, k) * (p ** k) * (q ** (n_more - k))
    return total


class _PromptStats:
    """Beta-Bernoulli posterior on per-prompt solve probability.

    For each prompt we accumulate ``(solves, attempts)`` over all rollouts
    the miner has executed against it. With a Beta(α₀, β₀) prior, the
    posterior solve probability is ``Beta(α₀ + solves, β₀ + attempts -
    solves)``. The picker Thompson-samples ``p`` from this posterior and
    scores candidates by the resulting in-zone probability — see
    ``pick_prompt_idx``.

    With the default Beta(1,1) (uniform) prior, unseen prompts have a wide
    posterior so Thompson samples spread across [0,1] — exploration is
    automatic. Well-observed prompts converge tightly to the empirical
    rate, so they're scored deterministically.

    Also tracks a running mean of completion length per prompt; the picker
    can use this as a tiebreak to favour faster-to-solve prompts (winning
    the validator's TCP-arrival SUPERSEDED race more often).
    """

    __slots__ = ("alpha_prior", "beta_prior", "_counts", "_lengths")

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> None:
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self._counts: dict[int, tuple[int, int]] = {}        # idx → (solves, attempts)
        self._lengths: dict[int, tuple[float, int]] = {}     # idx → (mean_len, n)

    def record_group(
        self,
        prompt_idx: int,
        rewards: list[float],
        completion_lens: list[int] | None = None,
    ) -> None:
        solves_prev, attempts_prev = self._counts.get(prompt_idx, (0, 0))
        attempts = len(rewards)
        solves = sum(1 for r in rewards if r >= 0.5)
        self._counts[prompt_idx] = (
            solves_prev + solves, attempts_prev + attempts,
        )
        if completion_lens:
            prev_mean, prev_n = self._lengths.get(prompt_idx, (0.0, 0))
            total_n = prev_n + len(completion_lens)
            new_mean = (prev_mean * prev_n + sum(completion_lens)) / total_n
            self._lengths[prompt_idx] = (new_mean, total_n)

    def posterior(self, prompt_idx: int) -> tuple[float, float]:
        """Return ``(α, β)`` of the Beta posterior for *prompt_idx*."""
        solves, attempts = self._counts.get(prompt_idx, (0, 0))
        return (
            self.alpha_prior + solves,
            self.beta_prior + (attempts - solves),
        )

    def sample_p(self, prompt_idx: int, rng: _random.Random) -> float:
        """Thompson-sample a solve probability from the posterior."""
        a, b = self.posterior(prompt_idx)
        return rng.betavariate(a, b)

    def mean_p(self, prompt_idx: int) -> float:
        a, b = self.posterior(prompt_idx)
        return a / (a + b)

    def attempts(self, prompt_idx: int) -> int:
        _, n = self._counts.get(prompt_idx, (0, 0))
        return n

    def avg_completion_len(self, prompt_idx: int) -> float | None:
        v = self._lengths.get(prompt_idx)
        return v[0] if v else None

    def warmed_count(self, min_attempts: int = 5) -> tuple[int, int]:
        """Return ``(warmed, observed)`` — count of prompts with ≥
        ``min_attempts`` rollouts vs. total prompts touched.

        Used to surface picker-warmup progress in the rolling metrics
        line; until ``warmed`` is in the hundreds the picker is still
        in exploration mode and σ-yield will lag steady-state.
        """
        observed = len(self._counts)
        warmed = sum(1 for solves_attempts in self._counts.values()
                     if solves_attempts[1] >= min_attempts)
        return warmed, observed

    def save_to(self, path: str) -> None:
        """Atomically persist posterior + length stats to *path* as JSON.

        Atomic-write pattern (write to ``path.tmp``, then ``os.replace``)
        guarantees the file is never partially-written even if the miner
        is killed mid-save — the validator's SUPERSEDED race makes
        graceful shutdown unreliable.
        """
        import json
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "counts": {str(k): list(v) for k, v in self._counts.items()},
            "lengths": {str(k): list(v) for k, v in self._lengths.items()},
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    def load_from(self, path: str) -> bool:
        """Load posterior + length stats from *path*. Returns True if loaded.

        Silently returns False if the file doesn't exist or is corrupt —
        a missing/malformed cache is not a fatal error, we just start
        with fresh priors. Corrupt caches usually result from a crash
        during save in older versions without atomic-replace; deleting
        the file is the right recovery.
        """
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
        return True


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    stats: _PromptStats | None = None,
    candidates: int = 32,
    max_attempts: int = 1000,
) -> int:
    """Best-of-K Thompson-sampled prompt selection optimised for GRPO yield.

    Strategy:

    1. Draw up to ``candidates`` distinct non-cooldown prompt indices
       uniformly at random.
    2. For each candidate, Thompson-sample a solve probability ``p`` from
       its Beta posterior and score it by ``_zone_probability(p)`` — the
       chance that an M_ROLLOUTS group will land in the σ ≥ SIGMA_MIN
       band. New prompts (Beta(1,1) prior) have wide posteriors so they
       get genuine exploration without an explicit ε-greedy schedule.
    3. Return the candidate with the highest sampled in-zone score. Ties
       are broken in favour of shorter average completion length when
       known (faster generation → more SUPERSEDED wins on subsequent
       prompts in the same window).

    ``candidates=32`` is a deliberate balance: large enough that warm
    prompts (concentrated Beta near p≈0.5, zone_p≈0.99) consistently
    surface, small enough that Thompson exploration on unseen Beta(1,1)
    prompts still wins ~5–10% of picks. K=16 was too noisy during the
    warmup phase; K≫64 over-exploits and starves exploration.

    Without ``stats`` (test path), falls back to uniform random with
    cooldown skip — backward compatible with the upstream miner.

    Raises ``RuntimeError`` if no eligible prompt can be found — typically
    because the env is fully in cooldown.
    """
    rng = rng or _random
    n = len(env)
    over_half_cooldown = len(cooldown_prompts) >= n / 2
    eligible_list: list[int] | None = None
    if over_half_cooldown:
        eligible_list = [i for i in range(n) if i not in cooldown_prompts]
        if not eligible_list:
            raise RuntimeError("no eligible prompt — env fully in cooldown")

    def _draw_one() -> int | None:
        if eligible_list is not None:
            return rng.choice(eligible_list)
        for _ in range(max_attempts):
            idx = rng.randrange(n)
            if idx not in cooldown_prompts:
                return idx
        return None

    if stats is None:
        # Test / bootstrap path: uniform random with cooldown skip.
        idx = _draw_one()
        if idx is None:
            raise RuntimeError("no eligible prompt — env fully in cooldown")
        return idx

    # Thompson-sampled best-of-K.
    seen: set[int] = set()
    best_idx: int | None = None
    best_score: float = -1.0
    best_len: float = float("inf")
    for _ in range(candidates):
        idx = _draw_one()
        if idx is None:
            break
        if idx in seen:
            continue
        seen.add(idx)
        p_sampled = stats.sample_p(idx, rng)
        score = _zone_probability(p_sampled)
        if score > best_score:
            best_idx, best_score = idx, score
            best_len = stats.avg_completion_len(idx) or float("inf")
        elif score == best_score:
            # Tiebreak: prefer the prompt whose historical generation is
            # cheaper, so we can race the SUPERSEDED window faster.
            alt_len = stats.avg_completion_len(idx) or float("inf")
            if alt_len < best_len:
                best_idx, best_len = idx, alt_len

    if best_idx is None:
        raise RuntimeError("no eligible prompt — env fully in cooldown")
    return best_idx


def _compute_merkle_root(rollouts) -> str:
    """Compute Merkle root over rollout leaves — returns 64-char hex.

    Uses canonical JSON (sort_keys=True, compact separators) for dict/list
    serialisation so the root is deterministic across Python
    implementations and refactor-stable against dict-construction-order
    changes.
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


class MiningEngine:
    """Two-GPU mining: vLLM (GPU 0) for generation, HF (GPU 1) for proofs."""

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

        # Lazy imports for heavy deps — keep module import cheap.
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

        # Cached per-window randomness so we don't re-derive it on every
        # iteration of the polling loop within the same window.
        self._cached_window_n: int | None = None
        self._cached_randomness: str = ""

        # Per-prompt success-rate tracker used by ``pick_prompt_idx`` to
        # avoid prompts that recently produced all-correct or all-wrong
        # groups (guaranteed OUT_OF_ZONE under the σ ≥ SIGMA_MIN gate).
        # Restored from disk if a cached posterior exists so a miner restart
        # doesn't reset σ-yield to its uniform-random baseline.
        self._prompt_stats = _PromptStats()
        self._stats_path = stats_path
        if self._stats_path:
            try:
                if self._prompt_stats.load_from(self._stats_path):
                    warmed, observed = self._prompt_stats.warmed_count()
                    logger.info(
                        "loaded prompt stats from %s: observed=%d warmed=%d",
                        self._stats_path, observed, warmed,
                    )
            except Exception:
                logger.exception(
                    "failed to load prompt stats from %s; starting fresh",
                    self._stats_path,
                )
        # Save-throttling — only flush to disk every _SUMMARY_EVERY
        # generations so we don't pay JSON-encode cost on the hot path.
        self._save_counter: int = 0

        # Rolling counters for the structured-summary log emitted every
        # ``_SUMMARY_EVERY`` submission attempts and on window transitions.
        self._metrics = _MinerMetrics()

    def _maybe_persist_stats(self) -> None:
        """Flush the Beta posterior + length stats to disk every
        ``_SUMMARY_EVERY`` writes. Bound to the same cadence as the metrics
        summary so an operator can correlate the two log lines.

        Atomic-write inside ``save_to`` guarantees we never leave a
        truncated file even if the miner is killed mid-flush. A failed
        save is logged but never raised — persistence is best-effort.
        """
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
    # Public API
    # ------------------------------------------------------------------

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,  # v2.0 param kept for CLI compat; ignored
        use_drand: bool = True,
    ) -> list:
        """v2.1: poll state, pull checkpoint on n-change, submit when OPEN.

        Returns the list of BatchSubmissionResponse objects collected
        across the loop. The loop exits only on external cancellation
        (asyncio.CancelledError) or if env becomes fully cooldown'd.
        """
        import httpx
        import random

        from reliquary.constants import POLL_INTERVAL_SECONDS
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
            get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import WindowState

        # Resolve validator URL (once).
        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)
        logger.info(
            "mine_window start: hotkey=%s validator=%s M=%d T=%.2f",
            self.wallet.hotkey.ss58_address[:12], url, M_ROLLOUTS, T_PROTO,
        )

        rng = random.Random()
        results = []
        local_n = 0
        local_hash = ""
        last_window_n: int | None = None

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                try:
                    state = await get_window_state_v2(url, client=client)
                except SubmissionError:
                    # /state may return 503 between windows; wait briefly.
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                except Exception as e:
                    logger.debug("state fetch failed: %s", e)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # Pull new checkpoint if needed (works at any state).
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
                    logger.debug(
                        "state=%s (not OPEN) window_n=%d — waiting",
                        state.state.value if hasattr(state.state, "value") else state.state,
                        state.window_n,
                    )
                    await asyncio.sleep(1)
                    continue

                # Window transition: emit rolling summary + reset prompt
                # selection signal (don't reset stats; they're cross-window).
                if last_window_n != state.window_n:
                    if last_window_n is not None:
                        logger.info(
                            "window %d → %d | %s",
                            last_window_n, state.window_n,
                            self._metrics.summary(self._prompt_stats),
                        )
                    else:
                        logger.info(
                            "first OPEN window window_n=%d valid_so_far=%d "
                            "cooldown_prompts=%d checkpoint_n=%d",
                            state.window_n, state.valid_submissions,
                            len(state.cooldown_prompts), state.checkpoint_n,
                        )
                    last_window_n = state.window_n

                # Per-window randomness: the validator re-derives the GRAIL
                # sketch seed from block_hash(state.window_n) for each window,
                # so the miner MUST match that or every submission GRAIL-fails.
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

                # Pick prompt, generate, submit.
                cooldown_set = set(state.cooldown_prompts)
                try:
                    prompt_idx = pick_prompt_idx(
                        self.env, cooldown_set,
                        rng=rng, stats=self._prompt_stats,
                    )
                except RuntimeError:
                    logger.info(
                        "env fully in cooldown (cooldown_set=%d / env_len=%d); sleeping",
                        len(cooldown_set), len(self.env),
                    )
                    await asyncio.sleep(5)
                    continue

                problem = self.env.get_problem(prompt_idx)
                a, b = self._prompt_stats.posterior(prompt_idx)
                p_hat = self._prompt_stats.mean_p(prompt_idx)
                attempts = self._prompt_stats.attempts(prompt_idx)
                logger.debug(
                    "pick prompt=%d posterior=Beta(%.1f,%.1f) p_hat=%.2f "
                    "attempts=%d zone_p=%.2f problem_id=%s",
                    prompt_idx, a, b, p_hat, attempts,
                    _zone_probability(p_hat), problem.get("id", "?"),
                )

                # Probe-or-full decision: cold prompts use the probe so we
                # can short-circuit on extreme observations; warm prompts
                # (≥ _PROBE_WARM_THRESHOLD attempts) skip the probe because
                # the picker already trusts the posterior and the split
                # batch (3 then 5) takes longer than one batch of 8 due to
                # poor GPU re-use across two kernel launches.
                if attempts >= _PROBE_WARM_THRESHOLD:
                    logger.debug(
                        "skip probe (warm) prompt=%d attempts=%d ≥ %d",
                        prompt_idx, attempts, _PROBE_WARM_THRESHOLD,
                    )
                    t_gen = time.monotonic()
                    generations = self._generate_n_rollouts(problem, M_ROLLOUTS)
                    gen_ms = (time.monotonic() - t_gen) * 1000.0
                    probe_ms = 0.0
                    more_ms = gen_ms
                    if len(generations) < M_ROLLOUTS:
                        logger.warning(
                            "generated %d/%d for prompt %d (gen_ms=%.0f); skipping",
                            len(generations), M_ROLLOUTS, prompt_idx, gen_ms,
                        )
                        continue
                else:
                    # Cold-start probe. Generate _PROBE_SIZE rollouts first,
                    # decide whether to fill out the group based on the
                    # Bayesian posterior-predictive in-zone probability.
                    t_probe = time.monotonic()
                    probe_gens = self._generate_n_rollouts(problem, _PROBE_SIZE)
                    probe_ms = (time.monotonic() - t_probe) * 1000.0
                    if len(probe_gens) < _PROBE_SIZE:
                        logger.warning(
                            "probe generated %d/%d for prompt %d (probe_ms=%.0f); skipping",
                            len(probe_gens), _PROBE_SIZE, prompt_idx, probe_ms,
                        )
                        continue

                    probe_scored = [self._score_rollout(g, problem) for g in probe_gens]
                    probe_rewards = [r for r, _ in probe_scored]
                    probe_lens = [length for _, length in probe_scored]
                    k_probe = sum(1 for r in probe_rewards if r >= 0.5)

                    a_prior, b_prior = self._prompt_stats.posterior(prompt_idx)
                    a_post = a_prior + k_probe
                    b_post = b_prior + (_PROBE_SIZE - k_probe)
                    p_hat_post = a_post / (a_post + b_post)
                    n_more = M_ROLLOUTS - _PROBE_SIZE
                    p_continue_zone = _continuation_in_zone_probability(
                        k_probe, p_hat_post, n_more,
                    )

                    if p_continue_zone < _PROBE_ABORT_THRESHOLD:
                        # Predicted continuation probability too low — abort.
                        # Record probe outcomes so the posterior moves toward
                        # the observed rate (informative for future picks).
                        self._prompt_stats.record_group(
                            prompt_idx, probe_rewards, completion_lens=probe_lens,
                        )
                        self._metrics.record_probe(k_probe, aborted=True)
                        logger.info(
                            "probe abort window=%d prompt=%d k_probe=%d/%d "
                            "p_hat=%.2f p_continue_zone=%.2f probe_ms=%.0f "
                            "(saved %d rollouts)",
                            state.window_n, prompt_idx, k_probe, _PROBE_SIZE,
                            p_hat_post, p_continue_zone, probe_ms,
                            M_ROLLOUTS - _PROBE_SIZE,
                        )
                        self._maybe_persist_stats()
                        if self._metrics.probe_attempts % _SUMMARY_EVERY == 0:
                            logger.info(
                                "metrics | %s",
                                self._metrics.summary(self._prompt_stats),
                            )
                        continue

                    # Probe wasn't extreme — fill out the group.
                    logger.debug(
                        "probe continue prompt=%d k_probe=%d/%d p_hat=%.2f "
                        "p_continue_zone=%.2f",
                        prompt_idx, k_probe, _PROBE_SIZE,
                        p_hat_post, p_continue_zone,
                    )
                    self._metrics.record_probe(k_probe, aborted=False)
                    t_more = time.monotonic()
                    more_gens = self._generate_n_rollouts(
                        problem, M_ROLLOUTS - _PROBE_SIZE
                    )
                    more_ms = (time.monotonic() - t_more) * 1000.0
                    gen_ms = probe_ms + more_ms
                    if len(more_gens) < M_ROLLOUTS - _PROBE_SIZE:
                        logger.warning(
                            "continuation generated %d/%d for prompt %d "
                            "(gen_ms=%.0f); skipping",
                            len(more_gens), M_ROLLOUTS - _PROBE_SIZE,
                            prompt_idx, gen_ms,
                        )
                        continue
                    generations = probe_gens + more_gens

                t_proof = time.monotonic()
                rollout_submissions = [
                    self._build_rollout_submission(gen, problem, randomness)
                    for gen in generations
                ]
                proof_ms = (time.monotonic() - t_proof) * 1000.0

                rewards = [r.reward for r in rollout_submissions]
                sigma, k_solved, in_zone = _zone_status(rewards)

                completion_lens = [
                    len(r.tokens) - r.commit["rollout"]["prompt_length"]
                    for r in rollout_submissions
                ]
                # Update the posterior BEFORE the submission round-trip so
                # the signal updates even when /submit fails. completion_lens
                # feeds the SUPERSEDED-race tiebreak in pick_prompt_idx.
                self._prompt_stats.record_group(
                    prompt_idx, rewards, completion_lens=completion_lens,
                )
                self._metrics.record_generation(k_solved)
                self._maybe_persist_stats()
                logger.info(
                    "rollouts ready window=%d prompt=%d k=%d/%d sigma=%.3f "
                    "in_zone=%s probe_ms=%.0f more_ms=%.0f proof_ms=%.0f "
                    "completion_len[min/med/max]=%d/%d/%d",
                    state.window_n, prompt_idx, k_solved, M_ROLLOUTS, sigma, in_zone,
                    probe_ms, more_ms, proof_ms,
                    min(completion_lens),
                    sorted(completion_lens)[len(completion_lens) // 2],
                    max(completion_lens),
                )

                # Local short-circuit: if our own rewards distribution can't
                # clear the validator's zone gate, skip the HTTP round-trip
                # entirely — the verdict is deterministic OUT_OF_ZONE and we
                # save bandwidth + lose less ground on the SUPERSEDED race
                # for the next prompt. Bootstrap-window's relaxed threshold
                # is ignored here intentionally; the steady-state cost of
                # the extra check is one submission per ~100 windows.
                if not in_zone:
                    self._metrics.record_local_oos()
                    logger.info(
                        "local OUT_OF_ZONE (sigma=%.3f < %.2f) prompt=%d k=%d/%d — skipping submit",
                        sigma, SIGMA_MIN, prompt_idx, k_solved, M_ROLLOUTS,
                    )
                    if self._metrics.generated % _SUMMARY_EVERY == 0:
                        logger.info("metrics | %s", self._metrics.summary(self._prompt_stats))
                    continue

                merkle_root = _compute_merkle_root(rollout_submissions)

                # Pre-submit state check: if generation took long enough that
                # the validator's window advanced (windows seal at B valid
                # submissions, typically every ~2-3 min), our request is born
                # doomed — the validator will return 409 / WINDOW_MISMATCH and
                # we'll have wasted both the long POST round-trip (the
                # validator queues submissions behind GRAIL verifications) and
                # the time we could have spent generating for the new window.
                # A cheap GET /state here catches this case explicitly and
                # bails before the doomed POST. Tradeoff: ~20s extra per
                # attempt vs. ~80s saved per doomed attempt. Net positive
                # whenever generation is slower than the window cycle.
                try:
                    fresh_state = await get_window_state_v2(url, client=client)
                    if fresh_state.window_n != state.window_n:
                        logger.warning(
                            "window advanced %d → %d during generation "
                            "(gen_ms=%.0f) — skipping doomed submit for prompt=%d",
                            state.window_n, fresh_state.window_n,
                            gen_ms, prompt_idx,
                        )
                        continue
                except SubmissionError as e:
                    # State endpoint refused (503 / 4xx) — proceed and let
                    # the actual /submit return the canonical verdict. Not
                    # worth bailing on a state poll glitch.
                    logger.debug(
                        "pre-submit state check failed: %s; proceeding with submit",
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
                t_http = time.monotonic()
                try:
                    resp = await submit_batch_v2(url, request, client=client)
                    http_ms = (time.monotonic() - t_http) * 1000.0
                    reason_str = (
                        resp.reason.value if hasattr(resp.reason, "value")
                        else str(resp.reason)
                    )
                    self._metrics.record(resp.accepted, reason_str)
                    log_fn = logger.info if resp.accepted else logger.warning
                    log_fn(
                        "submit window=%d prompt=%d k=%d/%d sigma=%.3f "
                        "merkle=%s gen_ms=%.0f proof_ms=%.0f http_ms=%.0f "
                        "accepted=%s reason=%s",
                        state.window_n, prompt_idx, k_solved, M_ROLLOUTS, sigma,
                        merkle_root[:12], gen_ms, proof_ms, http_ms,
                        resp.accepted, reason_str,
                    )
                    results.append(resp)
                except SubmissionError as exc:
                    self._metrics.record_network_error()
                    logger.error(
                        "submit network/4xx failure window=%d prompt=%d: %s",
                        state.window_n, prompt_idx, exc,
                    )

                if self._metrics.submitted and self._metrics.submitted % _SUMMARY_EVERY == 0:
                    logger.info("metrics | %s", self._metrics.summary(self._prompt_stats))

        return results

    def _load_checkpoint(self, local_path: str):
        """Reload both hf_model and vllm_model from *local_path*.

        Both attributes are ``AutoModelForCausalLM`` instances despite the
        historical ``vllm_model`` naming — vllm_model is the fast-generation
        copy on ``self.vllm_gpu``, hf_model is the GRAIL-proof copy on
        ``self.proof_gpu``.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from reliquary.constants import ATTN_IMPLEMENTATION

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        # 1. Reload hf_model (for GRAIL proofs) on the proof GPU.
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

        # 2. Reload vllm_model on the generation GPU.
        # Two paths: if ``self.vllm_model`` is a ``VLLMAdapter`` (detected
        # by the ``_is_vllm_adapter`` sentinel), call its ``reload()`` which
        # tears down the underlying vLLM ``LLM`` and rebuilds it on the
        # same GPU — vLLM has no clean weight-swap primitive. Otherwise
        # fall back to the HF ``AutoModelForCausalLM.from_pretrained`` path
        # used by the stock CLI / single-stack-HF installs.
        if getattr(self.vllm_model, "_is_vllm_adapter", False):
            try:
                self.vllm_model.reload(local_path)
            except Exception:
                logger.exception(
                    "vLLM reload failed for %s; miner generation is BROKEN "
                    "until the next successful pull. hf_model was swapped "
                    "so GRAIL proofs will be inconsistent.",
                    local_path,
                )
                self.vllm_model = None
                self._loaded_checkpoint_path = None
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
                    "BROKEN until the next successful pull. hf_model was swapped "
                    "so GRAIL proofs will be inconsistent.",
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

    def _generate_n_rollouts(self, problem, n: int) -> list[dict]:
        """Generate ``n`` completions at T_PROTO in one batched call.

        One .generate() with batch shape (n, prompt_len) is ~5-7× faster
        on GPU than n serial calls — the matmul tiling utilizes far more
        of the GPU's compute. Each row samples independently
        (do_sample=True), so GRPO-group semantics are preserved. Each
        output row is truncated at its first post-prompt EOS so trailing
        batch-padding (HF pads with pad_token_id = eos_token_id) is not
        carried downstream — otherwise the validator's GRAIL forward
        pass would see extra EOS tokens the miner didn't "generate" in
        the usual sense.

        Called twice per attempt by ``mine_window``: once with
        ``n=_PROBE_SIZE`` for the cold-start probe, and (only if the
        probe doesn't abort) once with ``n=M_ROLLOUTS - _PROBE_SIZE`` to
        fill out the group.
        """
        import torch

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        # Dynamic max_new_tokens budget. The validator accepts max-length
        # termination iff ``prompt_length + completion_length >=
        # MAX_NEW_TOKENS_PROTOCOL_CAP`` ([verifier.py:90-91]). So we only
        # need to allow enough completion tokens that hitting the cap also
        # satisfies that inequality — anything beyond ``cap - prompt_length``
        # is wasted compute on the same termination outcome. For typical
        # 500-2000 token math prompts this saves 500-2000 tokens × 8
        # rollouts per attempt (~3-13s per attempt at typical throughput).
        # ``self.max_new_tokens`` (the constructor cap) is still honored as
        # an upper bound — set lower to deliberately risk BAD_TERMINATION
        # for further savings.
        budget = max(
            512,  # floor: ensure rollouts have meaningful generation room
            MAX_NEW_TOKENS_PROTOCOL_CAP - prompt_length,
        )
        effective_max_new = min(self.max_new_tokens, budget)

        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * n,
                device=getattr(self.vllm_model, "device", "cpu"),
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
        eos = self.tokenizer.eos_token_id
        rollouts = []
        for i in range(n):
            seq = outputs[i].tolist()
            gen = seq[prompt_length:]
            try:
                first_eos = gen.index(eos)
                gen = gen[: first_eos + 1]
            except ValueError:
                pass
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
            })
        return rollouts

    def _score_rollout(self, generation: dict, problem: dict) -> tuple[float, int]:
        """Decode + score a single generation without building the GRAIL
        proof. Returns ``(reward, completion_length)``.

        Used during the cold-start probe so we can decide whether to
        abort BEFORE paying for proof construction. The proof is built
        later only for groups that survive the probe gate.
        """
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        reward = self.env.compute_reward(problem, completion_text)
        return reward, len(completion_tokens)

    def _build_rollout_submission(self, generation, problem, randomness):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit."""
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        reward = self.env.compute_reward(problem, completion_text)

        commit = self._build_grail_commit(generation, randomness)
        return RolloutSubmission(
            tokens=all_tokens,
            reward=reward,
            commit=commit,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _randomness_for_window(
        self, subtensor, window_n: int, use_drand: bool
    ) -> str:
        """Return cached randomness for *window_n*, deriving it on first miss.

        The validator computes its per-window GRAIL seed from
        ``compute_window_randomness(block_hash(window_n), drand[...])`` and
        the miner must match bit-for-bit or every commitment fails. We
        cache the derived value so each window incurs at most one chain
        round-trip even though the outer polling loop ticks multiple times
        per window.
        """
        if self._cached_window_n == window_n and self._cached_randomness:
            return self._cached_randomness
        t = time.monotonic()
        randomness = await self._compute_randomness(subtensor, window_n, use_drand)
        chain_ms = (time.monotonic() - t) * 1000.0
        self._cached_window_n = window_n
        self._cached_randomness = randomness
        logger.info(
            "randomness derived window_n=%d randomness=%s... chain_ms=%.0f use_drand=%s",
            window_n, randomness[:16], chain_ms, use_drand,
        )
        return randomness

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool
    ) -> str:
        """Derive window randomness from block hash (+ optional drand beacon)."""
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

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Construct a GRAIL proof commit dict from a generation dict.

        Reproduces the proof construction:
          - HF forward pass for hidden_states + logits
          - Commitment batch via GRAILVerifier
          - log-softmax token log-probs
          - Signature via sign_commit_binding
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        # HF forward pass on proof GPU
        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]

        # Build commitments
        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        # fp32 log_softmax to match the validator and reduce tail-token drift.
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs: list[float] = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        # Sign
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
