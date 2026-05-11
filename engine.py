"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.

This evaluation contains three corrections over the upstream reference
miner that materially improve a miner's acceptance rate:

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
# to avoid log spam, low enough to be a useful pulse during debugging.
_SUMMARY_EVERY = 10


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
    # Local short-circuits (we declined to submit because we predicted
    # OUT_OF_ZONE from the rewards distribution).
    local_out_of_zone: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

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

    def summary(self) -> str:
        # Compact one-liner: "submitted=N accepted=A (rate%) rejected=R
        # local_oos=L net_err=E top=reason:count,reason:count,..."
        rate = (self.accepted / self.submitted * 100.0) if self.submitted else 0.0
        top = sorted(self.reasons.items(), key=lambda kv: -kv[1])[:4]
        top_str = ",".join(f"{r}:{c}" for r, c in top) or "-"
        return (
            f"submitted={self.submitted} accepted={self.accepted} "
            f"({rate:.1f}%) rejected={self.rejected} "
            f"local_oos={self.local_out_of_zone} net_err={self.network_errors} "
            f"top=[{top_str}]"
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


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    stats: _PromptStats | None = None,
    candidates: int = 16,
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
        self._prompt_stats = _PromptStats()

        # Rolling counters for the structured-summary log emitted every
        # ``_SUMMARY_EVERY`` submission attempts and on window transitions.
        self._metrics = _MinerMetrics()

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
                            last_window_n, state.window_n, self._metrics.summary(),
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

                t_gen = time.monotonic()
                generations = self._generate_m_rollouts(problem, randomness)
                gen_ms = (time.monotonic() - t_gen) * 1000.0
                if len(generations) < M_ROLLOUTS:
                    logger.warning(
                        "generated %d/%d for prompt %d (gen_ms=%.0f); skipping",
                        len(generations), M_ROLLOUTS, prompt_idx, gen_ms,
                    )
                    continue

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
                logger.info(
                    "rollouts ready window=%d prompt=%d k=%d/%d sigma=%.3f "
                    "in_zone=%s gen_ms=%.0f proof_ms=%.0f completion_len[min/med/max]=%d/%d/%d",
                    state.window_n, prompt_idx, k_solved, M_ROLLOUTS, sigma, in_zone,
                    gen_ms, proof_ms,
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
                    if (self._metrics.submitted + self._metrics.local_out_of_zone) % _SUMMARY_EVERY == 0:
                        logger.info("metrics | %s", self._metrics.summary())
                    continue

                merkle_root = _compute_merkle_root(rollout_submissions)

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
                    logger.info("metrics | %s", self._metrics.summary())

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

    def _generate_m_rollouts(self, problem, randomness) -> list[dict]:
        """Generate M_ROLLOUTS completions at T_PROTO in one batched call.

        One .generate() with batch shape (M_ROLLOUTS, prompt_len) is ~5-7×
        faster on GPU than M_ROLLOUTS serial calls — the matmul tiling
        utilizes far more of the GPU's compute. Each row samples
        independently (do_sample=True), so GRPO-group semantics are
        preserved. Each output row is truncated at its first post-prompt
        EOS so trailing batch-padding (which HF pads with pad_token_id =
        eos_token_id) is not carried downstream — otherwise the validator's
        GRAIL forward pass would see extra EOS tokens the miner didn't
        "generate" in the usual sense.
        """
        import torch

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * M_ROLLOUTS,
                device=getattr(self.vllm_model, "device", "cpu"),
            )
            outputs = self.vllm_model.generate(
                input_tensor,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=TOP_K_PROTO,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        eos = self.tokenizer.eos_token_id
        rollouts = []
        for i in range(M_ROLLOUTS):
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
