"""Miner engine — smart prompt selection + pre-generation pipeline.

This file is the only miner-side knob that decides how to play the GRPO
market. The protocol-fixed parts (T_PROTO=0.9, M_ROLLOUTS=8, GRAIL
sketches against state.randomness, checkpoint_hash gate, envelope sig
over the full request envelope) are kept exactly as the validator
expects — divergence here causes WRONG_RANDOMNESS / WRONG_CHECKPOINT /
GRAIL_FAIL / OUT_OF_ZONE rejects.

What's new vs. the reference engine:

  1. Smart prompt selection. Per-prompt Bayesian σ predictor with reset
     on checkpoint advance. The picker mixes exploit (highest predicted
     in-zone probability) with exploration of fresh prompts. Avoids
     prompts the local history says are 0/8 or 8/8 on the current
     checkpoint — those are guaranteed OUT_OF_ZONE.

  2. Pre-generation pipeline. Token generation (the 60-100s cost) is
     decoupled from window-randomness: rollouts are produced before
     ``state.randomness`` is known. As soon as the next window OPENs
     with non-empty randomness, we only pay for GRAIL sketch + sign +
     POST (~5-15s) — far below the reference engine's per-window
     critical path. PregenBatch carries (local_n, local_hash) tags; a
     batch is discarded if the checkpoint advances under it.

  3. In-zone pre-filter. σ is computed locally before sketching. If
     the rollout group is OUT_OF_ZONE we drop it without paying the
     GRAIL forward — frees the GPU to pregenerate the next candidate.

  4. Per-window submission budget tracker. Caps at
     ``MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW`` so we never burn quota
     on RATE_LIMITED rejects, and never resubmit the same prompt twice
     (would just race ourselves under the per-prompt split).
"""

from __future__ import annotations

import asyncio
import collections
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import random as _random

from reliquary.constants import (
    BOOTSTRAP_SIGMA_MIN,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    MAX_SUBMISSIONS_PER_PROMPT,
    M_ROLLOUTS,
    SIGMA_MIN,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
    UPLOAD_BUFFER,
    WINDOW_LENGTH,
)
from reliquary.infrastructure import chain
from reliquary.protocol.signatures import sign_envelope
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)
from reliquary.validator.dedup import compute_rollout_hash

if TYPE_CHECKING:
    from reliquary.environment.base import Environment
    from reliquary.miner.persistence import SupabaseCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for back-compat with tests + main.py wiring).
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
    from huggingface_hub import snapshot_download

    return await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["model.safetensors", "config.json", "tokenizer*"],
    )


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
) -> int:
    """Uniform-random rejection sampling against the cooldown set.

    Reference picker. ``MiningEngine.mine_window`` uses the smart picker
    below by default; this remains importable for tests and for any
    caller that wants the legacy behaviour.
    """
    rng = rng or _random
    n = len(env)
    if len(cooldown_prompts) < n / 2:
        for _ in range(max_attempts):
            idx = rng.randrange(n)
            if idx not in cooldown_prompts:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(n) if i not in cooldown_prompts]
    if not eligible:
        raise RuntimeError("no eligible prompt — env fully in cooldown")
    return rng.choice(eligible)


def _compute_merkle_root(rollouts) -> str:
    """Merkle root over rollout leaves — canonical JSON, returns 64-char hex."""
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


def _current_drand_round_at_send() -> int:
    """Drand quicknet round currently in progress at wall-clock now.

    Called just before POSTing /submit so the attached round matches
    what the validator sees at receipt (modulo the configured
    tolerance).

    Boundary-safety: constants.py specifically warns: "a miner firing
    at t=2.99s of round R would land at the validator at t=3.00s of
    R+1." With DRAND_ROUND_BACKWARD_TOLERANCE=0, that crossing
    produces a stale_round reject. Absorb up to ``safety_s`` of POST
    + validator-queue latency by sleeping past the next boundary
    whenever we're within that window of one. Tuned via
    RELIQUARY_DRAND_BOUNDARY_SAFETY_S; default 0.5 s.
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    genesis = float(ci["genesis_time"])
    period = float(ci["period"])
    try:
        safety_s = float(
            os.environ.get("RELIQUARY_DRAND_BOUNDARY_SAFETY_S", "0.5")
        )
    except ValueError:
        safety_s = 0.5
    now = time.time()
    # Seconds remaining in the current drand round. If under safety,
    # sleep past the boundary so the round we attach has a fresh
    # ~``period`` window before the next boundary.
    elapsed_in_round = (now - genesis) % period
    remaining = period - elapsed_in_round
    if remaining < safety_s:
        time.sleep(remaining + 0.05)
        now = time.time()
    return compute_current_drand_round(now, genesis, period)


# ---------------------------------------------------------------------------
# Smart picker — Bayesian σ predictor per prompt.
# ---------------------------------------------------------------------------


def _binom_pmf_sum(theta: float, k_low: int, k_high: int, n: int = M_ROLLOUTS) -> float:
    """P(k ∈ [k_low, k_high]) under Binomial(n, θ)."""
    theta = min(max(theta, 1e-6), 1.0 - 1e-6)
    p_in = 0.0
    for k in range(k_low, k_high + 1):
        p_in += math.comb(n, k) * (theta ** k) * ((1 - theta) ** (n - k))
    return p_in


def predict_in_zone(alpha: float, beta: float, *, bootstrap: bool = False) -> float:
    """Predicted P(σ ≥ threshold) for a prompt with Beta(α, β) success rate.

    For binary {0,1} rewards (MATH/OpenMathInstruct) and n=8, σ ≥ SIGMA_MIN
    is equivalent to k ∈ [2, 6]; bootstrap threshold widens to [1, 7].
    """
    theta = alpha / max(alpha + beta, 1e-9)
    if bootstrap:
        return _binom_pmf_sum(theta, 1, 7)
    return _binom_pmf_sum(theta, 2, 6)


def smart_pick_prompt(
    env,
    cooldown_set: set[int],
    submitted_this_window: set[int],
    sigma_alpha: dict[int, float],
    sigma_beta: dict[int, float],
    rng: _random.Random,
    *,
    bootstrap: bool = False,
    exploit_p: float = 0.6,
    explore_max_attempts: int = 200,
    intel_hot: set[int] | None = None,
    intel_oof: set[int] | None = None,
    intel_hot_bias: float = 0.85,
) -> int:
    """Pick a prompt_idx mixing exploit + R2-intel + explore.

    Hot-intel path (probability ``intel_hot_bias``, when ``intel_hot`` is
    non-empty): pick uniformly from prompts the R2 archive shows have
    landed in-zone (in ``batch[]`` or ``runners_up[]``) on recent
    windows. These are real, validator-verified σ ∈ [0.43, 0.55]
    candidates the model can produce — the strongest possible signal
    that the current policy is at the frontier on these prompts.
    Skips entries in cooldown/submitted/oof.

    Exploit path (probability ``exploit_p``, when we have local
    σ-history): sample from our top-K predicted-in-zone prompts.

    Explore path: random fresh prompt, with soft negative caches
    (local σ score < 0.15 or in ``intel_oof``).

    Never picks anything in ``cooldown_set`` (validator would reject
    PROMPT_IN_COOLDOWN) or in ``submitted_this_window`` (per-hotkey
    quota / K_p split with ourselves).
    """
    n = len(env)
    intel_hot = intel_hot or set()
    intel_oof = intel_oof or set()

    # 1) Hot-intel path: prefer R2-verified in-zone prompts.
    if intel_hot and rng.random() < intel_hot_bias:
        eligible = [
            pid for pid in intel_hot
            if pid not in cooldown_set
            and pid not in submitted_this_window
            and pid not in intel_oof
        ]
        if eligible:
            return rng.choice(eligible)

    # 2) Exploit path: local σ history.
    if sigma_alpha and rng.random() < exploit_p:
        scored: list[tuple[float, int]] = []
        for pid, a in sigma_alpha.items():
            if pid in cooldown_set or pid in submitted_this_window:
                continue
            if pid in intel_oof:
                continue
            b = sigma_beta.get(pid, 1.0)
            scored.append((predict_in_zone(a, b, bootstrap=bootstrap), pid))
        scored.sort(reverse=True)
        topk = [(s, p) for s, p in scored[:20] if s >= 0.30]
        if topk:
            weights = [s for s, _ in topk]
            return rng.choices([p for _, p in topk], weights=weights, k=1)[0]

    # 3) Explore path: random fresh prompt with soft negative cache.
    for _ in range(explore_max_attempts):
        idx = rng.randrange(n)
        if idx in cooldown_set or idx in submitted_this_window:
            continue
        if idx in intel_oof:
            continue
        if idx in sigma_alpha:
            score = predict_in_zone(
                sigma_alpha[idx], sigma_beta.get(idx, 1.0), bootstrap=bootstrap,
            )
            if score < 0.15:
                continue
        return idx

    # 4) Last resort: anything not in cooldown/submitted.
    for _ in range(explore_max_attempts):
        idx = rng.randrange(n)
        if idx not in cooldown_set and idx not in submitted_this_window:
            return idx
    raise RuntimeError("no eligible prompt after explore_max_attempts")


# ---------------------------------------------------------------------------
# Pregen pipeline state.
# ---------------------------------------------------------------------------


class PromptIntel:
    """R2 archive intelligence for the prompt picker.

    Periodically fetches the public dashboard endpoint
    ``https://www.reliqua.ai/api/r2/window/<N>`` for recent windows
    and builds two sets:

      * ``hot_prompts`` — prompts in ``batch[]`` or ``runners_up[]``
        across the lookback window. These were *validator-verified*
        as σ ≥ 0.43 by some miner on the current network. They are
        the strongest available signal that the current policy sits
        at the learning frontier on these specific prompts. Note:
        prompts in ``batch[]`` enter cooldown for 1M windows under
        the v2.3 BATCH_PROMPT_COOLDOWN_WINDOWS — those will be
        filtered by the picker's cooldown check. Prompts in
        ``runners_up[]`` are valid in-zone but lost the FIFO race;
        they remain pickable.

      * ``oof_prompts`` — prompts rejected with ``out_of_zone`` in
        recent windows. These were σ < 0.43 for someone else; high
        odds of also being out-of-zone for us under the same
        checkpoint. Add to the picker's negative cache.

    Resets on checkpoint advance (handled by the engine, since the
    intel only applies to the policy that produced the verdicts).
    """

    DASHBOARD_URL = "https://www.reliqua.ai/api/r2/window/{n}"

    def __init__(self) -> None:
        self.hot_prompts: set[int] = set()
        self.oof_prompts: set[int] = set()
        self._loaded_windows: set[int] = set()

    def reset(self) -> None:
        self.hot_prompts.clear()
        self.oof_prompts.clear()
        self._loaded_windows.clear()

    async def refresh(
        self,
        current_window_n: int,
        *,
        lookback: int = 50,
        per_request_timeout: float = 8.0,
    ) -> int:
        """Fetch any windows in [current - lookback, current) we haven't seen.

        Returns the count of windows newly loaded. Errors per-window
        are swallowed (the next refresh tick retries).
        """
        import httpx

        added = 0
        async with httpx.AsyncClient(timeout=per_request_timeout) as client:
            for w in range(max(0, current_window_n - lookback), current_window_n):
                if w in self._loaded_windows:
                    continue
                try:
                    r = await client.get(self.DASHBOARD_URL.format(n=w))
                    if r.status_code != 200:
                        continue
                    payload = r.json()
                    inner = payload.get("data") or {}
                    if not isinstance(inner, dict):
                        continue
                    for b in inner.get("batch") or ():
                        pid = b.get("prompt_idx")
                        if isinstance(pid, int):
                            self.hot_prompts.add(pid)
                    for ru in inner.get("runners_up") or ():
                        pid = ru.get("prompt_idx")
                        if isinstance(pid, int):
                            self.hot_prompts.add(pid)
                    for rj in inner.get("rejected") or ():
                        pid = rj.get("prompt_idx")
                        reason = rj.get("reason", "")
                        if isinstance(pid, int) and reason == "out_of_zone":
                            self.oof_prompts.add(pid)
                    self._loaded_windows.add(w)
                    added += 1
                except Exception:
                    pass
        return added


@dataclass
class PregenBatch:
    """A pre-generated rollout group awaiting GRAIL sketching + submit.

    Generated under (local_n, local_hash). If the checkpoint advances
    before we get to submit, the batch is dropped — the new validator
    forward pass would reject WRONG_CHECKPOINT (and the sketch wouldn't
    match the new weights anyway).

    ``sigma`` is the local population σ of the 8 rewards. We only ever
    sketch batches that already pass the in-zone gate; OUT_OF_ZONE
    pregens are dropped before paying for the GRAIL forward.

    ``submit_attempts`` counts how many times the submit worker has
    tried to POST this batch. Each hard-cap timeout / pre-flight abort
    bumps it; once it crosses a threshold the batch is dropped instead
    of re-queued, preventing infinite spin on a batch the validator
    refuses to ack within the cap.
    """

    prompt_idx: int
    problem: dict
    generations: list[dict]
    rewards: list[float]
    sigma: float
    completion_texts: list[str]
    local_n: int
    local_hash: str
    built_at: float = field(default_factory=time.time)
    submit_attempts: int = 0


# ---------------------------------------------------------------------------
# MiningEngine
# ---------------------------------------------------------------------------


class MiningEngine:
    """Smart miner with σ prediction + pre-generation pipeline.

    The class signature is preserved; what changed is the body of
    ``mine_window``. Reference-engine semantics are still reachable via
    ``pick_prompt_idx`` (module-level) for tests.
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
        pregen_capacity: int = 4,
        bootstrap: bool = False,
        prescreen_rollouts: int = 0,
        prescreen_max_tokens: int = 1024,
        gen_batch_prompts: int = 2,
        cache: "SupabaseCache | None" = None,
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

        # Smart-miner state.
        self._sigma_alpha: dict[int, float] = {}
        self._sigma_beta: dict[int, float] = {}
        self._sigma_ckpt_n: int = -1
        # Hard negative cache. Prompts the pre-screen condemned as
        # 0/N (model can't solve at any reasonable budget) or N/N
        # (model trivially nails it short-form → full M=8 will be 8/8
        # → OUT_OF_ZONE). Recording {3 successes / 0 failures} into
        # the Beta posterior leaves predicted in-zone at ~0.5 — high
        # enough that the picker re-selects the prompt and burns the
        # full 8 × 8192 gen on the same dud. The dud set is consulted
        # by the picker as a hard skip; cleared on checkpoint advance.
        self._prescreen_dud_set: set[int] = set()
        # Known-good prompts hydrated from Supabase: scraped from other
        # miners' accepted submissions (via scripts/scrape_intel.py) or
        # promoted by our own past full-gen ready events. The picker still
        # full-gens these to produce fresh, unique-hash rollouts, but
        # skips the ~22s prescreen since we already know they're in-zone
        # at this ckpt.
        self._known_good_prompts: set[int] = set()
        self._pregen_queue: collections.deque[PregenBatch] = collections.deque(
            maxlen=pregen_capacity,
        )
        self._submitted_this_window: set[int] = set()
        self._current_window_n: int = -1
        self._submission_count_this_window: int = 0
        self._bootstrap = bootstrap
        # Speculative pre-screen knobs. Set ``prescreen_rollouts > 0`` to
        # generate a cheap mini-batch BEFORE the full M_ROLLOUTS pregen,
        # using it as a noisy signal of P(in-zone). 0 = disabled (fall back
        # to direct full pregen, the safest behaviour for cold-start).
        self._prescreen_rollouts = max(0, int(prescreen_rollouts))
        self._prescreen_max_tokens = max(64, int(prescreen_max_tokens))
        # Multi-prompt batched generation. K prompts × M rollouts batched
        # into ONE .generate() call instead of K serial calls. The H100's
        # bandwidth-bound memory loads are amortised across a wider batch
        # → ~25-40 %% higher per-prompt throughput vs. single-prompt
        # batches. Defaults to 2 (safe on 80 GB H100 with Qwen3-4B at
        # max_new_tokens=8192). Bump higher if VRAM headroom allows.
        self._gen_batch_prompts = max(1, int(gen_batch_prompts))
        # R2 archive intelligence (public dashboard endpoint, no auth
        # needed). Background refresher fills ``hot_prompts`` /
        # ``oof_prompts`` from recent windows; the picker biases
        # toward hot + away from oof. Reset on checkpoint advance.
        self._intel = PromptIntel()
        self._intel_lookback = 50
        # Shared state across the concurrent state/pregen/submit workers.
        # ``asyncio`` is single-threaded — these are only touched at await
        # points, and only one task mutates each at a time.
        self._latest_state = None
        self._latest_local_n: int = 0
        self._latest_local_hash: str = ""
        # An asyncio.Event raised when a checkpoint advance lands. The
        # pregen worker checks it between gens to drop early instead of
        # running the model to completion on stale weights.
        self._ckpt_advance_event: asyncio.Event | None = None
        # Optional Supabase cache: hydrates dud_set + pregen_queue on
        # startup or ckpt advance; persists outcomes + tokens during run.
        # See reliquary.miner.persistence. None = disabled.
        self._cache = cache
        self._hydrated_ckpt_hashes: set[str] = set()
        # Loaded checkpoint cache (used by main.py's seed-skip path and
        # by ``_load_checkpoint`` to skip redundant reloads).
        self._loaded_checkpoint_path: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,
        use_drand: bool = True,
    ) -> list:
        """Concurrent smart mining loop.

        Three independent coroutines run forever, sharing state via
        ``self``:

          * ``_state_poll_loop`` — polls /state every ~1s, owns
            checkpoint-advance handling, exposes ``self._latest_state``.
            Cheap HTTP, never blocks the GPU.

          * ``_pregen_worker_loop`` — owns the GPU's generation pipeline.
            ALWAYS gen-bound: even when the queue is at capacity it
            evicts the oldest entry and refreshes it, so the GPU is
            never idle. Pre-screens cheaply if enabled, then runs the
            full M=8 × max_new_tokens generation. Pushes
            ``PregenBatch`` onto the queue.

          * ``_submit_worker_loop`` — waits for OPEN + non-empty
            randomness, drains the pregen queue into /submit. Sketch
            + sign + POST. Submit-quota and per-window dedup live here.

        Under PyTorch's per-device CUDA queue, when submit needs the
        proof model the in-flight generate() finishes first then the
        sketch forward runs — so submit latency during OPEN is bounded
        by ``in_flight_gen_remaining + ~1-3s sketch``. The architectural
        win is that the GPU never sits idle: between submits, it's
        ALWAYS generating fresh candidates. With single-GPU shared
        memory this is the most we can extract without splitting M=8
        across multiple calls.
        """
        import httpx
        import random as _r

        from reliquary.miner.submitter import discover_validator_url

        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        rng = _r.Random()
        self._ckpt_advance_event = asyncio.Event()

        # Submitter-only mode: when RELIQUARY_DISABLE_LOCAL_GEN=1, skip
        # spawning the GPU-bound pregen worker. The miner becomes a pure
        # submitter that drains whatever a sibling machine wrote into
        # Supabase pregen_batches (via scripts/prep_dataset.py). The
        # state poller still hydrates _pregen_queue on every ckpt
        # advance, so as long as the sibling fills the table the submit
        # worker has batches to ship.
        disable_local_gen = os.environ.get(
            "RELIQUARY_DISABLE_LOCAL_GEN", ""
        ).strip().lower() in ("1", "true", "yes")
        if disable_local_gen:
            logger.info(
                "RELIQUARY_DISABLE_LOCAL_GEN=1 — local pregen worker DISABLED; "
                "submitting only from Supabase-hydrated pregen_batches"
            )

        async with httpx.AsyncClient(timeout=30) as client:
            tasks = [
                asyncio.create_task(
                    self._state_poll_loop(url, client),
                    name="state_poll",
                ),
                asyncio.create_task(
                    self._submit_worker_loop(url, client),
                    name="submit_worker",
                ),
                asyncio.create_task(
                    self._intel_refresh_loop(),
                    name="intel_refresh",
                ),
            ]
            if self._cache is not None and self._cache.enabled:
                tasks.append(
                    asyncio.create_task(
                        self._cache_refresh_loop(),
                        name="cache_refresh",
                    )
                )
            if not disable_local_gen:
                tasks.append(
                    asyncio.create_task(
                        self._pregen_worker_loop(rng),
                        name="pregen_worker",
                    )
                )
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                raise
            except Exception:
                logger.exception("mine_window worker crashed")
                for t in tasks:
                    t.cancel()
                raise
        return []

    # ------------------------------------------------------------------
    # Concurrent workers
    # ------------------------------------------------------------------

    async def _state_poll_loop(self, url: str, client) -> None:
        """Poll /state and keep ``self._latest_state`` fresh.

        Owns checkpoint-advance handling: when ``state.checkpoint_n``
        moves, pulls the new HF revision, reloads both model copies,
        clears the σ history (the policy shifted; old rates are now
        misleading), and clears the pregen queue (stale-weight rollouts
        would fail GRAIL on the validator's new model). Sets
        ``_ckpt_advance_event`` so any in-flight worker can react.
        """
        from reliquary.constants import POLL_INTERVAL_SECONDS
        from reliquary.miner.submitter import (
            SubmissionError, get_window_state_v2,
        )

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

            try:
                prev_n = self._latest_local_n
                new_n, new_hash, new_hf = await maybe_pull_checkpoint(
                    state=state,
                    local_n=prev_n,
                    local_hash=self._latest_local_hash,
                    local_model=self.hf_model,
                    download_fn=_hf_download,
                    load_fn=self._load_checkpoint,
                )
                self._latest_local_n = new_n
                self._latest_local_hash = new_hash
                self.hf_model = new_hf
                if new_n != prev_n:
                    if self._pregen_queue:
                        logger.info(
                            "checkpoint advanced %d→%d; dropping %d pregen batches",
                            prev_n, new_n, len(self._pregen_queue),
                        )
                    self._pregen_queue.clear()
                    self._reset_sigma_on_ckpt(new_n)
                    # Known-good prompts are ckpt-bound — a prompt
                    # that was in-zone for the old policy may now be
                    # 8/8 trivial on the new one, and the intel_trusted
                    # override would then waste full-gen on it. Drop
                    # the in-memory set; the hydrator below will
                    # repopulate it from rows keyed by the new ckpt.
                    self._known_good_prompts.clear()
                    if self._ckpt_advance_event is not None:
                        self._ckpt_advance_event.set()
                # Cache hydration — runs once per ckpt_hash. Cheap when
                # the cache is disabled (early return inside the helper).
                await self._hydrate_from_cache(new_hash)
            except Exception:
                logger.exception("checkpoint pull failed; keeping local")

            if state.window_n != self._current_window_n:
                self._current_window_n = state.window_n
                self._submitted_this_window = set()
                self._submission_count_this_window = 0

            self._latest_state = state
            # 250 ms cadence keeps the ``_latest_state`` snapshot fresh
            # enough that the submit worker rarely fires into a sealed
            # window — the race window where we POST while the validator
            # is mid-rollover (and ``active_batcher`` either swapped or
            # transiently None) shrinks from ~1 s to ~250 ms. The /state
            # endpoint is cheap and lock-free server-side.
            await asyncio.sleep(0.25)

    async def _pregen_worker_loop(self, rng) -> None:
        """Never-idle generation worker.

        Generates fresh candidate batches in a tight loop. When the queue
        is full, evicts the oldest entry so the GPU keeps working on
        new content (older pregens have a higher chance of being stale
        w.r.t. a near-future checkpoint advance, plus rotating keeps
        prompt-cooldown-coverage fresh).
        """
        while True:
            state = self._latest_state
            if state is None or not self._latest_local_hash:
                await asyncio.sleep(0.5)
                continue
            # Capacity management: if full, peek the oldest entry. If
            # the oldest is stale (different ckpt) drop it; otherwise
            # wait briefly for submit_worker to drain rather than
            # overwriting a fresh in-zone batch.
            if len(self._pregen_queue) >= self._pregen_queue.maxlen:
                oldest = self._pregen_queue[0]
                if oldest.local_n != self._latest_local_n:
                    self._pregen_queue.popleft()
                else:
                    await asyncio.sleep(0.5)
                    continue
            cooldown_set = set(state.cooldown_prompts)
            await self._maybe_pregen_one(
                cooldown_set=cooldown_set,
                rng=rng,
                local_n=self._latest_local_n,
                local_hash=self._latest_local_hash,
            )

    async def _intel_refresh_loop(self) -> None:
        """Periodically refresh R2 prompt intelligence.

        Polls every 30 s. Does NOT reset on checkpoint advance —
        runners_up prompts from previous ckpts are still likely to be
        in-zone on the new ckpt (the policy moves by ~1 GRPO step per
        publish interval, a small shift). Throwing the intel away on
        every advance loses 1-2 minutes of refresh data and forces a
        cold-start picker until the refresher rebuilds.

        Stale prompts age out naturally: the refresher only fetches
        windows in [current - lookback, current); old prompts that
        aren't re-validated within the lookback window drop out of
        new fetches (though they linger in hot_prompts until the
        process restarts). For longer-running miners we could add a
        per-prompt last_seen_window with a TTL prune; for now the
        set grows slowly enough that this isn't needed.
        """
        while True:
            state = self._latest_state
            if state is None:
                await asyncio.sleep(5)
                continue
            try:
                added = await self._intel.refresh(
                    current_window_n=state.window_n,
                    lookback=self._intel_lookback,
                )
                if added:
                    logger.info(
                        "intel refresh +%d windows; hot=%d oof=%d",
                        added,
                        len(self._intel.hot_prompts),
                        len(self._intel.oof_prompts),
                    )
            except Exception:
                logger.exception("intel refresh failed")
            await asyncio.sleep(30)

    async def _submit_worker_loop(self, url: str, client) -> None:
        """Drain the pregen queue into /submit whenever OPEN.

        Pauses on non-OPEN states, empty randomness, exhausted per-window
        quota. Otherwise sketches + signs + POSTs each queued batch.
        """
        from reliquary.protocol.submission import WindowState

        while True:
            state = self._latest_state
            if state is None:
                await asyncio.sleep(0.2)
                continue
            if state.state != WindowState.OPEN or not state.randomness:
                await asyncio.sleep(0.2)
                continue
            if (
                self._submission_count_this_window
                >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
            ):
                await asyncio.sleep(0.5)
                continue
            if not self._pregen_queue:
                await asyncio.sleep(0.05)
                continue
            cooldown_set = set(state.cooldown_prompts)
            fired = await self._drain_pregens_to_submit(
                state=state,
                url=url,
                client=client,
                randomness=state.randomness,
                local_n=self._latest_local_n,
                local_hash=self._latest_local_hash,
                cooldown_set=cooldown_set,
                results=[],
            )
            if not fired:
                await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Pregen + submit helpers
    # ------------------------------------------------------------------

    async def _maybe_pregen_one(
        self,
        *,
        cooldown_set: set[int],
        rng: _random.Random,
        local_n: int,
        local_hash: str,
    ) -> None:
        """Pre-generate K prompts in one batched .generate() call.

        K = ``self._gen_batch_prompts``. With K=2 on a single H100, the
        full-gen .generate() runs shape (K × M, max_new_tokens) = (16,
        8192) instead of (8, 8192) — same model-weight loads per decode
        step, ~2× the useful work, ~25-40 %% better per-prompt throughput.

        For K=1 this collapses to the single-prompt path (identical
        behaviour to the pre-refactor code).

        Per-prompt steps:
          1. Smart picker selects K distinct candidates (never in
             cooldown / submitted-this-window / already-pregen / dud).
          2. Batched pre-screen: K × prescreen_rollouts in ONE call.
             Each prompt's k_short + truncation pattern is analysed
             independently; duds + boundary cases skip locally, the
             survivors carry through to step 3.
          3. Batched full gen: ``len(survivors) × M_ROLLOUTS`` in ONE
             call at the protocol cap.
          4. Per-prompt σ filter: drop OUT_OF_ZONE batches before
             paying for the GRAIL forward in the submit worker.

        Padding is left-aligned with attention_mask=0 on pad positions
        so the model never reads them; the stored ``tokens`` field
        contains only the REAL prompt + completion, matching the
        validator's canonical_prompt_tokens binding.
        """
        K = self._gen_batch_prompts
        already_pregen = {b.prompt_idx for b in self._pregen_queue}
        skip = (
            cooldown_set
            | self._submitted_this_window
            | already_pregen
            | self._prescreen_dud_set
        )

        # Pick K distinct candidates.
        prompt_idxs: list[int] = []
        for _ in range(K):
            try:
                pid = smart_pick_prompt(
                    self.env,
                    cooldown_set=skip | set(prompt_idxs),
                    submitted_this_window=set(),
                    sigma_alpha=self._sigma_alpha,
                    sigma_beta=self._sigma_beta,
                    rng=rng,
                    bootstrap=self._bootstrap,
                    intel_hot=self._intel.hot_prompts,
                    intel_oof=self._intel.oof_prompts,
                )
                prompt_idxs.append(pid)
            except RuntimeError as e:
                logger.debug("smart picker exhausted at %d/%d: %s", len(prompt_idxs), K, e)
                break
        if not prompt_idxs:
            return

        # Resolve problems; drop ones whose env lookup fails.
        problems_kept: list[dict] = []
        idxs_kept: list[int] = []
        for pid in prompt_idxs:
            try:
                problems_kept.append(self.env.get_problem(pid))
                idxs_kept.append(pid)
            except Exception:
                logger.exception("get_problem failed for idx=%d", pid)
        if not idxs_kept:
            return

        await self._pregen_batch_impl(
            prompt_idxs=idxs_kept,
            problems=problems_kept,
            local_n=local_n,
            local_hash=local_hash,
        )

    async def _pregen_batch_impl(
        self,
        *,
        prompt_idxs: list[int],
        problems: list[dict],
        local_n: int,
        local_hash: str,
    ) -> None:
        """Pre-screen (batched) + full gen (batched) + per-prompt σ filter.

        ``prompt_idxs[i]`` and ``problems[i]`` are parallel arrays. A
        prompt may be eliminated at the pre-screen step (set of skipped
        indices ``ps_skipped``) or at the post-full-gen σ step. Anything
        that survives is appended to the pregen queue.
        """
        # ---- Pre-screen (batched) -----------------------------------
        # Pre-screen everything without σ history except known duds.
        # Why pre-screen intel-hot too: the R2 signal can be stale
        # (the source miner solved it on an earlier ckpt; the policy
        # has since drifted toward solving it 8/8, making it
        # OUT_OF_ZONE for us now). Pre-screen at 512 tokens takes only
        # ~25-40 s for K=6 and surfaces those drift-stale cases
        # cheaply. Without pre-screen, a stale intel-hot prompt forces
        # the whole K=6 full-gen batch to wait ~5-15 min on a slow tail
        # rollout — net throughput is similar but the latency kills
        # race timing.
        ps_target_idx: list[int] = []
        for i, pid in enumerate(prompt_idxs):
            if pid in self._sigma_alpha:
                continue
            if pid in self._prescreen_dud_set:
                continue
            # Known-good prompts (scraped from R2 or promoted by our own
            # past pregen-ready) bypass the prescreen entirely: another
            # miner already verified this prompt is in-zone at this ckpt,
            # so the ~22s prescreen is dead weight. They still go through
            # full gen below to produce unique-hash rollouts.
            if pid in self._known_good_prompts:
                continue
            ps_target_idx.append(i)

        ps_skipped: set[int] = set()
        if self._prescreen_rollouts > 0 and ps_target_idx:
            t_pre = time.time()
            ps_problems = [problems[i] for i in ps_target_idx]
            try:
                ps_gens_per = await asyncio.to_thread(
                    self._generate_rollouts_multi_prompt,
                    ps_problems,
                    self._prescreen_rollouts,
                    self._prescreen_max_tokens,
                )
            except Exception:
                logger.exception(
                    "prescreen batched failed for %d prompts; falling through",
                    len(ps_problems),
                )
                ps_gens_per = []

            if ps_gens_per:
                pre_secs = time.time() - t_pre
                for slot, target_i in enumerate(ps_target_idx):
                    pid = prompt_idxs[target_i]
                    problem = problems[target_i]
                    short_gens = ps_gens_per[slot]
                    short_rewards: list[float] = []
                    for gen in short_gens:
                        text = self.tokenizer.decode(
                            gen["tokens"][gen["prompt_length"]:]
                        )
                        try:
                            short_rewards.append(
                                float(self.env.compute_reward(problem, text))
                            )
                        except Exception:
                            short_rewards.append(0.0)
                    k_short = sum(1 for r in short_rewards if r >= 0.5)
                    n_short = len(short_rewards)

                    # Truncation detector — see commentary in the
                    # single-prompt version of this block (now inlined).
                    wrong_indices = [
                        j for j, r in enumerate(short_rewards) if r < 0.5
                    ]
                    wrong_truncated = 0
                    for j in wrong_indices:
                        g = short_gens[j]
                        completion_len = len(g["tokens"]) - g["prompt_length"]
                        if completion_len >= self._prescreen_max_tokens - 4:
                            wrong_truncated += 1
                    trunc_ratio = (
                        wrong_truncated / len(wrong_indices)
                        if wrong_indices else 0.0
                    )

                    if k_short == 0 or k_short == n_short:
                        self._prescreen_dud_set.add(pid)
                        self._record_sigma(pid, short_rewards)
                        ps_skipped.add(target_i)
                        logger.info(
                            "pregen skip prescreen prompt=%d k=%d/%d pre=%.1fs "
                            "(dud_set_size=%d)",
                            pid, k_short, n_short, pre_secs,
                            len(self._prescreen_dud_set),
                        )
                        self._persist_outcome(
                            pid, k_short, 0.0,
                            "dud" if k_short == 0 else "oof",
                            avg_completion_len=None,
                            truncated_count=None,
                        )
                        continue

                    # Average completion length across the pre-screen
                    # rollouts. Fast-EOS prompts (model commits and
                    # terminates with EOS) average well below
                    # prescreen_max_tokens. Slow-tail prompts (model
                    # rambles) average near the cap. The latter cost
                    # disproportionate GRAIL time on the validator,
                    # pushing us into later drand-round buckets at
                    # seal time — even when σ is in zone, slow prompts
                    # tend to land as runners_up rather than batch
                    # winners.
                    avg_completion_len = sum(
                        len(g["tokens"]) - g["prompt_length"]
                        for g in short_gens
                    ) / max(1, len(short_gens))
                    SLOW_EOS_RATIO = 0.60   # > 60 %% of cap = slow
                    slow_eos = avg_completion_len >= (
                        self._prescreen_max_tokens * SLOW_EOS_RATIO
                    )

                    low_boundary = (k_short <= 1) and (trunc_ratio >= 0.75)
                    high_boundary = (
                        k_short >= n_short - 1
                        and len(wrong_indices) > 0
                        and wrong_truncated == len(wrong_indices)
                    )
                    # All-truncated-wrongs rule. Empirical observation
                    # on Qwen3-4B at ckpt 150+: when k_short ≥ 1 AND
                    # EVERY wrong rollout is a truncation (no \boxed),
                    # the prompt almost always resolves to 8/8 at full
                    # length — the "wrong" rollouts are simply slower
                    # correct ones that hadn't reached the boxed answer
                    # in 512 tokens. Path C cases keep at least one
                    # wrong rollout with an explicit (short, EOS-
                    # terminated) wrong answer, so they slip past this
                    # rule and commit. Requires ≥2 truncated wrongs so
                    # a single-wrong outlier doesn't trigger the skip.
                    all_wrongs_truncated = (
                        len(wrong_indices) >= 2
                        and wrong_truncated == len(wrong_indices)
                        and k_short >= 1
                    )
                    # Intel override: if this prompt is in the R2-hot
                    # set OR persisted as known-good in Supabase under
                    # this ckpt, some miner (us or a sibling) already
                    # produced an in-zone batch on it. Trust the
                    # cross-checkpoint signal — commit to full gen even
                    # if all_wrongs_truncated would otherwise skip.
                    # Promotes _known_good_prompts from a prescreen
                    # bypass (which doesn't apply here — we're already
                    # inside the prescreen branch on a fresh pid) into a
                    # heuristic override for borderline cases.
                    intel_trusted = (
                        pid in self._intel.hot_prompts
                        or pid in self._known_good_prompts
                    )
                    skip_decision = (
                        low_boundary
                        or high_boundary
                        or (all_wrongs_truncated and not intel_trusted)
                    )
                    if skip_decision:
                        self._prescreen_dud_set.add(pid)
                        self._record_sigma(pid, short_rewards)
                        ps_skipped.add(target_i)
                        logger.info(
                            "pregen skip prescreen prompt=%d k=%d/%d pre=%.1fs "
                            "trunc_wrong=%d/%d (likely 8/8 at full → OUT_OF_ZONE) "
                            "(dud_set_size=%d)",
                            pid, k_short, n_short, pre_secs,
                            wrong_truncated, len(wrong_indices),
                            len(self._prescreen_dud_set),
                        )
                        self._persist_outcome(
                            pid, k_short, 0.0, "dud",
                            avg_completion_len=int(avg_completion_len),
                            truncated_count=int(wrong_truncated),
                        )
                        continue

                    logger.info(
                        "prescreen pass prompt=%d k=%d/%d pre=%.1fs "
                        "trunc_wrong=%d/%d — committing full gen",
                        pid, k_short, n_short, pre_secs,
                        wrong_truncated, len(wrong_indices),
                    )

        # ---- Full gen (batched) -------------------------------------
        survivors_idx = [
            i for i in range(len(prompt_idxs)) if i not in ps_skipped
        ]
        if not survivors_idx:
            return
        survivor_problems = [problems[i] for i in survivors_idx]
        survivor_pids = [prompt_idxs[i] for i in survivors_idx]

        t0 = time.time()
        try:
            full_gens_per = await asyncio.to_thread(
                self._generate_rollouts_multi_prompt,
                survivor_problems,
                M_ROLLOUTS,
                self.max_new_tokens,
            )
        except Exception:
            logger.exception(
                "full gen batched failed for %d prompts", len(survivor_problems),
            )
            return
        gen_secs = time.time() - t0

        # Stale-ckpt bailout — applies to the entire batch since gen ran
        # under a single checkpoint snapshot.
        if (
            self._ckpt_advance_event is not None
            and self._ckpt_advance_event.is_set()
            and self._latest_local_n != local_n
        ):
            logger.info(
                "pregen drop stale-ckpt (gen ran across %d→%d) on %d prompts",
                local_n, self._latest_local_n, len(survivor_pids),
            )
            self._ckpt_advance_event.clear()
            return

        for slot, pid in enumerate(survivor_pids):
            problem = survivor_problems[slot]
            generations = full_gens_per[slot] if slot < len(full_gens_per) else []
            if len(generations) < M_ROLLOUTS:
                logger.warning(
                    "generated %d/%d for prompt %d; skipping",
                    len(generations), M_ROLLOUTS, pid,
                )
                continue

            completion_texts: list[str] = []
            rewards: list[float] = []
            for gen in generations:
                text = self.tokenizer.decode(
                    gen["tokens"][gen["prompt_length"]:]
                )
                completion_texts.append(text)
                try:
                    rewards.append(
                        float(self.env.compute_reward(problem, text))
                    )
                except Exception:
                    rewards.append(0.0)

            self._record_sigma(pid, rewards)
            sigma = self._population_std(rewards)
            k_correct = int(sum(1 for r in rewards if r >= 0.5))
            threshold = BOOTSTRAP_SIGMA_MIN if self._bootstrap else SIGMA_MIN
            in_zone = sigma >= threshold
            # Upstream PR #54 (Merge ed21e90) removed the k ∈ [3,5] binary
            # frontier filter — steady-state acceptance now widens to
            # k ∈ [2, 6] (natural σ ≥ 0.43 band) and bootstrap stays at
            # k ∈ [1, 7] (BOOTSTRAP_SIGMA_MIN = 0.33). The explicit
            # BINARY_REWARD_MIN/MAX_CORRECT gate that lived here is gone;
            # the σ check above is now the only zone gate.
            if not in_zone:
                # Pin OUT_OF_ZONE prompts in the dud set so the picker
                # never wastes another full gen on them — including
                # after checkpoint advances (the policy rarely flips
                # a true 0/8 or 8/8 prompt into the [2,6] zone in a
                # single training step). The pre-screen guard already
                # filters obvious duds; this catches the false-positive
                # commits (pre-screen looked split, full was binary).
                self._prescreen_dud_set.add(pid)
                logger.info(
                    "pregen reject OUT_OF_ZONE prompt=%d k=%d/%d sigma=%.3f gen=%.1fs",
                    pid, k_correct, M_ROLLOUTS, sigma, gen_secs,
                )
                self._persist_outcome(
                    pid, k_correct, float(sigma), "oof",
                )
                continue

            batch = PregenBatch(
                prompt_idx=pid,
                problem=problem,
                generations=generations,
                rewards=rewards,
                sigma=sigma,
                completion_texts=completion_texts,
                local_n=local_n,
                local_hash=local_hash,
            )
            self._pregen_queue.append(batch)
            logger.info(
                "pregen ready prompt=%d k=%d/%d sigma=%.3f gen=%.1fs queue=%d",
                pid, int(sum(rewards)), M_ROLLOUTS, sigma, gen_secs,
                len(self._pregen_queue),
            )
            self._persist_outcome(
                pid, int(sum(rewards)), float(sigma), "good",
            )
            self._persist_batch(batch)

    def _generate_n_rollouts_short(
        self, problem: dict, n_rollouts: int, max_new_tokens: int,
    ) -> list[dict]:
        """Single-prompt convenience wrapper around the multi-prompt path."""
        return self._generate_rollouts_multi_prompt(
            [problem], n_rollouts, max_new_tokens,
        )[0]

    def _generate_rollouts_multi_prompt(
        self,
        problems: list[dict],
        n_rollouts: int,
        max_new_tokens: int,
    ) -> list[list[dict]]:
        """Batched generation across K distinct prompts.

        Builds a single ``(K × n_rollouts, max_prompt_len)`` tensor with
        LEFT-padding (causal LM convention), runs ONE ``.generate()``
        call, then splits the output back per-prompt and strips the
        padding so the returned rollout ``tokens`` are the real
        sequences the validator will tokenise to.

        Why this matters: autoregressive decoding on the H100 is HBM-
        bandwidth-bound. Each decode step loads the 8 GB of model
        weights from HBM once, regardless of batch width. Doubling the
        batch (K=2 prompts × M=8 rollouts = 16 sequences) amortises
        that load over twice the useful work — same wall time, ~2× the
        rollouts produced. Net effect: ~25-40 %% higher per-prompt
        throughput than running K serial single-prompt calls.

        Padding considerations:
        - Different prompts have different lengths. We left-pad each
          row to ``max_prompt_len`` with ``pad_token_id``; the
          attention_mask zeroes out padding positions so the model's
          attention never reads them.
        - The validator's GRAIL forward is ALWAYS single-row, no pad.
          Our miner stores the REAL (un-padded) prompt + completion in
          ``tokens``; the GRAIL forward is also single-row on the
          proof GPU. So padding here is purely a gen-time efficiency
          trick — it never leaks into the submitted commits.
        """
        import torch

        K = len(problems)
        if K == 0:
            return []

        prompt_token_lists = [
            self.tokenizer.encode(p["prompt"], add_special_tokens=False)
            for p in problems
        ]
        prompt_lengths = [len(t) for t in prompt_token_lists]

        # Resolve the EOS id set ONCE — both branches need it. Qwen3-Instruct
        # ships generation_config.eos_token_id = [151645, 151643] AND
        # pad_token_id = 151643, so trimming on a single eos id misses
        # rows that stopped on the other.
        gen_cfg = getattr(self.hf_model, "generation_config", None)
        _eos_ids = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        if _eos_ids is None:
            _eos_ids = self.tokenizer.eos_token_id
        if isinstance(_eos_ids, int):
            eos_set = {_eos_ids}
        elif _eos_ids is None:
            eos_set = set()
        else:
            eos_set = {int(e) for e in _eos_ids if e is not None}

        # vLLM branch: when self.vllm_model is a vllm.LLM instance use
        # its native batched-with-n API. ~2-3x faster than HF.generate
        # because of PagedAttention + continuous batching. The HF sketch
        # path downstream still runs on self.hf_model so GRAIL verifies
        # against the same numerics the validator uses.
        try:
            from vllm import LLM as _VllmLLM, SamplingParams as _SP
            is_vllm = isinstance(self.vllm_model, _VllmLLM)
        except Exception:
            is_vllm = False

        if is_vllm:
            # HF p_stop threshold for the regen filter. validator floor is
            # 0.005 and the validator does the SAME HF forward we do (same
            # model, same kernels) — no inter-HF drift, so threshold == floor
            # is correct. Earlier 1.5x margin was over-cautious and rejected
            # too many borderline rollouts; regen couldn't catch up in 2
            # attempts, batches fell back to dirty, preverify-skipped.
            HF_P_STOP_FLOOR = 0.005

            def _process_completion(completion, prompt_ids, prompt_length):
                """Append EOS for natural terminations; return (entry, vllm_clean).

                vllm_clean means: vLLM thought this finished on a stop token
                AND the last token is now in eos_set after our re-append.
                Does NOT guarantee HF preverify will pass — that's checked
                by ``_compute_hf_p_stops`` downstream.
                """
                gen_tokens = list(completion.token_ids)
                finish = getattr(completion, "finish_reason", None)
                stop = getattr(completion, "stop_reason", None)
                # vLLM v1 strips stop tokens from token_ids when it
                # terminates on an EOS-set id. Validator requires
                # last_token ∈ eos_set AND p_stop ≥ floor at the
                # pre-last position — re-append the stop token when
                # finish_reason=="stop" so natural terminations are
                # actually submittable.
                if eos_set and (not gen_tokens or int(gen_tokens[-1]) not in eos_set):
                    if finish == "stop" and isinstance(stop, int) and stop in eos_set:
                        gen_tokens.append(stop)
                vllm_clean = (
                    finish == "stop"
                    and bool(gen_tokens)
                    and int(gen_tokens[-1]) in eos_set
                )
                return {
                    "tokens": prompt_ids + gen_tokens,
                    "prompt_length": prompt_length,
                }, vllm_clean

            def _hf_classify(completions, prompt_ids, prompt_length,
                             k_idx, clean_results, dirty_results):
                """Run vLLM-gate + HF p_stop classification on completions
                and bucket into clean_results[k_idx] / dirty_results[k_idx].

                Only candidates that pass BOTH vLLM gates (finish_reason=="stop"
                + EOS last token) AND HF p_stop >= HF_P_STOP_FLOOR end up in
                clean. The rest go to dirty for the regen loop. Avoids
                spending an HF forward on vLLM-already-dirty rollouts.
                """
                vllm_clean_entries: list[dict] = []
                for completion in completions:
                    entry, vllm_clean = _process_completion(
                        completion, prompt_ids, prompt_length,
                    )
                    if vllm_clean:
                        vllm_clean_entries.append(entry)
                    else:
                        dirty_results[k_idx].append(entry)
                if not vllm_clean_entries:
                    return
                t_hf = time.time()
                try:
                    hf_p_stops = self._compute_hf_p_stops(
                        vllm_clean_entries, eos_set,
                    )
                except Exception:
                    logger.exception(
                        "HF p_stop check failed; treating vLLM-clean as clean",
                    )
                    clean_results[k_idx].extend(vllm_clean_entries)
                    return
                hf_clean_count = 0
                for entry, p in zip(vllm_clean_entries, hf_p_stops):
                    if p >= HF_P_STOP_FLOOR:
                        clean_results[k_idx].append(entry)
                        hf_clean_count += 1
                    else:
                        dirty_results[k_idx].append(entry)
                logger.info(
                    "HF p_stop check k=%d: %d/%d kept (took %.1fs) p_stops=%s",
                    k_idx, hf_clean_count, len(vllm_clean_entries),
                    time.time() - t_hf,
                    [round(p, 4) for p in hf_p_stops],
                )

            sampling = _SP(
                n=n_rollouts,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=TOP_K_PROTO,
                max_tokens=max_new_tokens,
            )
            # vLLM accepts {"prompt_token_ids": [...]} dicts directly —
            # avoids any re-tokenization roundtrip.
            vllm_prompts = [
                {"prompt_token_ids": tokens} for tokens in prompt_token_lists
            ]
            outputs = self.vllm_model.generate(
                vllm_prompts, sampling, use_tqdm=False,
            )
            # Two buckets per prompt: clean (HF p_stop ≥ HF_P_STOP_FLOOR AND
            # ends with EOS) and dirty (everything else). vLLM-clean alone is
            # insufficient because vLLM↔HF drift on FA2 produces bimodal
            # p_stops at HF (some vLLM-clean rollouts have HF p_stop ≈ 0).
            # ``_hf_classify`` runs the HF forward once per completion set
            # and re-buckets accordingly. Validator only checks submitted
            # rollouts; how the miner produced them (rejection sampling) is
            # unconstrained.
            clean_results: list[list[dict]] = [[] for _ in range(K)]
            dirty_results: list[list[dict]] = [[] for _ in range(K)]
            for k in range(K):
                req_out = outputs[k]
                prompt_ids = list(req_out.prompt_token_ids)
                prompt_length = len(prompt_ids)
                _hf_classify(
                    req_out.outputs, prompt_ids, prompt_length,
                    k, clean_results, dirty_results,
                )

            # Regen loop: after PR #54 the validator accepts up to
            # MAX_TRUNCATED_PER_SUBMISSION=5 dirty rollouts per batch. So
            # we only need (n_rollouts - 5) = 3 HF-clean rollouts to land
            # an acceptable submission — the remaining 5 slots can come
            # from the dirty bucket. Most batches now satisfy this on
            # initial gen and skip regen entirely. We also drop the second
            # regen attempt: with the budget of 5 the cost/benefit of
            # spending another ~60-200 s of full gen on the residual gap
            # rarely pays off vs just submitting and moving on.
            DIRTY_BUDGET = 5
            MIN_CLEAN_NEEDED = max(0, n_rollouts - DIRTY_BUDGET)
            MAX_REGEN_ATTEMPTS = 1
            regen_oversample = 2
            for attempt in range(MAX_REGEN_ATTEMPTS):
                short_indices = [
                    k for k in range(K) if len(clean_results[k]) < MIN_CLEAN_NEEDED
                ]
                if not short_indices:
                    break
                t_regen_start = time.time()
                for k_idx in short_indices:
                    needed = MIN_CLEAN_NEEDED - len(clean_results[k_idx])
                    if needed <= 0:
                        continue
                    regen_n = max(1, needed * regen_oversample)
                    regen_sampling = _SP(
                        n=regen_n,
                        temperature=T_PROTO,
                        top_p=TOP_P_PROTO,
                        top_k=TOP_K_PROTO,
                        max_tokens=max_new_tokens,
                    )
                    regen_outputs = self.vllm_model.generate(
                        [vllm_prompts[k_idx]], regen_sampling, use_tqdm=False,
                    )
                    if not regen_outputs:
                        continue
                    regen_req = regen_outputs[0]
                    prompt_ids = list(regen_req.prompt_token_ids)
                    prompt_length = len(prompt_ids)
                    _hf_classify(
                        regen_req.outputs, prompt_ids, prompt_length,
                        k_idx, clean_results, dirty_results,
                    )
                logger.info(
                    "regen attempt %d/%d: %d prompts below clean floor=%d, took %.1fs "
                    "(clean counts: %s)",
                    attempt + 1, MAX_REGEN_ATTEMPTS, len(short_indices),
                    MIN_CLEAN_NEEDED,
                    time.time() - t_regen_start,
                    [len(clean_results[k]) for k in range(K)],
                )

            # Final assembly: fill any remaining slots with dirty rollouts
            # (better to submit and let preverify reject than to skip the
            # batch entirely — preverify-skip wastes the gen cost; sometimes
            # the validator's bf16 numerics put a borderline dirty rollout
            # back above the floor).
            results: list[list[dict]] = [[] for _ in range(K)]
            for k in range(K):
                results[k].extend(clean_results[k][:n_rollouts])
                while len(results[k]) < n_rollouts and dirty_results[k]:
                    results[k].append(dirty_results[k].pop(0))
                # Last resort: duplicate the last entry if everything was empty.
                while len(results[k]) < n_rollouts and results[k]:
                    results[k].append(results[k][-1])
            return results

        max_prompt_len = max(prompt_lengths)
        pad_id = self.tokenizer.pad_token_id

        # Build flat (K × n_rollouts, max_prompt_len) left-padded input.
        flat_input: list[list[int]] = []
        flat_mask: list[list[int]] = []
        for tokens, length in zip(prompt_token_lists, prompt_lengths):
            pad_count = max_prompt_len - length
            padded = [pad_id] * pad_count + tokens
            mask = [0] * pad_count + [1] * length
            for _ in range(n_rollouts):
                flat_input.append(padded)
                flat_mask.append(mask)

        device = getattr(self.vllm_model, "device", "cpu")
        input_tensor = torch.tensor(flat_input, device=device)
        mask_tensor = torch.tensor(flat_mask, device=device)

        with torch.no_grad():
            outputs = self.vllm_model.generate(
                input_tensor,
                attention_mask=mask_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=TOP_K_PROTO,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Trim at first EOS so trailing batch-pad EOS tokens (HF pads finished
        # rows with pad_token_id) don't leak into the submission. eos_set was
        # hoisted above the vLLM branch so both paths share it.
        results: list[list[dict]] = [[] for _ in range(K)]
        for k in range(K):
            prompt_length = prompt_lengths[k]
            start_idx = max_prompt_len - prompt_length
            for r in range(n_rollouts):
                row_idx = k * n_rollouts + r
                full_seq = outputs[row_idx].tolist()
                # Strip the LEFT pad: real content starts at start_idx.
                real_seq = full_seq[start_idx:]
                prompt_part = real_seq[:prompt_length]
                gen_part = real_seq[prompt_length:]
                if eos_set:
                    first_eos = next(
                        (i for i, t in enumerate(gen_part) if int(t) in eos_set),
                        None,
                    )
                    if first_eos is not None:
                        gen_part = gen_part[: first_eos + 1]
                results[k].append({
                    "tokens": prompt_part + gen_part,
                    "prompt_length": prompt_length,
                })
        return results

    async def _drain_pregens_to_submit(
        self,
        *,
        state,
        url: str,
        client,
        randomness: str,
        local_n: int,
        local_hash: str,
        cooldown_set: set[int],
        results: list,
    ) -> bool:
        """Sketch + sign + POST every queued pregen batch that's still valid.

        Returns True if at least one submission was fired. A batch is
        dropped (not submitted) if:
          * its prompt_idx is now in cooldown (validator added it
            mid-window via another miner's seal)
          * we already submitted on this prompt this window
          * its (local_n, local_hash) has gone stale relative to the
            engine's current loaded checkpoint
          * adding it would exceed the per-hotkey-per-window cap

        Each surviving batch is sketched against ``state.randomness``,
        signed under the envelope binding the validator verifies, and
        POSTed to /submit.
        """
        from reliquary.miner.submitter import SubmissionError, submit_batch_v2
        from reliquary.protocol.submission import BatchSubmissionRequest

        any_fired = False
        # Iterate over a snapshot — we mutate the deque as we go.
        snapshot = list(self._pregen_queue)
        self._pregen_queue.clear()

        for batch in snapshot:
            if self._submission_count_this_window >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
                # Quota done for this window: requeue the rest unchanged
                # so they survive the window rollover IF the checkpoint
                # hasn't advanced (which would be caught above).
                if (
                    batch.local_n == local_n
                    and batch.local_hash == local_hash
                ):
                    if len(self._pregen_queue) < self._pregen_queue.maxlen:
                        self._pregen_queue.append(batch)
                continue
            if batch.local_n != local_n or batch.local_hash != local_hash:
                logger.info(
                    "dropping stale pregen prompt=%d (ckpt %d→%d)",
                    batch.prompt_idx, batch.local_n, local_n,
                )
                continue
            if batch.prompt_idx in cooldown_set:
                logger.info(
                    "dropping pregen prompt=%d (entered cooldown)", batch.prompt_idx,
                )
                continue
            if batch.prompt_idx in self._submitted_this_window:
                logger.info(
                    "dropping pregen prompt=%d (already submitted this window)",
                    batch.prompt_idx,
                )
                continue

            # Pre-flight HASH_DUPLICATE check. A submission is rejected
            # whole if ANY rollout in it collides with the validator's
            # accepted hash_set, so even one stale-cached rollout poisons
            # the entire batch. Check against the hashes scrape_intel
            # mirrored from R2 windows for this (prompt, ckpt). Hashing
            # M=8 token sequences locally is ~µs; saves the ~1-5s sketch
            # + GRAIL forward + network round-trip a guaranteed-loss
            # submit would burn.
            #
            # Dormant on this device until persistence.py also gets the
            # ``accepted_hashes_for_prompt`` method + the SQL table. The
            # ``hasattr`` guard keeps log noise low (no per-submit
            # AttributeError tracebacks) and the check becomes a no-op
            # until the cache side lands.
            if (
                self._cache is not None
                and self._cache.enabled
                and hasattr(self._cache, "accepted_hashes_for_prompt")
            ):
                try:
                    accepted_set = await asyncio.to_thread(
                        self._cache.accepted_hashes_for_prompt,
                        batch.prompt_idx, local_hash,
                    )
                except Exception:
                    logger.exception(
                        "accepted_hashes lookup failed prompt=%d", batch.prompt_idx,
                    )
                    accepted_set = set()
                if accepted_set:
                    collided = sum(
                        1 for gen in batch.generations
                        if compute_rollout_hash(gen["tokens"]).hex() in accepted_set
                    )
                    if collided > 0:
                        logger.info(
                            "dropping pregen prompt=%d "
                            "(%d/%d rollouts collide with accepted hash_set)",
                            batch.prompt_idx, collided, len(batch.generations),
                        )
                        continue

            t_sketch_start = time.time()
            try:
                # Batched sketch: one GRAIL forward over all M
                # rollouts instead of M sequential forwards. Cuts the
                # ``sketch`` component of our submit time from ~5 s
                # to ~1 s, which is the difference between landing
                # before vs after the validator's seal under load.
                commits, p_stops = self._build_grail_commits_batched(
                    batch.generations, randomness,
                )
                rollout_submissions = [
                    RolloutSubmission(
                        tokens=gen["tokens"],
                        reward=reward,
                        commit=commit,
                    )
                    for gen, reward, commit in zip(
                        batch.generations, batch.rewards, commits,
                    )
                ]
            except Exception:
                logger.exception("sketching failed prompt=%d", batch.prompt_idx)
                continue

            # Preverify: check every cheap validator gate locally and
            # skip the batch if any rollout would fail. Stops the
            # quota-bleed pattern where bad_termination POSTs burn
            # one of our 8 per-window slots each, leaving no room
            # for clean batches that arrive later in the window.
            # Configurable floor via RELIQUARY_PSTOP_MIN; 0.005 is
            # half the validator's MIN_EOS_PROBABILITY=0.01 — looser
            # than safe-margin to let cross-stack variance through.
            try:
                p_stop_floor = float(
                    os.environ.get("RELIQUARY_PSTOP_MIN", "0.005")
                )
            except ValueError:
                p_stop_floor = 0.005
            preverify_fail = self._preverify_batch(
                rollout_submissions, p_stops, p_stop_floor,
            )
            if preverify_fail:
                self._prescreen_dud_set.add(batch.prompt_idx)
                # Mark Supabase row consumed so the next cache refresh
                # doesn't re-hydrate this dud — without this the batch
                # cycles forever between in-memory drop and cache pull.
                self._persist_consumed(batch.prompt_idx, batch.local_hash)
                logger.info(
                    "skip submit prompt=%d (preverify=%s p_stops=%s)",
                    batch.prompt_idx, preverify_fail,
                    [round(p, 4) for p in p_stops],
                )
                continue
            merkle_root = _compute_merkle_root(rollout_submissions)

            current_round = _current_drand_round_at_send()
            nonce = os.urandom(16).hex()
            envelope_sig = sign_envelope(
                wallet=self.wallet,
                miner_hotkey=self.wallet.hotkey.ss58_address,
                window_start=state.window_n,
                prompt_idx=batch.prompt_idx,
                merkle_root=merkle_root,
                checkpoint_hash=local_hash,
                drand_round=current_round,
                randomness=randomness,
                nonce=nonce,
            ).hex()
            request = BatchSubmissionRequest(
                miner_hotkey=self.wallet.hotkey.ss58_address,
                prompt_idx=batch.prompt_idx,
                window_start=state.window_n,
                merkle_root=merkle_root,
                rollouts=rollout_submissions,
                checkpoint_hash=local_hash,
                drand_round=current_round,
                nonce=nonce,
                envelope_signature=envelope_sig,
            )
            sketch_secs = time.time() - t_sketch_start

            # Pre-flight: if the latest state already shifted off OPEN
            # or to a different window between when we picked up the
            # pregen and now, abort — the POST would just earn
            # WINDOW_NOT_ACTIVE or WINDOW_MISMATCH and waste the
            # round-trip. Re-queue the batch for the next OPEN, but
            # cap attempts so we don't infinite-spin on a batch the
            # validator never accepts within the hard cap.
            #
            # The state field check is required even when window_n
            # and randomness still match: the validator transitions
            # OPEN -> TRAINING -> PUBLISHING within a single window_n
            # at the seal trigger. Without this guard the POST lands
            # in TRAINING and earns window_not_active.
            from reliquary.protocol.submission import WindowState as _WS
            latest = self._latest_state
            if (
                latest is None
                or latest.state != _WS.OPEN
                or latest.window_n != state.window_n
                or not latest.randomness
                or latest.randomness != randomness
            ):
                batch.submit_attempts += 1
                MAX_SUBMIT_ATTEMPTS = 3
                if (
                    batch.submit_attempts < MAX_SUBMIT_ATTEMPTS
                    and batch.local_n == local_n
                    and batch.local_hash == local_hash
                    and len(self._pregen_queue) < self._pregen_queue.maxlen
                ):
                    self._pregen_queue.append(batch)
                    logger.info(
                        "submit aborted pre-flight prompt=%d "
                        "(state shifted, attempt %d/%d, re-queued)",
                        batch.prompt_idx, batch.submit_attempts, MAX_SUBMIT_ATTEMPTS,
                    )
                else:
                    logger.info(
                        "submit aborted pre-flight prompt=%d "
                        "(state shifted, attempt %d/%d, dropped)",
                        batch.prompt_idx, batch.submit_attempts, MAX_SUBMIT_ATTEMPTS,
                    )
                continue

            try:
                t_post = time.time()
                # Hard cap on submit duration. ``submit_batch_v2``'s
                # internal retry can stack up to 3 attempts (10 s
                # timeout each + 1+2+4 backoff = ~37 s worst case).
                # With validator response under load reaching 14-20 s
                # per attempt, retries push our total submit time past
                # the window cadence — by the time we get a verdict,
                # the window has sealed (window_not_active) or rolled
                # (bad_envelope_signature on the now-stale randomness).
                # asyncio.wait_for caps total wall time regardless of
                # retries. On timeout we re-queue the batch so the
                # next OPEN gets another shot.
                resp = await asyncio.wait_for(
                    submit_batch_v2(
                        url, request, client=client, timeout=3.0,
                    ),
                    timeout=4.0,
                )
                post_secs = time.time() - t_post
                reason_value = (
                    resp.reason.value if hasattr(resp.reason, "value") else resp.reason
                )
                # bad_envelope_signature and window_not_active are
                # transient race losses on the validator side that do
                # NOT bump the validator's per-window counter (per
                # MAX_BAD_ENVELOPE_PER_HOTKEY_PER_WINDOW design and
                # the pre-batcher reject path). Mirror that: don't
                # bump our local counter either, and re-queue the
                # batch for the next OPEN — the rollouts are still
                # valid for the same ckpt.
                transient_race = (
                    not resp.accepted
                    and reason_value in ("bad_envelope_signature", "window_not_active")
                )
                if transient_race:
                    batch.submit_attempts += 1
                    MAX_RETRY_ATTEMPTS = 3
                    if (
                        batch.submit_attempts < MAX_RETRY_ATTEMPTS
                        and batch.local_n == local_n
                        and batch.local_hash == local_hash
                        and len(self._pregen_queue) < self._pregen_queue.maxlen
                    ):
                        self._pregen_queue.append(batch)
                    logger.info(
                        "submit race prompt=%d window=%d reason=%s "
                        "(attempt %d/%d, %s)",
                        batch.prompt_idx, state.window_n, reason_value,
                        batch.submit_attempts, MAX_RETRY_ATTEMPTS,
                        "re-queued" if batch.submit_attempts < MAX_RETRY_ATTEMPTS else "dropped",
                    )
                    continue
                self._submitted_this_window.add(batch.prompt_idx)
                self._submission_count_this_window += 1
                any_fired = True
                # If the validator says we're rate-limited, our local
                # counter is behind the validator's — usually because
                # earlier POSTs timed out client-side but were still
                # counted server-side. Sync forward immediately so we
                # stop firing into a closed quota window. The local
                # counter is reset to 0 at every window roll.
                if not resp.accepted and reason_value == "rate_limited":
                    self._submission_count_this_window = (
                        MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
                    )
                logger.info(
                    "submitted window=%d prompt=%d sigma=%.3f sketch=%.2fs post=%.2fs "
                    "accepted=%s reason=%s",
                    state.window_n, batch.prompt_idx, batch.sigma,
                    sketch_secs, post_secs,
                    resp.accepted,
                    resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
                )
                results.append(resp)
                self._persist_consumed(batch.prompt_idx, batch.local_hash)
            except asyncio.TimeoutError:
                # The hard cap fired. Re-queue the batch (if still
                # ckpt-current and there's room) so the next OPEN
                # gets a fresh shot — sign + envelope will be rebuilt
                # at that time against the new randomness. Capped at
                # MAX_SUBMIT_ATTEMPTS to avoid infinite spin when the
                # validator is persistently overloaded and never
                # responds within the cap.
                batch.submit_attempts += 1
                MAX_SUBMIT_ATTEMPTS = 3
                if (
                    batch.submit_attempts < MAX_SUBMIT_ATTEMPTS
                    and batch.local_n == local_n
                    and batch.local_hash == local_hash
                    and len(self._pregen_queue) < self._pregen_queue.maxlen
                ):
                    self._pregen_queue.append(batch)
                    logger.warning(
                        "submit timeout prompt=%d after hard cap "
                        "(attempt %d/%d, re-queued)",
                        batch.prompt_idx, batch.submit_attempts, MAX_SUBMIT_ATTEMPTS,
                    )
                else:
                    logger.warning(
                        "submit timeout prompt=%d after hard cap "
                        "(attempt %d/%d, DROPPED — validator likely overloaded)",
                        batch.prompt_idx, batch.submit_attempts, MAX_SUBMIT_ATTEMPTS,
                    )
            except SubmissionError as exc:
                logger.error("submit failed prompt=%d: %s", batch.prompt_idx, exc)

        return any_fired

    # ------------------------------------------------------------------
    # Supabase cache (optional)
    # ------------------------------------------------------------------

    async def _cache_refresh_loop(self) -> None:
        """Periodically re-pull Supabase rows for the active ckpt so a
        sibling prep machine's new writes land in this miner without
        waiting for the next ckpt advance. 60s cadence — Supabase reads
        are cheap and idempotent (the hydrator skips already-known
        prompts/batches). Calls hydrate with force=True to bypass the
        per-ckpt-hash guard that the state poller relies on.
        """
        REFRESH_INTERVAL = 60.0
        while True:
            await asyncio.sleep(REFRESH_INTERVAL)
            try:
                ckpt_hash = self._latest_local_hash
                if ckpt_hash:
                    await self._hydrate_from_cache(ckpt_hash, force=True)
            except Exception:
                logger.exception("cache refresh loop iteration failed")

    async def _hydrate_from_cache(self, ckpt_hash: str, *, force: bool = False) -> None:
        """Pull dud_set + known_good + pregen_queue for ``ckpt_hash``
        from Supabase. Idempotent for the same hash unless ``force=True``
        (used by the periodic refresh loop). Called from the state
        poller on every iteration BUT the idempotency guard means we
        only actually hit Supabase the first time we see a hash.

        All Supabase I/O is pushed to a thread so the asyncio loop
        stays responsive.
        """
        if self._cache is None or not self._cache.enabled:
            return
        if not ckpt_hash:
            return
        first_time = ckpt_hash not in self._hydrated_ckpt_hashes
        if not first_time and not force:
            return
        self._hydrated_ckpt_hashes.add(ckpt_hash)

        try:
            outcomes = await asyncio.to_thread(
                self._cache.load_outcomes, ckpt_hash,
            )
        except Exception:
            logger.exception("hydrate outcomes failed ckpt=%s", ckpt_hash)
            outcomes = []
        added_duds = 0
        added_good = 0
        for o in outcomes:
            if o.status in ("dud", "oof"):
                if o.prompt_idx not in self._prescreen_dud_set:
                    self._prescreen_dud_set.add(o.prompt_idx)
                    added_duds += 1
            elif o.status == "good":
                if o.prompt_idx not in self._known_good_prompts:
                    self._known_good_prompts.add(o.prompt_idx)
                    added_good += 1

        try:
            cached_batches = await asyncio.to_thread(
                self._cache.load_unconsumed_batches, ckpt_hash,
            )
        except Exception:
            logger.exception("hydrate batches failed ckpt=%s", ckpt_hash)
            cached_batches = []
        added_batches = 0
        already = {b.prompt_idx for b in self._pregen_queue}
        for pb in cached_batches:
            if pb.prompt_idx in already:
                continue
            if pb.prompt_idx in self._prescreen_dud_set:
                continue
            if len(self._pregen_queue) >= self._pregen_queue.maxlen:
                break
            try:
                problem = self.env.get_problem(pb.prompt_idx)
            except Exception:
                continue
            generations = []
            rewards = []
            completion_texts = []
            for r in pb.rollouts:
                generations.append({
                    "tokens": list(r.get("tokens") or []),
                    "prompt_length": int(r.get("prompt_length", 0)),
                })
                rewards.append(float(r.get("reward", 0.0)))
                gen_tokens = generations[-1]["tokens"][generations[-1]["prompt_length"]:]
                try:
                    completion_texts.append(self.tokenizer.decode(gen_tokens))
                except Exception:
                    completion_texts.append("")
            batch = PregenBatch(
                prompt_idx=pb.prompt_idx,
                problem=problem,
                generations=generations,
                rewards=rewards,
                sigma=float(pb.sigma),
                completion_texts=completion_texts,
                # Stamp the batch with the CURRENT local_n, not whatever
                # value the writer (sibling prep machine, our own past
                # run) recorded. ckpt_hash is the authoritative ckpt
                # identifier; local_n is a redundant int that the
                # in-memory stale-check uses. A mismatched local_n with
                # matching ckpt_hash would otherwise create an infinite
                # drop+rehydrate loop (the queue drops "stale" batches,
                # the refresh loop re-pulls them seconds later).
                local_n=self._latest_local_n,
                local_hash=ckpt_hash,
            )
            self._pregen_queue.append(batch)
            already.add(pb.prompt_idx)
            added_batches += 1
        if first_time or added_duds or added_batches or added_good:
            logger.info(
                "cache %s ckpt=%s duds+=%d good+=%d batches+=%d "
                "(dud_set=%d known_good=%d queue=%d)",
                "hydrated" if first_time else "refreshed",
                ckpt_hash[:16], added_duds, added_good, added_batches,
                len(self._prescreen_dud_set),
                len(self._known_good_prompts),
                len(self._pregen_queue),
            )

    def _persist_outcome(
        self, prompt_idx: int, k: int, sigma: float, status: str,
        *, avg_completion_len: int | None = None,
        truncated_count: int | None = None,
    ) -> None:
        """Fire-and-forget outcome write. Schedules a background task on
        the running asyncio loop so the caller never blocks on network.
        Safe to call from any async context; a no-op when cache disabled.
        """
        if self._cache is None or not self._cache.enabled:
            return
        ckpt_hash = self._latest_local_hash
        if not ckpt_hash:
            return
        from reliquary.miner.persistence import PromptOutcome
        o = PromptOutcome(
            prompt_idx=int(prompt_idx),
            checkpoint_hash=ckpt_hash,
            k=int(k),
            sigma=float(sigma),
            status=status,
            avg_completion_len=avg_completion_len,
            truncated_count=truncated_count,
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._cache.upsert_outcome, o))
        except RuntimeError:
            pass

    def _persist_batch(self, batch: "PregenBatch") -> None:
        """Fire-and-forget pregen-batch write."""
        if self._cache is None or not self._cache.enabled:
            return
        ckpt_hash = batch.local_hash or self._latest_local_hash
        if not ckpt_hash:
            return
        from reliquary.miner.persistence import PersistedBatch
        rollouts = []
        for gen, reward in zip(batch.generations, batch.rewards):
            rollouts.append({
                "tokens": [int(t) for t in (gen.get("tokens") or [])],
                "prompt_length": int(gen.get("prompt_length", 0)),
                "reward": float(reward),
            })
        pb = PersistedBatch(
            prompt_idx=int(batch.prompt_idx),
            checkpoint_hash=ckpt_hash,
            local_n=int(batch.local_n),
            sigma=float(batch.sigma),
            k=int(sum(1 for r in batch.rewards if r >= 0.5)),
            rollouts=rollouts,
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._cache.save_batch, pb))
        except RuntimeError:
            pass

    def _persist_consumed(self, prompt_idx: int, ckpt_hash: str) -> None:
        if self._cache is None or not self._cache.enabled:
            return
        if not ckpt_hash:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                asyncio.to_thread(
                    self._cache.mark_consumed, int(prompt_idx), ckpt_hash,
                )
            )
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # σ predictor
    # ------------------------------------------------------------------

    def _reset_sigma_on_ckpt(self, n: int) -> None:
        if n != self._sigma_ckpt_n:
            # The Bayesian σ posterior is updated against the OLD
            # policy's rewards; the new policy can shift them, so
            # reset α/β.
            self._sigma_alpha.clear()
            self._sigma_beta.clear()
            # The dud set is INTENTIONALLY kept across checkpoint
            # advances: prompts the pre-screen identified as true
            # 0/8 (model can't solve) or true 8/8 (model nails it
            # trivially) are rarely flipped to in-zone by a single
            # GRPO training step. Keeping the cache spares us from
            # re-paying ~50-400s of full gen on the same dud each
            # time the policy advances — the most expensive bug the
            # original ckpt-reset behaviour caused.
            self._sigma_ckpt_n = n

    def _record_sigma(self, prompt_idx: int, rewards: list[float]) -> None:
        """Update Beta posterior with this observation.

        Treat binary {0, 1} rewards as Bernoulli successes; for
        continuous-reward envs we'd map by thresholding at 0.5.
        """
        n_correct = sum(1 for r in rewards if r >= 0.5)
        n_wrong = len(rewards) - n_correct
        a = self._sigma_alpha.get(prompt_idx, 1.0) + n_correct
        b = self._sigma_beta.get(prompt_idx, 1.0) + n_wrong
        self._sigma_alpha[prompt_idx] = a
        self._sigma_beta[prompt_idx] = b

    @staticmethod
    def _population_std(rewards: list[float]) -> float:
        n = len(rewards)
        if n < 2:
            return 0.0
        mean = sum(rewards) / n
        var = sum((r - mean) ** 2 for r in rewards) / n
        return math.sqrt(var)

    # ------------------------------------------------------------------
    # Model + generation
    # ------------------------------------------------------------------

    def _load_checkpoint(self, local_path: str):
        """Reload model(s) from *local_path*.

        Three branches:
        - vLLM gen + HF sketch: rebuild the vllm.LLM (no in-place hot-
          swap exists in vLLM) AND reload hf_model in-place.
        - Shared HF: one model backs both; load once.
        - Two-copy HF: reload both, propagate carefully.

        Either way, ``_loaded_checkpoint_path`` is set so subsequent
        calls with the same path are no-ops.
        """
        import gc
        import torch
        from transformers import AutoModelForCausalLM

        from reliquary.constants import ATTN_IMPLEMENTATION

        if self._loaded_checkpoint_path == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        # Detect vLLM gen engine. Import lazily so a vLLM-less venv can
        # still import this module.
        is_vllm_gen = False
        try:
            from vllm import LLM as _VllmLLM
            is_vllm_gen = isinstance(self.vllm_model, _VllmLLM)
        except Exception:
            pass

        if is_vllm_gen:
            # vLLM 0.20 uses a multiprocessing engine worker. In-process
            # del + recreate is unreliable: the worker subprocess may
            # not release CUDA cleanly, and the next LLM() init crashes
            # with "EngineCore failed to start". The supported recipe is
            # to spawn a fresh process. We use os.execv() to replace
            # the current Python process image with an identical
            # invocation — CUDA + multiproc state is fully reset, and
            # the new process re-runs cli/main, which now sees the new
            # ckpt at startup and loads it from scratch.
            import os as _os
            import sys as _sys
            logger.warning(
                "vLLM ckpt advance detected (path=%s); execv()-ing self to "
                "rebuild engine cleanly. PID stays; argv preserved.",
                local_path,
            )
            # Tear down vLLM's subprocess worker BEFORE execv: execv
            # replaces our Python image but does NOT signal forked
            # children. A surviving VLLM::EngineCore subprocess keeps
            # ~50 GB of VRAM locked, starving the new vLLM init.
            try:
                del self.vllm_model
            except Exception:
                pass
            self.vllm_model = None
            try:
                import subprocess as _sp
                _sp.run(
                    ["pkill", "-9", "-f", "VLLM::EngineCore"],
                    check=False, timeout=5,
                )
            except Exception:
                pass
            try:
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass
            try:
                _os.execv(_sys.executable, [_sys.executable] + _sys.argv)
            except Exception:
                logger.exception(
                    "execv failed; exiting so the supervisor (or systemd / "
                    "launch_miner.sh restart loop) can respawn us.",
                )
                _os._exit(42)

        logger.info("Loading checkpoint from %s", local_path)
        shared = self.vllm_model is self.hf_model

        if shared:
            try:
                new_model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=ATTN_IMPLEMENTATION,
                ).to(f"cuda:{self.vllm_gpu}").eval()
            except Exception:
                logger.exception(
                    "Failed to reload shared model from %s; keeping old",
                    local_path,
                )
                return self.hf_model
            old = self.hf_model
            self.hf_model = new_model
            self.vllm_model = new_model
            del old
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._loaded_checkpoint_path = local_path
            logger.info("Checkpoint %s loaded into shared model", local_path)
            return self.hf_model

        # Two-copy path (legacy / multi-GPU setups).
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

    def _generate_m_rollouts(self, problem, randomness) -> list[dict]:
        """Single-prompt M=8 rollouts at T_PROTO.

        Convenience wrapper around ``_generate_rollouts_multi_prompt``.
        ``randomness`` is signature-compat only — sampling is fully
        determined by T_PROTO/top_p/top_k and the GRAIL randomness
        enters at sketch-build time, never at gen time. That decoupling
        is what lets pregen run before the window's randomness lands.
        """
        return self._generate_rollouts_multi_prompt(
            [problem], M_ROLLOUTS, self.max_new_tokens,
        )[0]

    def _build_rollout_submission(self, generation, problem, randomness):
        """Build a RolloutSubmission (sketch + sign) from a generation dict.

        Reference path used when callers haven't pre-scored. The smart
        loop uses ``_build_rollout_submission_from_gen`` directly to
        skip the redundant reward call.
        """
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        reward = float(self.env.compute_reward(problem, completion_text))
        return self._build_rollout_submission_from_gen(
            gen=generation,
            problem=problem,
            randomness=randomness,
            completion_text=completion_text,
            reward=reward,
        )

    def _build_rollout_submission_from_gen(
        self,
        *,
        gen: dict,
        problem: dict,
        randomness: str,
        completion_text: str,
        reward: float,
    ) -> RolloutSubmission:
        """Build a RolloutSubmission for a pre-scored generation.

        Splits out from ``_build_rollout_submission`` so the smart loop
        can re-use cached reward + completion_text from the pregen
        batch instead of paying ``env.compute_reward`` twice.
        """
        commit = self._build_grail_commit(gen, randomness)
        return RolloutSubmission(
            tokens=gen["tokens"],
            reward=reward,
            commit=commit,
        )

    # ------------------------------------------------------------------
    # Private GRAIL helpers
    # ------------------------------------------------------------------

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool,
    ) -> str:
        """Legacy: derive window randomness locally.

        Kept for back-compat with any caller wiring outside the main
        loop. The smart loop reads ``state.randomness`` directly off
        /state and never calls this.
        """
        if use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"],
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                None, beacon["randomness"], drand_round=beacon["round"],
            )
        block_hash = await chain.get_block_hash(subtensor, window_start)
        return chain.compute_window_randomness(block_hash)

    def _preverify_batch(
        self,
        rollout_submissions: list,
        p_stops: list[float],
        p_stop_floor: float,
    ) -> str:
        """Return empty string if the batch would pass the validator's
        per-rollout gates, else a short failure reason for logging.

        Post PR #54 the validator runs a TRUNCATION BUDGET over
        ``verify_termination``: any rollout that fails (cap_truncated
        OR low_p_stop OR non_eos_last) increments a counter; the batch
        is rejected only when the counter exceeds
        MAX_TRUNCATED_PER_SUBMISSION = 5 (BOOTSTRAP_... in bootstrap
        windows). Mirror that here — count failures, reject only when
        over budget. has_eos_padding remains a hard per-rollout fail
        (rollout-level corruption, not termination tolerance).
        """
        from reliquary.constants import (
            MAX_TRUNCATED_PER_SUBMISSION,
            BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION,
        )
        budget = (
            BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION
            if self._bootstrap
            else MAX_TRUNCATED_PER_SUBMISSION
        )
        gen_cfg = getattr(self.hf_model, "generation_config", None)
        eos_ids = (
            getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        )
        if eos_ids is None:
            eos_ids = getattr(self.tokenizer, "eos_token_id", None)
        if eos_ids is None:
            return ""
        if isinstance(eos_ids, int):
            eos_set = {int(eos_ids)}
        else:
            eos_set = {int(e) for e in eos_ids if e is not None}

        truncated_count = 0
        first_trunc_reason = ""
        for i, rs in enumerate(rollout_submissions):
            commit = rs.commit
            tokens = commit.get("tokens") or []
            meta = commit.get("rollout", {}) or {}
            prompt_length = int(meta.get("prompt_length", 0))
            completion_length = int(meta.get("completion_length", 0))
            total = prompt_length + completion_length
            if total < 2 or completion_length <= 0 or not tokens:
                return f"empty(idx={i})"
            last_tok = int(tokens[-1])
            cap_hit = total >= MAX_NEW_TOKENS_PROTOCOL_CAP
            p_stop = float(p_stops[i]) if i < len(p_stops) else 0.0
            # Classify this rollout's termination outcome. The validator's
            # batcher counts ANY rollout that fails verify_termination OR
            # is_cap_truncation against the same per-submission budget.
            rollout_trunc_reason = ""
            if cap_hit:
                if last_tok not in eos_set or p_stop < p_stop_floor:
                    rollout_trunc_reason = f"cap_truncation(idx={i}, p_stop={p_stop:.4f})"
            else:
                if last_tok not in eos_set:
                    rollout_trunc_reason = f"non_eos_last(idx={i}, last={last_tok})"
                elif p_stop < p_stop_floor:
                    rollout_trunc_reason = (
                        f"low_p_stop(idx={i}, p_stop={p_stop:.4f}, floor={p_stop_floor:.4f})"
                    )
            if rollout_trunc_reason:
                truncated_count += 1
                if not first_trunc_reason:
                    first_trunc_reason = rollout_trunc_reason
                if truncated_count > budget:
                    return (
                        f"over_budget({truncated_count}/{budget+1}+) first={first_trunc_reason}"
                    )
            # has_eos_padding: completion must contain EXACTLY one EOS,
            # at the final position. Multiple EOS or EOS-not-last is
            # rollout corruption, not termination tolerance — hard fail
            # outside the truncation budget.
            completion = tokens[prompt_length: prompt_length + completion_length]
            eos_positions = [
                j for j, tok in enumerate(completion) if int(tok) in eos_set
            ]
            if eos_positions and (
                len(eos_positions) > 1
                or eos_positions[0] != len(completion) - 1
            ):
                return (
                    f"eos_padding(idx={i}, n_eos={len(eos_positions)}, "
                    f"first_pos={eos_positions[0]}, comp_len={len(completion)})"
                )

            # OMI boxed-answer tamper guard (PR #54). Every token inside the
            # last \boxed{...} content must have chosen-token probability
            # ≥ BOXED_ANSWER_MIN_PROB (0.5). Hard fail — outside the
            # truncation budget. We have the chosen-token log-probs in the
            # commit's rollout dict (from _build_grail_commits_batched);
            # exp them only on the boxed range to keep this cheap.
            try:
                from reliquary.validator.verifier import _find_last_boxed_token_range
                from reliquary.constants import BOXED_ANSWER_MIN_PROB
            except Exception:
                # Older validator without the guard — skip cleanly.
                continue
            token_logprobs = meta.get("token_logprobs") or []
            if not token_logprobs or len(token_logprobs) < completion_length:
                continue
            rng = _find_last_boxed_token_range(completion, self.tokenizer)
            if rng is None:
                continue
            start, end = rng
            import math
            probs_boxed = [
                math.exp(float(token_logprobs[j]))
                for j in range(start, end + 1)
                if 0 <= j < len(token_logprobs)
            ]
            if probs_boxed:
                min_prob = min(probs_boxed)
                if min_prob < BOXED_ANSWER_MIN_PROB:
                    return (
                        f"boxed_low_prob(idx={i}, min={min_prob:.3f}, "
                        f"floor={BOXED_ANSWER_MIN_PROB:.3f}, n={len(probs_boxed)})"
                    )
        return ""

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Single-rollout convenience wrapper around the batched path."""
        commits, _ = self._build_grail_commits_batched([generation], randomness)
        return commits[0]

    def _compute_hf_p_stops(
        self, generations: list[dict], eos_set: set[int],
    ) -> list[float]:
        """Run HF forward on rollout token sequences and return per-rollout
        p_stop. p_stop[i] = sum of softmax(logits at position n_i-2)[eos_ids].

        Same formula the validator uses in _gpu_p_stop and our preverify uses
        downstream. Called from the regen filter inside ``_generate_full`` so
        we can drop vLLM-clean rollouts that HF disagrees on (the bimodal
        drift case where vLLM samples EOS at positions HF treats as p≈0) and
        regenerate via vLLM. Without this, regen-on-cap-truncation only
        catches half of the failures.
        """
        import torch
        if not generations or not eos_set:
            return [0.0] * len(generations)

        pad_id = self.tokenizer.pad_token_id
        token_lists = [g["tokens"] for g in generations]
        max_len = max(len(t) for t in token_lists)

        flat_input: list[list[int]] = []
        for tokens in token_lists:
            pad_count = max_len - len(tokens)
            flat_input.append(tokens + [pad_id] * pad_count)

        device = f"cuda:{self.proof_gpu}"
        input_tensor = torch.tensor(flat_input, device=device)
        with torch.no_grad():
            out = self.hf_model(input_ids=input_tensor)
        logits_batch = out.logits  # [batch, max_len, vocab]

        eos_tensor = torch.tensor(
            sorted(eos_set), device=device, dtype=torch.long,
        )
        p_stops: list[float] = []
        for i, tokens in enumerate(token_lists):
            n = len(tokens)
            if n < 2:
                p_stops.append(0.0)
                continue
            probs = torch.softmax(logits_batch[i, n - 2, :].float(), dim=-1)
            p_stop = probs.index_select(-1, eos_tensor).sum().item()
            p_stops.append(float(p_stop))

        return p_stops

    def _build_grail_commits_batched(
        self, generations: list[dict], randomness: str,
    ) -> tuple[list[dict], list[float]]:
        """Build M GRAIL commits in ONE forward pass.

        Previously each of M=8 rollouts ran its own batch=1 forward —
        8 sequential GPU calls, ~4-5 s total wall time on H100. This
        was the dominant component of submit latency (post=1.5s vs
        sketch=5s), pushing our submissions past the validator's window
        seal even when the network was responsive.

        Bit-equivalence with the validator's per-rollout forward:
        right-pad to the longest rollout, leave attention_mask=None.
        Causal attention can ONLY see prior positions, so non-pad
        positions never read pad tokens — every real position's
        hidden state is bit-identical to a single-row forward of the
        same row. The validator's forward in verify_commitment_proofs
        also passes attention_mask=None, so we match its numerics
        exactly.
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        if not generations:
            return [], []

        pad_id = self.tokenizer.pad_token_id
        token_lists = [g["tokens"] for g in generations]
        prompt_lengths = [g["prompt_length"] for g in generations]
        max_len = max(len(t) for t in token_lists)

        # Right-pad. Pad tokens go AFTER real tokens; causal mask makes
        # them invisible to non-pad outputs.
        flat_input: list[list[int]] = []
        for tokens in token_lists:
            pad_count = max_len - len(tokens)
            flat_input.append(tokens + [pad_id] * pad_count)

        device = f"cuda:{self.proof_gpu}"
        input_tensor = torch.tensor(flat_input, device=device)
        with torch.no_grad():
            hidden_states_batch, logits_batch = forward_single_layer(
                self.hf_model, input_tensor, None, LAYER_INDEX,
            )

        r_vec = self._verifier.generate_r_vec(randomness)
        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")

        # Resolve EOS ids once for the inline p_stop computation per
        # rollout — same logic the validator uses in _gpu_p_stop, but
        # against our HF forward instead of the validator's. We mirror
        # the validator's check so preverify can reject rollouts that
        # would fail termination at the validator BEFORE we burn
        # quota with a doomed POST.
        import torch
        gen_cfg = getattr(self.hf_model, "generation_config", None)
        eos_ids_cfg = (
            getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        )
        if eos_ids_cfg is None:
            eos_ids_cfg = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_ids_cfg, int):
            eos_id_list = [int(eos_ids_cfg)]
        elif eos_ids_cfg is None:
            eos_id_list = []
        else:
            eos_id_list = [int(e) for e in eos_ids_cfg if e is not None]
        eos_id_tensor = (
            torch.tensor(
                eos_id_list, device=logits_batch.device, dtype=torch.long,
            )
            if eos_id_list
            else None
        )

        commits: list[dict] = []
        p_stops: list[float] = []
        for i, tokens in enumerate(token_lists):
            n_tokens = len(tokens)
            prompt_length = prompt_lengths[i]

            # Slice off the pad portion before computing sketches —
            # pad-position hidden states are bit-equivalent to a
            # single-row forward's, but we don't want to include them
            # in the commitments anyway.
            h = hidden_states_batch[i, :n_tokens, :]
            commitments = self._verifier.create_commitments_batch(h, r_vec)

            # Memory-efficient log_softmax: we only need the log-prob
            # of ONE token per position (the actually-sampled token at
            # the next position), but a naive ``torch.log_softmax(...)``
            # over [n_tokens, vocab=151936] in float32 allocates
            # ~5 GiB per rollout — sufficient to OOM the GPU when
            # vLLM is co-resident. The identity
            #
            #     log_softmax(x)[t] = x[t] - logsumexp(x, dim=-1)
            #
            # lets us compute only the n_tokens scalars we need:
            # gather the target-token logits along the vocab dim,
            # logsumexp per row, subtract. Output is shape [n_tokens]
            # instead of [n_tokens, vocab] — ~150k× smaller.
            logits_i = logits_batch[i, :n_tokens, :].float()
            # Target token IDs at positions 1..n_tokens (predictions
            # of logits rows 0..n_tokens-1). We only need positions
            # prompt_length..n_tokens for the completion logprobs.
            tokens_t = torch.tensor(tokens, device=logits_i.device, dtype=torch.long)
            # Per-row logsumexp normaliser.
            log_norm = torch.logsumexp(logits_i, dim=-1)  # [n_tokens]
            # Gather the logit for each row's "next token" — i.e.
            # for row r the logit at index tokens[r+1]. We compute it
            # only for r in [prompt_length-1, n_tokens-1], which is
            # exactly the range used downstream.
            row_idx = torch.arange(
                max(prompt_length - 1, 0), n_tokens - 1,
                device=logits_i.device,
            )
            tok_idx = tokens_t[row_idx + 1]
            gathered = logits_i[row_idx, tok_idx]  # [completion_length]
            row_log_probs = gathered - log_norm[row_idx]
            token_logprobs: list[float] = row_log_probs.detach().cpu().tolist()

            # p_stop: raw softmax probability of any EOS token at the
            # position predicting the LAST token. Same formula as
            # validator's _gpu_p_stop. Computed inline from the
            # already-loaded logits_batch — zero extra GPU cost.
            if eos_id_tensor is not None and n_tokens >= 2:
                probs_last = torch.softmax(
                    logits_batch[i, n_tokens - 2, :].float(), dim=-1,
                )
                p_stop = float(
                    probs_last.index_select(-1, eos_id_tensor).sum().item()
                )
            else:
                p_stop = 0.0
            p_stops.append(p_stop)

            signature = sign_commit_binding(
                tokens, randomness, model_name, LAYER_INDEX,
                commitments, self.wallet,
            )
            commits.append({
                "tokens": tokens,
                "commitments": commitments,
                "proof_version": GRAIL_PROOF_VERSION,
                "model": {"name": model_name, "layer_index": LAYER_INDEX},
                "signature": signature.hex(),
                "beacon": {"randomness": randomness},
                "rollout": {
                    "prompt_length": prompt_length,
                    "completion_length": n_tokens - prompt_length,
                    "success": True,
                    "total_reward": 0.0,
                    "advantage": 0.0,
                    "token_logprobs": token_logprobs,
                },
            })
        return commits, p_stops
