"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

import random as _random

from reliquary.constants import (
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MAX_TRUNCATED_PER_SUBMISSION,
    M_ROLLOUTS,
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

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)


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


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    The reference miner uses uniform-random selection with rejection
    sampling against the cooldown set. More sophisticated strategies
    (pre-screening zone probability, etc.) are left to miner operators.

    Raises ``RuntimeError`` if no eligible prompt can be found — typically
    because the env is fully in cooldown.
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


def _population_std(values: list[float]) -> float:
    """Population stddev — matches validator.verifier.rewards_std exactly.

    Replicated locally so the engine doesn't import from the validator
    package (which pulls in heavy R2/HF deps).
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return variance ** 0.5


def _current_drand_round_at_send() -> int:
    """Drand quicknet round currently in progress at wall-clock now.

    Called just before POSTing /submit so the attached round matches
    what the validator sees at receipt (modulo configured tolerance).

    Boundary safety: a miner firing at t=2.99s of round R can land at
    the validator at t=3.00s of R+1; with
    DRAND_ROUND_BACKWARD_TOLERANCE=0 that produces a stale_round
    reject. Absorb up to ``safety_s`` of POST+queue latency by
    sleeping past the next boundary when we're within that window of
    one. Tuned via RELIQUARY_DRAND_BOUNDARY_SAFETY_S (default 0.5 s).
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
    elapsed_in_round = (now - genesis) % period
    remaining = period - elapsed_in_round
    if remaining < safety_s:
        time.sleep(remaining + 0.05)
        now = time.time()
    return compute_current_drand_round(now, genesis, period)


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

        from reliquary.miner.prompt_picker import build_picker_from_env
        self._picker = build_picker_from_env(wallet.hotkey.ss58_address)
        if self._picker is not None:
            logger.info(
                "Supabase prompt picker enabled (sigma_min=%.2f)",
                self._picker.sigma_min,
            )
        else:
            logger.info(
                "Supabase prompt picker disabled — using uniform-random selection"
            )

        # Supabase pregen cache — drains batches produced by sibling
        # boxes (or our own prior runs) so the GPU sees a hot queue
        # immediately after ckpt advance without paying for fresh
        # vLLM gen. Disabled cleanly when SUPABASE_URL/KEY are empty.
        from reliquary.miner.persistence import cache_from_env
        self._cache = cache_from_env(
            miner_hotkey=wallet.hotkey.ss58_address,
        )
        # Per-ckpt tracking of which prompt_idx values we've already
        # pulled from the cache into our local _pregen_queue, to avoid
        # re-loading the same batch on every refresh tick. mark_consumed
        # then runs on a successful submit so other miners see it as
        # taken.
        self._cache_loaded_by_ckpt: dict[str, set[int]] = {}
        self._cache_poll_interval = float(
            os.environ.get("RELIQUARY_CACHE_POLL_S", "15")
        )

        # Speculative pre-generation queue. A background coroutine
        # continuously generates rollouts for prompts the picker would
        # select (excluding cooldown / known_bad / already-attempted),
        # so when a new window opens with fresh randomness we have a
        # ready batch and can sketch+sign+submit in seconds — beating
        # the seal-extension race against ~256 other miners. Each
        # queued entry is tagged with the checkpoint_hash it was
        # generated under and validated against the live state at
        # drain time (stale entries are dropped).
        self._pregen_queue: asyncio.Queue = asyncio.Queue(
            maxsize=int(os.environ.get("RELIQUARY_PREGEN_CAPACITY", "4"))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _submit_one(
        self,
        *,
        prompt_idx: int,
        problem: dict,
        generations: list[dict],
        sketch: dict | None,
        randomness: str,
        state,
        local_hash: str,
        url: str,
        client,
        results: list,
    ) -> bool:
        """Process one (cache-loaded or fresh) batch through the validator
        pipeline: length / truncation / hash-dup pre-checks → build →
        zone check → malformed check → window-race guard → envelope sig
        → POST. Records the outcome to the picker and marks cache rows
        consumed on accept. Returns ``True`` iff a POST to /submit was
        actually made — counts against the per-hotkey/window quota
        regardless of whether the validator accepted, batch-filled, or
        rate-limited the request. Returns ``False`` on any pre-POST
        abandonment (length / truncation-skip / hash-dup-skip /
        zone-skip / malformed-skip / window-race / GRAIL error).

        Extracted from the legacy single-iteration main loop so that
        multiple cache pregens can drain in a single outer iteration
        (cuts batch_filled losses on windows where the cache has >1
        ready batch). Cache items skip the fresh-only pre-checks
        (truncation + hash dup); their tokens were vetted upstream by
        the producer that wrote them to ``pregen_batches``.
        """
        from reliquary.miner.submitter import (
            SubmissionError, get_window_state_v2, submit_batch_v2,
        )

        if len(generations) < M_ROLLOUTS:
            logger.warning(
                "generated %d/%d for prompt %d; skipping",
                len(generations), M_ROLLOUTS, prompt_idx,
            )
            return False

        # Pre-check cap-truncation (fresh-only).
        if sketch is None:
            trunc_count = sum(
                1 for g in generations
                if g.get("cap_truncated", False)
            )
            if trunc_count > MAX_TRUNCATED_PER_SUBMISSION:
                logger.info(
                    "truncation-skip window=%d prompt=%d "
                    "%d/%d rollouts cap-truncated (>%d) "
                    "— abandoning batch (would reject "
                    "bad_termination)",
                    state.window_n, prompt_idx,
                    trunc_count, len(generations),
                    MAX_TRUNCATED_PER_SUBMISSION,
                )
                if self._picker is not None:
                    self._picker._attempted_by_ckpt \
                        .setdefault(local_hash, set()) \
                        .add(prompt_idx)
                    # Persist so the picker (and the pregen producer's
                    # picker) deprioritises prompts that repeatedly
                    # cap-truncate. sigma=0.0 marks it bad for this ckpt,
                    # matching the malformed/clone skip pattern; truncation
                    # has bounded re-roll variance so this is consistent
                    # with the picker's "bad wins per-ckpt" policy. Local
                    # prompt selection only — never affects what is
                    # submitted or how it is proven.
                    asyncio.create_task(asyncio.to_thread(
                        self._picker.record,
                        prompt_idx=prompt_idx,
                        checkpoint_hash=local_hash,
                        k=0,
                        sigma=0.0,
                        status="truncation_skip",
                        truncated_count=trunc_count,
                    ))
                return False

        # Fresh-rollout HASH_DUPLICATE pre-flight (fresh-only).
        if (
            sketch is None
            and self._cache is not None
            and getattr(self._cache, "enabled", False)
        ):
            try:
                from reliquary.validator.dedup import (
                    compute_rollout_hash,
                )
                accepted = await asyncio.to_thread(
                    self._cache.accepted_hashes_for_prompt,
                    prompt_idx, local_hash,
                )
                if accepted:
                    dup_hits = sum(
                        1 for g in generations
                        if compute_rollout_hash(g["tokens"]).hex()
                        in accepted
                    )
                    if dup_hits:
                        logger.info(
                            "hash-dup-skip window=%d prompt=%d "
                            "%d/%d rollouts already in "
                            "accepted_hashes — picking another "
                            "prompt",
                            state.window_n, prompt_idx,
                            dup_hits, len(generations),
                        )
                        if self._picker is not None:
                            self._picker._attempted_by_ckpt \
                                .setdefault(local_hash, set()) \
                                .add(prompt_idx)
                        return False
            except Exception:
                logger.exception(
                    "fresh-rollout hash preflight failed "
                    "for prompt=%d; submitting anyway",
                    prompt_idx,
                )

        # Build (fast or slow path).
        if sketch is not None:
            rollout_submissions = self._finalize_rollouts_from_sketch(
                sketch, randomness,
            )
        else:
            rollout_submissions = self._build_rollout_submissions_batched(
                generations, problem, randomness,
            )

        # Zone pre-check.
        rewards = [float(r.reward) for r in rollout_submissions]
        sigma = _population_std(rewards)
        k_correct = sum(1 for r in rewards if r > 0.5)
        sigma_min = self._picker.sigma_min if self._picker else 0.43
        if sigma < sigma_min:
            logger.info(
                "zone-skip window=%d prompt=%d k=%d/%d sigma=%.3f < %.2f",
                state.window_n, prompt_idx, k_correct, M_ROLLOUTS,
                sigma, sigma_min,
            )
            if self._picker is not None:
                asyncio.create_task(asyncio.to_thread(
                    self._picker.record,
                    prompt_idx=prompt_idx,
                    checkpoint_hash=local_hash,
                    k=k_correct,
                    sigma=sigma,
                    status="zone_skip",
                ))
            return False

        # Malformed-final-answer pre-check.
        from reliquary.validator.boxed_integrity import (
            has_malformed_final_answer,
        )
        malformed_idx: int | None = None
        for i, r in enumerate(rollout_submissions):
            completion_tokens = r.tokens[generations[i]["prompt_length"]:]
            completion_text = self.tokenizer.decode(completion_tokens)
            bad, _ = has_malformed_final_answer(
                reward=float(r.reward), text=completion_text,
            )
            if bad:
                malformed_idx = i
                break
        if malformed_idx is not None:
            logger.info(
                "malformed-skip window=%d prompt=%d rollout=%d "
                "k=%d/%d sigma=%.3f — abandoning batch",
                state.window_n, prompt_idx, malformed_idx,
                k_correct, M_ROLLOUTS, sigma,
            )
            if self._picker is not None:
                asyncio.create_task(asyncio.to_thread(
                    self._picker.record,
                    prompt_idx=prompt_idx,
                    checkpoint_hash=local_hash,
                    k=k_correct,
                    sigma=min(sigma, 0.0),
                    status="malformed_skip",
                ))
            return False

        # Opposite-reward clone pre-check. The validator runs the same
        # detect_opposite_reward_clones (rollout_patterns.py:66) and
        # rejects distribution_suspicious when 3+ pairs of opposite-
        # reward rollouts share similarity >= 0.965 AND length ratio
        # >= 0.94. Math problems with consistent reasoning naturally
        # produce highly similar completions across rollouts even at
        # T=0.9; this can fire as a FALSE POSITIVE on honest output.
        # Catching it client-side saves a per-window slot and a wasted
        # GRAIL proof round-trip.
        try:
            from reliquary.validator.rollout_patterns import (
                detect_opposite_reward_clones,
            )
            clone_texts: list[str] = []
            for i, r in enumerate(rollout_submissions):
                comp_tokens = r.tokens[generations[i]["prompt_length"]:]
                clone_texts.append(self.tokenizer.decode(comp_tokens))
            clone_metrics = detect_opposite_reward_clones(
                clone_texts, rewards,
            )
            if clone_metrics.suspicious:
                logger.info(
                    "clone-skip window=%d prompt=%d matched=%d "
                    "max_sim=%.3f mean_sim=%.3f min_len_ratio=%.3f "
                    "k=%d/%d sigma=%.3f — abandoning batch "
                    "(would reject distribution_suspicious)",
                    state.window_n, prompt_idx,
                    clone_metrics.matched_pairs,
                    clone_metrics.max_similarity,
                    clone_metrics.mean_similarity,
                    clone_metrics.min_length_ratio,
                    k_correct, M_ROLLOUTS, sigma,
                )
                if self._picker is not None:
                    asyncio.create_task(asyncio.to_thread(
                        self._picker.record,
                        prompt_idx=prompt_idx,
                        checkpoint_hash=local_hash,
                        k=k_correct,
                        sigma=min(sigma, 0.0),
                        status="clone_skip",
                    ))
                return False
        except Exception:
            logger.exception(
                "clone preflight failed for prompt=%d; submitting anyway",
                prompt_idx,
            )

        merkle_root = _compute_merkle_root(rollout_submissions)

        # Window-race guard (always-on, including cache items).
        # Previously gated on ``sketch is None`` (fresh-gen only); but
        # cache-loaded items still take 1-2s for sketch finalize +
        # POST, during which state.randomness can roll over, producing
        # bad_envelope_signature at the validator. The state re-fetch
        # adds ~100-200ms but eliminates the race-loss class.
        try:
            state_now = await get_window_state_v2(url, client=client)
        except Exception as exc:
            logger.warning(
                "window-race state-fetch-failed (%s) prompt=%d k=%d "
                "sigma=%.3f — abandoning batch", exc, prompt_idx,
                k_correct, sigma,
            )
            return False
        if (
            state_now.window_n != state.window_n
            or state_now.randomness != state.randomness
            or not state_now.randomness
        ):
            logger.warning(
                "window-race window=%d→%d randomness=%s→%s prompt=%d "
                "k=%d/%d sigma=%.3f — abandoning batch",
                state.window_n, state_now.window_n,
                (state.randomness or "")[:8],
                (state_now.randomness or "")[:8],
                prompt_idx, k_correct, M_ROLLOUTS, sigma,
            )
            return False

        current_round = _current_drand_round_at_send()
        import os as _os
        _nonce = _os.urandom(16).hex()
        _envelope_sig = sign_envelope(
            wallet=self.wallet,
            miner_hotkey=self.wallet.hotkey.ss58_address,
            window_start=state.window_n,
            prompt_idx=prompt_idx,
            merkle_root=merkle_root,
            checkpoint_hash=local_hash,
            drand_round=current_round,
            randomness=state.randomness or "",
            nonce=_nonce,
        ).hex()
        request = BatchSubmissionRequest(
            miner_hotkey=self.wallet.hotkey.ss58_address,
            prompt_idx=prompt_idx,
            window_start=state.window_n,
            merkle_root=merkle_root,
            rollouts=rollout_submissions,
            checkpoint_hash=local_hash,
            drand_round=current_round,
            nonce=_nonce,
            envelope_signature=_envelope_sig,
        )
        try:
            resp = await submit_batch_v2(url, request, client=client)
            reason_str = (
                resp.reason.value if hasattr(resp.reason, "value")
                else str(resp.reason)
            )
            logger.info(
                "submitted window=%d prompt=%d accepted=%s reason=%s "
                "k=%d/%d sigma=%.3f",
                state.window_n, prompt_idx, resp.accepted, reason_str,
                k_correct, M_ROLLOUTS, sigma,
            )
            results.append(resp)
            if self._picker is not None:
                status_label = (
                    "submitted_accepted" if resp.accepted
                    else f"submitted_rejected:{reason_str}"
                )
                asyncio.create_task(asyncio.to_thread(
                    self._picker.record,
                    prompt_idx=prompt_idx,
                    checkpoint_hash=local_hash,
                    k=k_correct,
                    sigma=sigma,
                    status=status_label,
                ))
            if (
                resp.accepted
                and self._cache is not None
                and self._cache.enabled
                and prompt_idx in self._cache_loaded_by_ckpt.get(
                    local_hash, set()
                )
            ):
                asyncio.create_task(asyncio.to_thread(
                    self._cache.mark_consumed,
                    prompt_idx, local_hash,
                ))
        except SubmissionError as exc:
            logger.error("submit failed: %s", exc)
        # POST was attempted (whether validator accepted, batch_filled,
        # rate_limited, or the transport itself errored) — counts
        # against the per-hotkey/window quota either way.
        return True

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

        from reliquary.constants import (
            B_BATCH, M_ROLLOUTS, MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
            POLL_INTERVAL_SECONDS,
        )
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
            get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import (
            BatchSubmissionRequest, WindowState,
        )

        # Resolve validator URL (once).
        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        # v2.3: randomness is fetched per-window from /state instead of
        # recomputed locally. The validator aligns window OPEN to a drand
        # boundary and binds randomness to the round publishing at that
        # boundary — a value that didn't exist a few seconds earlier, so
        # nothing to pre-fetch. The miner just reads what /state reports.
        rng = random.Random()
        results = []
        local_n = 0
        local_hash = ""
        picker_hydrated_for: str | None = None
        # Per-window POST budget. Validator caps each hotkey at
        # MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW=8 /submit calls per
        # window. The cache-multi-drain easily exceeds this when the
        # pregen_queue is full; the surplus reject as ``rate_limited``
        # which still counts against the same budget on the validator.
        # Track POSTs locally and skip both cache-drain and fresh-gen
        # once we're at cap until the next window opens.
        last_window_seen = -1
        submitted_this_window = 0
        # Shared mutable reference to the freshest /state the main loop
        # has seen — the pregen worker reads this to pick prompts
        # against the current cooldown_prompts + checkpoint without
        # having to make its own HTTP polls.
        latest_state_ref: list = [None]

        pregen_task: asyncio.Task | None = None
        cache_task: asyncio.Task | None = None
        if self._picker is not None:
            pregen_task = asyncio.create_task(
                self._pregen_loop(latest_state_ref, rng)
            )
            logger.info("pregen worker started (queue maxsize=%d)",
                        self._pregen_queue.maxsize)
        if self._cache is not None and self._cache.enabled:
            cache_task = asyncio.create_task(
                self._cache_consumer_loop(latest_state_ref)
            )
            logger.info("cache consumer worker started (poll=%.1fs)",
                        self._cache_poll_interval)

        # /verdicts poller. /submit returns only a provisional SUBMITTED
        # sentinel; the REAL outcome of each submission (worker_dropped,
        # distribution_suspicious, accepted, GRAIL_FAIL, etc.) is
        # published at GET /verdicts/{hotkey} after the validator's
        # async worker decides. Polling surfaces those decisions into
        # our log so we can debug async-rejection patterns without
        # depending on the dashboard (whose IP is currently Vercel-
        # banned for this box).
        verdicts_task = asyncio.create_task(
            self._verdicts_loop(url)
        )
        logger.info(
            "verdicts poller started (poll=%.1fs)",
            float(os.environ.get("RELIQUARY_VERDICTS_POLL_S", "10")),
        )

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                try:
                    state = await get_window_state_v2(url, client=client)
                    latest_state_ref[0] = state
                except SubmissionError:
                    # /state may return 503 between windows; wait briefly.
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                except Exception as e:
                    logger.debug("state fetch failed: %s", e)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # Pull new checkpoint if needed (works at any state).
                try:
                    local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
                        state=state, local_n=local_n, local_hash=local_hash,
                        local_model=self.hf_model,
                        download_fn=_hf_download,
                        load_fn=self._load_checkpoint,
                    )
                except Exception:
                    logger.exception("checkpoint pull failed; keeping local")

                # Hydrate the Supabase known-bad set whenever the active
                # checkpoint changes — the prepared-prompt cache is scoped
                # per ckpt_n because a new fine-tune can flip a previously
                # hopeless prompt into the in-zone range.
                if self._picker is not None and picker_hydrated_for != local_hash:
                    try:
                        n_good, n_bad = await asyncio.to_thread(
                            self._picker.hydrate, local_hash
                        )
                        logger.info(
                            "picker hydrated for ckpt=%s (n=%d): %d good, %d bad",
                            (local_hash or "")[:12], local_n, n_good, n_bad,
                        )
                        picker_hydrated_for = local_hash
                    except Exception:
                        logger.exception(
                            "picker hydrate failed for ckpt=%s; running with "
                            "empty known-bad set", (local_hash or "")[:12],
                        )
                        picker_hydrated_for = local_hash  # don't retry every loop

                if state.state != WindowState.OPEN:
                    # Tight poll while waiting for OPEN. With a 1s sleep,
                    # we'd detect OPEN up to 1000ms late and our first
                    # POST lands well after top miners have filled the
                    # validator's per-window batch (batch_filled). At
                    # 0.1s we catch the transition within ~100ms, which
                    # is the dominant component of the OPEN→POST gap.
                    await asyncio.sleep(
                        float(os.environ.get("RELIQUARY_OPEN_POLL_S", "0.1"))
                    )
                    continue

                # NOTE: we used to gate on ``state.valid_submissions >=
                # B_BATCH`` here, but that proxy is wrong in two ways:
                #
                #   1. ``valid_submissions`` counts ALL hotkeys' validated
                #      submissions in the window, not distinct prompts.
                #      With 256 miners competing, this number routinely
                #      passes B_BATCH while the validator's *distinct-
                #      prompts-eligible* count is still below the seal
                #      threshold — i.e. the batch is still accepting.
                #   2. Even after seal triggers, submissions at the SAME
                #      drand round as ``seal_trigger_round`` are still
                #      accepted (batcher.py:392-395). Skipping forfeits
                #      that window.
                #
                # The cost of submitting late is one BATCH_FILLED
                # rejection (no rate-limit consumption against
                # MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW=8). The benefit of
                # keeping the GPU saturated is direct: more generated
                # rollouts in the same wall-clock means more chances at
                # the next acceptable round. Net: let the validator
                # decide.

                # v2.3: trust the validator's per-window randomness rather
                # than recomputing locally. Empty string means the validator
                # hasn't yet finished _set_window_randomness — wait briefly.
                randomness = state.randomness
                if not randomness:
                    await asyncio.sleep(0.1)
                    continue

                cooldown_set = set(state.cooldown_prompts)

                # Reset per-window POST counter on window advance.
                if state.window_n != last_window_seen:
                    last_window_seen = state.window_n
                    submitted_this_window = 0

                # If we've already used our entire per-hotkey budget for
                # this window, no point doing more work — wait briefly
                # for the next window to open.
                if submitted_this_window >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
                    await asyncio.sleep(0.1)
                    continue

                # Drain ALL valid cache items into a list, then submit
                # each in turn within this iteration via _submit_one.
                # Previously one cache item was taken per outer iter
                # (state poll + window check + cache pop) — that added
                # ~100-300ms of latency between successive cache
                # submissions and caused us to lose the batch_filled
                # race when the cache had >1 ready batch. Now multiple
                # cache pregens drain in fast succession; fresh-gen
                # falls through only when the cache is empty.
                cache_items: list[dict] = []
                while not self._pregen_queue.empty():
                    try:
                        item = self._pregen_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    # Backwards-compatible unpack: pre-sketch refactor
                    # pushed 4-tuples, post-refactor pushes 5-tuples.
                    if len(item) == 5:
                        p_idx, p_problem, p_gens, p_sketch, p_hash = item
                    else:
                        p_idx, p_problem, p_gens, p_hash = item
                        p_sketch = None
                    if p_hash != local_hash:
                        continue  # stale ckpt
                    if p_idx in cooldown_set:
                        continue  # already cooldowned by validator
                    if (
                        self._picker is not None
                        and p_idx in self._picker.known_bad(local_hash)
                    ):
                        continue  # turned out bad during another iteration
                    cache_items.append({
                        "prompt_idx": p_idx, "problem": p_problem,
                        "generations": p_gens, "sketch": p_sketch,
                    })

                if cache_items:
                    for ci in cache_items:
                        if (
                            submitted_this_window
                            >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
                        ):
                            break  # hit per-hotkey/window quota
                        posted = await self._submit_one(
                            prompt_idx=ci["prompt_idx"],
                            problem=ci["problem"],
                            generations=ci["generations"],
                            sketch=ci["sketch"],
                            randomness=randomness,
                            state=state,
                            local_hash=local_hash,
                            url=url,
                            client=client,
                            results=results,
                        )
                        if posted:
                            submitted_this_window += 1
                    continue  # cache drained — skip fresh-gen this iter

                # No cache items — pick + generate fresh inline.
                try:
                    if self._picker is not None:
                        prompt_idx = self._picker.pick(
                            self.env, cooldown_set, local_hash, rng=rng,
                        )
                    else:
                        prompt_idx = pick_prompt_idx(
                            self.env, cooldown_set, rng=rng,
                        )
                except RuntimeError:
                    logger.info("env fully in cooldown; sleeping")
                    await asyncio.sleep(5)
                    continue

                problem = self.env.get_problem(prompt_idx)
                # Wrap the generation thread in an asyncio timeout.
                # A bare ``await asyncio.to_thread(...)`` blocks the
                # main coroutine indefinitely if the CUDA kernel
                # wedges (observed on Blackwell sm_120 + FA2-kernel-
                # hub: thread sits in cudaLaunchKernel forever, no
                # exception ever raised).
                gen_timeout_s = float(
                    os.environ.get("RELIQUARY_GEN_TIMEOUT_S", "120")
                )
                try:
                    generations = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._generate_m_rollouts, problem, randomness,
                        ),
                        timeout=gen_timeout_s,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "reject reason=generation_timeout prompt=%d "
                        "after %.0fs; clearing cache and continuing",
                        prompt_idx, gen_timeout_s,
                    )
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    await asyncio.sleep(1)
                    continue
                except Exception as exc:
                    logger.error(
                        "generation failed for prompt %d: %s; clearing "
                        "cache and continuing", prompt_idx, exc,
                    )
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    await asyncio.sleep(1)
                    continue

                posted = await self._submit_one(
                    prompt_idx=prompt_idx,
                    problem=problem,
                    generations=generations,
                    sketch=None,
                    randomness=randomness,
                    state=state,
                    local_hash=local_hash,
                    url=url,
                    client=client,
                    results=results,
                )
                if posted:
                    submitted_this_window += 1

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

        # 2. Reload vLLM gen engine on the generation GPU. vLLM doesn't
        # expose a weights-swap path, so we destroy and recreate the
        # engine. The engine runs in a separate VLLM::EngineCore
        # subprocess that holds VRAM until killed — `del` on the LLM
        # object alone does NOT release it. We must call shutdown() (if
        # exposed) AND fall back to killing the subprocess directly.
        try:
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            from vllm import LLM as _VllmLLM
            _gpu_mem_util = float(os.environ.get("RELIQUARY_VLLM_GPU_MEM_UTIL", "0.35"))
        except Exception:
            logger.exception(
                "vLLM import failed; cannot reload gen engine. "
                "Generation is BROKEN until restart."
            )
            self.vllm_model = None
            self._loaded_checkpoint_path = None
            return self.hf_model

        old_gen = self.vllm_model
        self.vllm_model = None
        if old_gen is not None:
            # Call any exposed shutdown method first. Different vLLM
            # versions name this differently (`shutdown`, `close`,
            # `llm_engine.shutdown`); we try all of them and ignore
            # missing-attribute errors.
            for path in ("shutdown", "close"):
                fn = getattr(old_gen, path, None)
                if fn is not None:
                    try:
                        fn()
                    except Exception:
                        logger.debug("vLLM %s() raised; continuing", path)
            inner = getattr(old_gen, "llm_engine", None)
            if inner is not None:
                fn = getattr(inner, "shutdown", None)
                if fn is not None:
                    try:
                        fn()
                    except Exception:
                        logger.debug("vLLM llm_engine.shutdown() raised")
            del old_gen
            try:
                import gc
                gc.collect()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            # Belt-and-braces: kill any lingering EngineCore subprocess.
            # vLLM spawns these as children of the miner; pkill on the
            # well-known process name reliably reaps them.
            try:
                import subprocess
                subprocess.run(
                    ["pkill", "-9", "-f", "VLLM::EngineCore"],
                    check=False, timeout=10,
                )
            except Exception:
                pass
            # Give the OS a moment to reclaim VRAM from the killed
            # subprocess before the new engine probes free memory.
            import time as _t
            _t.sleep(3)

        try:
            new_gen = _VllmLLM(
                model=local_path,
                dtype="bfloat16",
                max_model_len=MAX_NEW_TOKENS_PROTOCOL_CAP,
                gpu_memory_utilization=_gpu_mem_util,
                tensor_parallel_size=1,
                trust_remote_code=False,
            )
        except Exception:
            logger.exception(
                "Failed to reload vLLM engine from %s; miner generation "
                "is BROKEN until the next successful pull. hf_model was "
                "swapped so GRAIL proofs would be inconsistent.",
                local_path,
            )
            self._loaded_checkpoint_path = None
            return self.hf_model

        self.vllm_model = new_gen
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self._loaded_checkpoint_path = local_path
        logger.info("Checkpoint %s loaded into both models", local_path)
        return self.hf_model

    def _generate_m_rollouts(self, problem, randomness) -> list[dict]:
        """Generate M_ROLLOUTS completions at T_PROTO via vLLM.

        Uses ``SamplingParams(n=M_ROLLOUTS)`` so vLLM produces all 8
        completions inside one continuous-batched request — ~5-10x
        faster than HF's batched ``.generate()`` on the same hardware,
        which is what makes the miner fit inside a 60s window.

        Each completion is EOS-truncated at the first stop token so the
        validator's GRAIL forward pass sees exactly the tokens the
        sampler emitted (no padding tail).

        The vLLM generation engine and the HF sketch engine use
        different attention kernels — that's fine for envelope and
        commit verification, which only depend on the SKETCH model
        (HF + FA2). It DOES introduce drift on per-token logprobs and
        p(EOS), which is the price for fast generation: validator
        rejections like ``logprob_mismatch`` / ``bad_termination`` /
        ``distribution_suspicious`` are normal background noise at
        well-tuned operating points (the top miners run at ~25% accept).
        """
        from vllm import SamplingParams

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)
        max_tokens = min(
            self.max_new_tokens,
            MAX_NEW_TOKENS_PROTOCOL_CAP - prompt_length,
        )
        if max_tokens <= 0:
            return []

        # Stop on EVERY EOS the generation_config lists, not just the
        # tokenizer's primary eos_token_id. Qwen3 declares both
        # <|im_end|> (151645) and <|endoftext|> (151643) as valid stops;
        # vLLM only stops on tokenizer.eos_token_id by default, so a
        # rollout that emits <|endoftext|> first runs all the way to
        # max_tokens and gets counted as cap-truncated by the validator
        # (>5/8 truncations → BAD_TERMINATION). Pull the full set from
        # the loaded model's generation_config when available.
        eos_ids: list[int] = []
        gen_cfg = getattr(self.hf_model, "generation_config", None)
        if gen_cfg is not None and gen_cfg.eos_token_id is not None:
            eid = gen_cfg.eos_token_id
            eos_ids = list(eid) if isinstance(eid, (list, tuple)) else [int(eid)]
        if not eos_ids and self.tokenizer.eos_token_id is not None:
            eos_ids = [self.tokenizer.eos_token_id]
        eos_set = set(eos_ids)

        sp = SamplingParams(
            n=M_ROLLOUTS,
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            top_k=-1 if TOP_K_PROTO <= 0 else TOP_K_PROTO,
            max_tokens=max_tokens,
            stop_token_ids=eos_ids or None,
        )
        request_outputs = self.vllm_model.generate(
            {"prompt_token_ids": prompt_tokens},
            sp,
            use_tqdm=False,
        )
        completions = request_outputs[0].outputs
        rollouts = []
        for comp in completions:
            gen = list(comp.token_ids)
            # Truncate at the first occurrence of ANY configured EOS.
            first_eos = next(
                (i for i, t in enumerate(gen) if t in eos_set), None,
            )
            if first_eos is not None:
                gen = gen[: first_eos + 1]
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
                # Cap-truncated = sampler hit max_tokens without emitting
                # any configured EOS. The validator counts these against
                # MAX_TRUNCATED_PER_SUBMISSION and rejects bad_termination
                # if >5/8 in the batch — pre-check it in the main loop.
                "cap_truncated": first_eos is None,
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

    def _compute_pregen_sketch(
        self, generations: list[dict], problem: dict,
    ) -> dict:
        """Run the HF forward + reward in pregen, store everything that
        does NOT depend on per-window randomness.

        This is the expensive ~3-5s step. Moved out of the submit
        critical path so that when window OPENs, finalising a batch
        only costs commitments (~50ms matmul) + signatures (~10ms) +
        POST (~270ms) — total ~400ms vs ~5-8s for the inline path.
        Beats the validator's MAX_PROOF_CANDIDATES_PER_WINDOW=32 race
        much more reliably.

        Returns a dict with per-rollout pre-computed arrays:
          ``hidden_states[i]`` — kept on GPU for the fast commitments
            matmul at submit time (~53 MB total for 8 rollouts × 1024
            tokens × 2560 hidden dim @ bf16). Fits 4× pregen capacity
            comfortably alongside vLLM's ~50 GB.
          ``token_logprobs[i]`` — full list per rollout, ready to drop
            into the commit dict.
          ``completion_text[i]``, ``reward[i]``, ``real_len[i]``,
            ``prompt_length[i]``, ``all_tokens[i]`` — bookkeeping.

        Falls back to None on any error so the pregen worker logs and
        moves on; the slow inline path remains the safety net.
        """
        import torch

        from reliquary.shared.forward import forward_single_layer

        device = f"cuda:{self.proof_gpu}"
        token_lists: list[list[int]] = [g["tokens"] for g in generations]
        prompt_lens: list[int] = [g["prompt_length"] for g in generations]
        real_lens = [len(t) for t in token_lists]
        max_len = max(real_lens)
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        padded = [t + [pad_id] * (max_len - len(t)) for t in token_lists]
        attn_mask_list = [
            [1] * real_lens[i] + [0] * (max_len - real_lens[i])
            for i in range(len(token_lists))
        ]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn_mask_list, dtype=torch.long, device=device)

        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, input_ids, attention_mask, LAYER_INDEX
            )

        per_rollout: list[dict] = []
        for i, _ in enumerate(generations):
            all_tokens = token_lists[i]
            prompt_length = prompt_lens[i]
            real_len = real_lens[i]
            completion_tokens = all_tokens[prompt_length:]
            completion_text = self.tokenizer.decode(completion_tokens)
            reward = float(self.env.compute_reward(problem, completion_text))

            hs_i = hidden_states[i, :real_len].contiguous()        # [seq, hidden]
            logits_i = logits[i, :real_len]                        # [seq, vocab]
            log_probs = torch.log_softmax(logits_i.float(), dim=-1)
            token_logprobs: list[float] = [
                log_probs[j - 1, all_tokens[j]].item()
                for j in range(prompt_length, real_len)
            ]
            per_rollout.append({
                "all_tokens": all_tokens,
                "prompt_length": prompt_length,
                "real_len": real_len,
                "completion_text": completion_text,
                "reward": reward,
                "hidden_states": hs_i,        # stays on GPU
                "token_logprobs": token_logprobs,
            })
        # Free the big logits tensor — we already extracted the per-token
        # log-probs we need; keeping it would waste ~3 GB for nothing.
        del logits
        return {
            "model_name": getattr(self.hf_model, "name_or_path", "unknown"),
            "per_rollout": per_rollout,
        }

    def _finalize_rollouts_from_sketch(
        self, sketch: dict, randomness: str,
    ) -> list[RolloutSubmission]:
        """Submit-time fast path: compose commitments + sign from pre-computed sketch.

        ``sketch`` is the dict returned by ``_compute_pregen_sketch``.
        Only the commitments matmul (depends on r_vec(randomness)) and
        the signature are evaluated here — no model forward.
        """
        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding

        model_name: str = sketch["model_name"]
        r_vec = self._verifier.generate_r_vec(randomness)
        rollouts: list[RolloutSubmission] = []
        for r in sketch["per_rollout"]:
            commitments = self._verifier.create_commitments_batch(
                r["hidden_states"], r_vec,
            )
            signature = sign_commit_binding(
                r["all_tokens"], randomness, model_name, LAYER_INDEX,
                commitments, self.wallet,
            )
            commit = {
                "tokens": r["all_tokens"],
                "commitments": commitments,
                "proof_version": GRAIL_PROOF_VERSION,
                "model": {"name": model_name, "layer_index": LAYER_INDEX},
                "signature": signature.hex(),
                "beacon": {"randomness": randomness},
                "rollout": {
                    "prompt_length": r["prompt_length"],
                    "completion_length": r["real_len"] - r["prompt_length"],
                    "success": True,
                    "total_reward": 0.0,
                    "advantage": 0.0,
                    "token_logprobs": r["token_logprobs"],
                },
            }
            rollouts.append(RolloutSubmission(
                tokens=r["all_tokens"], reward=r["reward"], commit=commit,
            ))
        return rollouts

    def _build_rollout_submissions_batched(
        self, generations: list[dict], problem: dict, randomness: str,
    ) -> list[RolloutSubmission]:
        """Build all M rollout submissions in one batched HF forward.

        The per-rollout ``_build_grail_commit`` path runs a separate HF
        forward per generation — 8 sequential ~2-3s forwards at
        max_new_tokens=2048 dominates the submit-iteration wall-clock
        (~15-20s of GPU time per submission vs ~10s for the vLLM gen
        batch). Pad-and-batch them into a single forward, then extract
        per-rollout hidden_states/logits slices for the commitment
        build. Causal attention with right-padding + a proper
        attention_mask gives bit-identical hidden_states/logits for
        non-padded positions, so commitments and log_probs match what
        a per-rollout forward would have produced (and what the
        validator computes when verifying).
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        device = f"cuda:{self.proof_gpu}"
        token_lists: list[list[int]] = [g["tokens"] for g in generations]
        prompt_lens: list[int] = [g["prompt_length"] for g in generations]
        real_lens = [len(t) for t in token_lists]
        max_len = max(real_lens)
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        padded = [t + [pad_id] * (max_len - len(t)) for t in token_lists]
        attn_mask_list = [
            [1] * real_lens[i] + [0] * (max_len - real_lens[i])
            for i in range(len(token_lists))
        ]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn_mask_list, dtype=torch.long, device=device)

        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, input_ids, attention_mask, LAYER_INDEX
            )

        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")
        r_vec = self._verifier.generate_r_vec(randomness)

        rollouts: list[RolloutSubmission] = []
        for i, gen in enumerate(generations):
            all_tokens = token_lists[i]
            prompt_length = prompt_lens[i]
            real_len = real_lens[i]
            completion_tokens = all_tokens[prompt_length:]
            completion_text = self.tokenizer.decode(completion_tokens)
            reward = self.env.compute_reward(problem, completion_text)

            # Slice to the real (non-padded) prefix — bit-identical to
            # a single-sequence forward on this rollout's tokens.
            hs_i = hidden_states[i, :real_len]            # [seq_len, hidden]
            logits_i = logits[i, :real_len]               # [seq_len, vocab]
            commitments = self._verifier.create_commitments_batch(hs_i, r_vec)

            log_probs = torch.log_softmax(logits_i.float(), dim=-1)
            token_logprobs: list[float] = [
                log_probs[j - 1, all_tokens[j]].item()
                for j in range(prompt_length, real_len)
            ]
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
            rollouts.append(RolloutSubmission(
                tokens=all_tokens, reward=reward, commit=commit,
            ))
        return rollouts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _cache_consumer_loop(self, latest_state_ref: list) -> None:
        """Drain Supabase ``pregen_batches`` rows into the local queue.

        Sibling box(es) (or our own prior run) write completed M=8 token
        batches to ``pregen_batches`` keyed by (prompt_idx, ckpt_hash)
        with a ``tier`` label and a ``consumed_at IS NULL`` flag.
        We poll for unconsumed rows on the current ckpt, synthesise the
        engine's local ``generations`` format, run the HF sketch
        forward locally (~3-5s per batch — still cheaper than a fresh
        vLLM gen) and push into ``_pregen_queue`` so the main loop's
        submit path drains them like any locally pre-generated batch.

        Deduplication: we track loaded prompt_idx per ckpt_hash; a
        prompt is loaded at most once per process lifetime. Cleared on
        ckpt advance.

        ``mark_consumed`` is fired only on a successful POST (any
        accept/reject reason that reached the validator) — the row
        being "consumed" means it was used, not that the validator
        accepted it. This is in the main loop's submit success path.
        """
        TIER_ORDER = ("stable", "proven", "exploratory", "untagged")
        while True:
            try:
                state = latest_state_ref[0]
                if state is None:
                    await asyncio.sleep(self._cache_poll_interval)
                    continue
                ckpt_hash = getattr(state, "checkpoint_revision", "") or ""
                if not ckpt_hash:
                    await asyncio.sleep(self._cache_poll_interval)
                    continue
                loaded_set = self._cache_loaded_by_ckpt.setdefault(
                    ckpt_hash, set()
                )

                # Pull each tier; "stable" first (highest confidence),
                # then "proven", then everything else. Stop as soon as
                # the local queue is near capacity.
                pulled_this_tick = 0
                for tier in TIER_ORDER:
                    if self._pregen_queue.full():
                        break
                    try:
                        batches = await asyncio.to_thread(
                            self._cache.load_unconsumed_batches_by_tier,
                            ckpt_hash, tier,
                        )
                    except Exception:
                        logger.exception(
                            "cache load tier=%s failed", tier,
                        )
                        continue
                    if not batches:
                        continue
                    for b in batches:
                        if self._pregen_queue.full():
                            break
                        if b.prompt_idx in loaded_set:
                            continue
                        # Reconstruct the engine-native generations list
                        # from the persisted rollouts. Each persisted
                        # rollout carries {tokens, prompt_length, reward};
                        # the engine's gen format only needs
                        # {tokens, prompt_length}.
                        try:
                            generations = [
                                {"tokens": list(r["tokens"]),
                                 "prompt_length": int(r["prompt_length"])}
                                for r in (b.rollouts or [])
                            ]
                        except Exception:
                            logger.exception(
                                "cache: malformed rollouts for prompt=%d "
                                "(skipping)", b.prompt_idx,
                            )
                            loaded_set.add(b.prompt_idx)  # don't retry
                            continue
                        if len(generations) < M_ROLLOUTS:
                            logger.warning(
                                "cache: prompt=%d has only %d rollouts "
                                "(expected %d); skipping",
                                b.prompt_idx, len(generations), M_ROLLOUTS,
                            )
                            loaded_set.add(b.prompt_idx)
                            continue
                        try:
                            problem = self.env.get_problem(b.prompt_idx)
                        except Exception:
                            logger.exception(
                                "cache: env.get_problem(%d) failed",
                                b.prompt_idx,
                            )
                            loaded_set.add(b.prompt_idx)
                            continue
                        # Pre-flight HASH_DUPLICATE check. If any rollout
                        # in this batch matches an already-accepted hash
                        # on the validator (scraped from R2 by the
                        # producer side), the whole batch will reject:
                        # the validator's per-rollout dedup fails the
                        # submission. We can't re-seed cache-loaded
                        # rollouts, so skip the batch and mark loaded.
                        try:
                            from reliquary.validator.dedup import (
                                compute_rollout_hash,
                            )
                            accepted_set = await asyncio.to_thread(
                                self._cache.accepted_hashes_for_prompt,
                                b.prompt_idx, ckpt_hash,
                            )
                            if accepted_set:
                                dup = False
                                for g in generations:
                                    h_hex = compute_rollout_hash(g["tokens"]).hex()
                                    if h_hex in accepted_set:
                                        dup = True
                                        break
                                if dup:
                                    logger.info(
                                        "cache skip prompt=%d tier=%s — "
                                        "rollout already in accepted_hashes "
                                        "(HASH_DUPLICATE pre-flight); marking "
                                        "consumed to drop the row",
                                        b.prompt_idx, b.tier or "untagged",
                                    )
                                    loaded_set.add(b.prompt_idx)
                                    # Also mark consumed so a sibling
                                    # doesn't keep re-loading this dead row.
                                    asyncio.create_task(asyncio.to_thread(
                                        self._cache.mark_consumed,
                                        b.prompt_idx, ckpt_hash,
                                    ))
                                    continue
                        except Exception:
                            logger.exception(
                                "cache: accepted-hash pre-flight failed "
                                "prompt=%d (continuing anyway)", b.prompt_idx,
                            )
                        # Compute the HF sketch locally so the main loop
                        # takes the fast finalize-from-sketch path.
                        try:
                            sketch = await asyncio.to_thread(
                                self._compute_pregen_sketch,
                                generations, problem,
                            )
                        except Exception:
                            logger.exception(
                                "cache: sketch failed prompt=%d", b.prompt_idx,
                            )
                            continue
                        # Mark loaded BEFORE put so a slow put doesn't
                        # cause a duplicate enqueue on the next tick.
                        loaded_set.add(b.prompt_idx)
                        try:
                            await self._pregen_queue.put(
                                (b.prompt_idx, problem, generations,
                                 sketch, ckpt_hash)
                            )
                            pulled_this_tick += 1
                            logger.info(
                                "cache pulled prompt=%d tier=%s k=%d "
                                "sigma=%.3f → queued",
                                b.prompt_idx, b.tier or "untagged",
                                b.k, b.sigma,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                "cache: queue put failed prompt=%d",
                                b.prompt_idx,
                            )
                            loaded_set.discard(b.prompt_idx)
                if pulled_this_tick:
                    logger.info(
                        "cache: +%d batches this tick (loaded total=%d ckpt=%s)",
                        pulled_this_tick, len(loaded_set), ckpt_hash[:12],
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "cache consumer iteration failed; sleeping briefly"
                )
            await asyncio.sleep(self._cache_poll_interval)

    async def _verdicts_loop(self, url: str) -> None:
        """Poll ``GET /verdicts/{hotkey}`` and log each new validator
        verdict so the miner log surfaces the REAL async outcome of
        every submission (worker_dropped, distribution_suspicious,
        accepted, GRAIL_FAIL, etc.) instead of only the provisional
        ``SUBMITTED`` returned by ``/submit``.

        Tracks seen merkle_roots in-process so each verdict logs once.
        The validator stores verdicts in a capacity-limited ring
        buffer, so the response is naturally bounded; our seen-set is
        capped at 5000 entries with periodic trimming to avoid leak.

        Default 10s poll interval — fast enough that worker decisions
        appear in the log within ~10s of being made, infrequent
        enough that we don't add meaningful load to the validator.
        """
        import httpx

        hotkey = self.wallet.hotkey.ss58_address
        seen_roots: set[str] = set()
        poll_s = float(os.environ.get("RELIQUARY_VERDICTS_POLL_S", "10"))
        endpoint = url.rstrip("/") + f"/verdicts/{hotkey}"
        async with httpx.AsyncClient(timeout=10) as client:
            while True:
                try:
                    r = await client.get(endpoint)
                    if r.status_code == 200:
                        payload = r.json()
                        verdicts = payload.get("verdicts") or []
                        for v in verdicts:
                            mr = v.get("merkle_root") or ""
                            if not mr or mr in seen_roots:
                                continue
                            seen_roots.add(mr)
                            accepted = v.get("accepted")
                            # ``reject_reason`` is the canonical detail
                            # field; ``reason`` is the enum value.
                            reason = (
                                v.get("reject_reason")
                                or v.get("reason")
                                or "?"
                            )
                            window_n = v.get("window_n")
                            qw = v.get("queue_wait_ms")
                            tt = v.get("total_ms")
                            stage = v.get("reject_stage") or "-"
                            logger.info(
                                "verdict window=%s merkle=%s accepted=%s "
                                "reason=%s stage=%s queue_wait_ms=%s "
                                "total_ms=%s",
                                window_n, mr[:12], accepted, reason,
                                stage, qw, tt,
                            )
                        # Trim seen-set periodically to bound memory.
                        if len(seen_roots) > 5000:
                            # Keep the latest 2000 (insertion order).
                            seen_roots = set(list(seen_roots)[-2000:])
                    elif r.status_code == 404:
                        # Validator hasn't seen this hotkey yet — normal
                        # on a fresh restart with no submissions yet.
                        pass
                    else:
                        logger.debug(
                            "verdicts poll non-200: %d", r.status_code,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("verdicts poll failed", exc_info=True)
                await asyncio.sleep(poll_s)

    async def _pregen_loop(self, latest_state_ref: list, rng) -> None:
        """Background coroutine: continuously pregen rollouts for likely picks.

        Generation is independent of the validator's per-window randomness
        (the only thing randomness binds is the GRAIL commit beacon, not
        the sampling), so we can speculatively generate at any time and
        cache the (prompt_idx, problem, generations, checkpoint_hash)
        tuple. The main loop drains the queue at submit time and
        validates each entry against the live state (cooldown_prompts,
        checkpoint_hash) — anything stale gets dropped.

        Backpressure: the queue is bounded; ``put`` blocks when full so
        we don't spin pointlessly while the main loop hasn't drained.
        On any error the loop sleeps briefly and retries. Cancellation
        is propagated so ``mine_window``'s ``finally`` cleanup works.
        """
        from reliquary.constants import M_ROLLOUTS
        from reliquary.protocol.submission import WindowState

        # Gate: skip local gen when the cache consumer is keeping the
        # queue topped up. Cache-loaded batches are validator-verified
        # in-zone; local pregens for unseen prompts mostly zone-skip
        # (k=0/8 or k=8/8 → σ=0) and waste GPU + queue slot. Threshold
        # configurable via RELIQUARY_LOCAL_GEN_GATE_DEPTH (default 4).
        gate_depth = int(
            os.environ.get("RELIQUARY_LOCAL_GEN_GATE_DEPTH", "4")
        )
        while True:
            try:
                state = latest_state_ref[0]
                if state is None:
                    await asyncio.sleep(0.5)
                    continue
                checkpoint_hash = ""  # filled by main loop tracking; safe ""
                # We use the same local_hash tracking as the main loop,
                # but here we just snapshot from the state we last saw.
                # The main loop's drain step re-validates against the
                # CURRENT local_hash before consuming.
                ckpt_revision = getattr(state, "checkpoint_revision", "") or ""
                if not ckpt_revision:
                    await asyncio.sleep(0.5)
                    continue
                if self._picker is None:
                    await asyncio.sleep(1)
                    continue
                # Cache-supply gate: when the queue is already well-fed
                # by the cache consumer, defer local gen — let the main
                # loop drain the higher-quality cache batches first.
                if (
                    self._cache is not None
                    and self._cache.enabled
                    and self._pregen_queue.qsize() >= gate_depth
                ):
                    await asyncio.sleep(2)
                    continue
                cooldown = set(state.cooldown_prompts)
                try:
                    prompt_idx = self._picker.pick(
                        self.env, cooldown, ckpt_revision, rng=rng,
                    )
                except RuntimeError:
                    await asyncio.sleep(2)
                    continue
                problem = self.env.get_problem(prompt_idx)
                # Block in a worker thread so the asyncio loop stays
                # responsive (state polling, supabase writes, queue
                # consumers). vLLM's continuous batching means a main-
                # loop ``.generate()`` running in parallel will share
                # the engine — both rollouts get batched on the GPU.
                try:
                    generations = await asyncio.to_thread(
                        self._generate_m_rollouts, problem, "",
                    )
                except Exception:
                    logger.exception("pregen: generation failed prompt=%d", prompt_idx)
                    await asyncio.sleep(1)
                    continue
                if len(generations) < M_ROLLOUTS:
                    continue
                # Pre-compute the HF sketch (hidden_states + token
                # log-probs + reward + completion_text). This is the
                # 3-5s critical-path step we want OUT of the submit
                # path so window-OPEN → POST drops to ~400ms and we
                # can actually win the proof-admission race against
                # MAX_PROOF_CANDIDATES_PER_WINDOW=32. The pregen
                # path swallows the cost during the previous window's
                # SEALING phase.
                try:
                    sketch = await asyncio.to_thread(
                        self._compute_pregen_sketch, generations, problem,
                    )
                except Exception:
                    logger.exception(
                        "pregen: sketch failed prompt=%d; falling back to "
                        "inline sketch at submit time", prompt_idx,
                    )
                    sketch = None
                # Tentatively mark as attempted so the next pregen pick
                # doesn't keep producing the same prompt while this one
                # sits in the queue. If the main loop later drops the
                # entry (stale ckpt, cooldown), the bookkeeping stays —
                # cheaper than re-rolling the same prompt.
                self._picker._attempted_by_ckpt.setdefault(
                    ckpt_revision, set()
                ).add(prompt_idx)
                try:
                    await self._pregen_queue.put(
                        (prompt_idx, problem, generations, sketch, ckpt_revision)
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("pregen: queue put failed")

                # Mirror to Supabase pregen_batches so sibling miners
                # (and our own future runs after restart) can pick it
                # up. Only write when sketch succeeded (we have rewards)
                # AND the batch landed in-zone — out-of-zone pregens
                # would zone-skip on every consumer's pre-check and
                # just clutter the table.
                if (
                    sketch is not None
                    and self._cache is not None
                    and self._cache.enabled
                ):
                    try:
                        per_rollout = sketch.get("per_rollout") or []
                        rewards = [
                            float(r.get("reward", 0.0)) for r in per_rollout
                        ]
                        if len(rewards) == M_ROLLOUTS:
                            sigma_p = _population_std(rewards)
                            k_p = sum(1 for r in rewards if r > 0.5)
                            sigma_min = (
                                self._picker.sigma_min
                                if self._picker else 0.43
                            )
                            if sigma_p >= sigma_min:
                                from reliquary.miner.persistence import (
                                    PersistedBatch,
                                )
                                serialisable_rollouts = [
                                    {
                                        "tokens": list(r["all_tokens"]),
                                        "prompt_length": int(r["prompt_length"]),
                                        "reward": float(r["reward"]),
                                    }
                                    for r in per_rollout
                                ]
                                local_n_for_save = int(
                                    getattr(state, "checkpoint_n", 0) or 0
                                )
                                batch = PersistedBatch(
                                    prompt_idx=int(prompt_idx),
                                    checkpoint_hash=ckpt_revision,
                                    local_n=local_n_for_save,
                                    sigma=float(sigma_p),
                                    k=int(k_p),
                                    rollouts=serialisable_rollouts,
                                    miner_hotkey=(
                                        self.wallet.hotkey.ss58_address
                                    ),
                                    tier="exploratory",
                                )
                                asyncio.create_task(asyncio.to_thread(
                                    self._cache.save_batch, batch,
                                ))
                                logger.info(
                                    "pregen saved prompt=%d k=%d sigma=%.3f "
                                    "→ supabase (tier=exploratory)",
                                    prompt_idx, k_p, sigma_p,
                                )
                    except Exception:
                        logger.exception(
                            "pregen: supabase save failed prompt=%d",
                            prompt_idx,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("pregen_loop iteration failed; sleeping 1s")
                await asyncio.sleep(1)

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool
    ) -> str:
        """Derive window randomness from the drand beacon (v2.3+: drand-only).

        Matches the validator's ``service._derive_randomness``: block_hash is
        no longer mixed in, so the miner does not need a substrate roundtrip
        for the GRAIL seed. The legacy ``use_drand=False`` path remains for
        offline tests and uses block_hash as a single-source seed.
        """
        if use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                None, beacon["randomness"], drand_round=beacon["round"]
            )
        block_hash = await chain.get_block_hash(subtensor, window_start)
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
