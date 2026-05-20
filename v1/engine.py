"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.
"""

from __future__ import annotations

import asyncio
import logging
import time
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
    cache=None,
    max_attempts: int = 1000,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    When the env exposes ``promising_indices()``, we sample from that
    subset (filters out the grade-school-difficulty MATH bucket that
    Qwen3 solves all-correct → σ→0 → OUT_OF_ZONE). When ``cache`` is
    provided, sampling is biased toward indices that have previously
    landed in σ-zone — see :class:`PromptSigmaCache` for the policy.

    Raises ``RuntimeError`` if no eligible prompt can be found.
    """
    rng = rng or _random

    pool: list[int] | None = None
    promising_fn = getattr(env, "promising_indices", None)
    if callable(promising_fn):
        try:
            pool = promising_fn()
        except Exception:
            pool = None

    if pool:
        if cache is not None:
            return cache.pick(pool, cooldown_prompts, rng, max_attempts=max_attempts)
        n = len(pool)
        if len(cooldown_prompts) < n / 2:
            for _ in range(max_attempts):
                idx = pool[rng.randrange(n)]
                if idx not in cooldown_prompts:
                    return idx
            raise RuntimeError("no eligible prompt found after max attempts")
        eligible = [i for i in pool if i not in cooldown_prompts]
        if not eligible:
            raise RuntimeError("no eligible prompt — promising subset exhausted")
        return rng.choice(eligible)

    # Fallback for envs without ``promising_indices`` (tests, future envs).
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


def _current_drand_round_at_send() -> int:
    """Drand quicknet round currently in progress at wall-clock now.

    Called just before POSTing /submit so the attached round matches what
    the validator sees at receipt (modulo the 1-round tolerance). Uses
    chain params cached at process start; one drand period of skew is
    tolerated by the validator.
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    return compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])


# Buffer (seconds) we leave between our send and the next drand boundary
# before falling back to ``current_round`` from the competitive
# ``current_round - 1`` claim. Covers network latency + miner/validator
# clock skew. drand quicknet period is 3 s, so 1.0 s leaves ~2 s of
# round during which we'll claim the previous round.
_DRAND_BOUNDARY_SAFETY_MARGIN = 1.0


def _competitive_drand_round_at_send() -> tuple[int, str]:
    """Pick the most-ahead-in-seal-order drand_round we can safely claim.

    The validator's batch selector iterates accepted rounds ascending,
    so a submission with ``drand_round = current - 1`` sorts *ahead* of
    one with ``drand_round = current`` in the seal-trigger ordering.
    Both are valid per the [current - 1, current] tolerance.

    We claim ``current - 1`` only when we're at least
    ``_DRAND_BOUNDARY_SAFETY_MARGIN`` seconds away from the next drand
    boundary — otherwise the validator's round may tick to ``current+1``
    by the time the POST arrives, which would put ``current - 1`` two
    rounds behind and trigger ``STALE_ROUND``.

    Returns ``(round, "current"|"previous")`` so the caller can log
    which strategy fired.
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    genesis = ci["genesis_time"]
    period = ci["period"]
    now = time.time()
    current_round = compute_current_drand_round(now, genesis, period)
    seconds_into_round = (now - genesis) % period
    if (
        current_round > 1
        and seconds_into_round < (period - _DRAND_BOUNDARY_SAFETY_MARGIN)
    ):
        return current_round - 1, "previous"
    return current_round, "current"


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
        vllm_kwargs: dict | None = None,
        initial_checkpoint_n: int = 0,
        initial_checkpoint_hash: str = "",
        sigma_filter: bool = True,
        prompt_cache=None,
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
        # vLLM construction kwargs (minus ``model``) — used by
        # _load_checkpoint to tear down and re-create the LLM instance
        # when a new HF revision is published. None disables vLLM reload:
        # tests that pass a stub generator don't need it.
        self.vllm_kwargs = vllm_kwargs
        # Seed values used as ``local_n`` / ``local_hash`` at the top of
        # mine_window. When the CLI has already snapshot_download'd and
        # loaded the validator's current revision into vllm/hf_model,
        # these stop maybe_pull_checkpoint from triggering a redundant
        # download + reload on the very first iteration.
        self._initial_checkpoint_n = initial_checkpoint_n
        self._initial_checkpoint_hash = initial_checkpoint_hash
        # When False, σ_filter still computes σ for logging but won't
        # drop the submission — every generated group goes through to
        # GRAIL + POST. Useful for seeing validator verdicts on prompts
        # that would otherwise be silently filtered.
        self._sigma_filter_enabled = sigma_filter
        # Optional per-prompt σ history. When set, pick_prompt_idx biases
        # sampling toward indices that have previously landed in σ-zone.
        # See reliquary.miner.prompt_cache.PromptSigmaCache.
        self._prompt_cache = prompt_cache
        # Pre-generation state. Token generation is independent of the
        # validator's per-window randomness (only GRAIL r_vec depends on
        # it), so we can pre-generate rollouts during validator down-time
        # and only do the sub-second commit/sign at window OPEN. The
        # protocol-level limit on cycle latency is therefore the GRAIL
        # build + sign + POST, not the multi-second model.generate().
        # We keep a *queue* of staged groups (not just one) so that when
        # OPEN fires, the main loop has multiple candidates to try in
        # case the first is now in cooldown / submitted / stale-ckpt.
        # Initialized lazily inside mine_window to avoid creating an
        # asyncio.Lock outside an event loop.
        self._pre_gen_queue: list[dict] = []
        # Match MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW so we can in
        # principle fire all 8 hotkey-slots simultaneously when OPEN
        # flips. Even if the GPU can't realistically refill 8 between
        # windows, having the queue this size means we won't bottleneck
        # on staging capacity once the GPU catches up.
        self._pre_gen_queue_max: int = 8
        self._pre_gen_lock = None
        self._last_cooldown_set: set[int] = set()
        # Pre-gen runs when (a) the validator window is NOT OPEN
        # (main is idle waiting for OPEN), or (b) window IS OPEN but
        # main has filled the batch / hit the per-hotkey cap (main is
        # sleeping). During OPEN with submission slots still available
        # we keep vLLM reserved for the main loop's time-critical cycle.
        # All three flags are set by the main loop on every iteration.
        self._window_is_open: bool = False
        self._batch_full_in_window: bool = False
        self._main_at_submission_cap: bool = False
        # vLLM's ``LLM.generate`` is not thread-safe — its internal
        # ``_run_engine`` loop drives ``llm_engine.step()`` until all
        # outstanding requests finish. Two concurrent callers from
        # different threads would both step the same engine and collect
        # each other's outputs, deadlocking. Serialize with a threading
        # lock so pre-gen and the main loop alternate cleanly.
        import threading as _threading
        self._vllm_lock = _threading.Lock()

        # Lazy imports for heavy deps — keep module import cheap.
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,  # v2.0 param kept for CLI compat; ignored
        use_drand: bool = True,
    ) -> list:
        """v2.3 multi-submission: race up to 8 distinct-prompt rollout groups
        per window from a single hotkey.

        MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8 — equal to B_BATCH, so one
        hotkey can in principle claim every slot in a window. The reference
        single-submit loop left 7 of 8 slots on the table. This loop fires
        sequential submissions until the per-hotkey cap is hit, BATCH_FILLED
        latches, or the window closes.
        """
        import httpx
        import random

        from reliquary.constants import (
            MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW, POLL_INTERVAL_SECONDS,
        )
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url, get_window_state_v2,
        )
        from reliquary.protocol.submission import RejectReason, WindowState

        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        rng = random.Random()
        results: list = []
        local_n = self._initial_checkpoint_n
        local_hash = self._initial_checkpoint_hash

        active_window_n: int = -1
        submitted_this_window: int = 0
        submitted_prompts_this_window: set[int] = set()
        batch_full: bool = False

        # asyncio primitives must be constructed inside a running loop.
        if self._pre_gen_lock is None:
            self._pre_gen_lock = asyncio.Lock()

        async with httpx.AsyncClient(timeout=30) as client:
            # Background poller that surfaces the validator's real
            # post-worker verdict for each submission — /submit's response
            # is only an enqueue ACK. Cancelled in the finally block when
            # the main loop exits (KeyboardInterrupt etc.).
            verdicts_task = asyncio.create_task(
                self._poll_verdicts_loop(url, client),
                name="reliquary-verdicts-poll",
            )
            # Pre-generation worker: keeps one rollout group ready so the
            # main cycle only needs GRAIL + sign + POST at window OPEN.
            # Token sampling is independent of the validator's per-window
            # randomness — only the GRAIL r_vec depends on it.
            pre_gen_task = asyncio.create_task(
                self._pre_gen_loop(),
                name="reliquary-pre-gen",
            )
            try:
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
                        local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
                            state=state, local_n=local_n, local_hash=local_hash,
                            local_model=self.hf_model,
                            download_fn=_hf_download,
                            load_fn=self._load_checkpoint,
                        )
                    except Exception:
                        logger.exception("checkpoint pull failed; keeping local")

                    if state.window_n != active_window_n:
                        active_window_n = state.window_n
                        submitted_this_window = 0
                        submitted_prompts_this_window = set()
                        batch_full = False

                    self._window_is_open = state.state == WindowState.OPEN
                    self._batch_full_in_window = batch_full
                    self._main_at_submission_cap = (
                        submitted_this_window >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
                    )

                    if state.state != WindowState.OPEN:
                        # Window-OPEN boundary is the time-critical race:
                        # other miners on the same subnet fire their POSTs
                        # within ~100-500 ms of the validator flipping to
                        # OPEN. With ~120 ms US↔EU RTT to validator, each
                        # /state poll already costs the RTT — sleeping
                        # 50 ms between polls keeps us at ~5-8 polls/sec,
                        # i.e. avg detection lag ≈ RTT/2 + 25 ms.
                        await asyncio.sleep(0.05)
                        continue

                    if batch_full or submitted_this_window >= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW:
                        await asyncio.sleep(0.5)
                        continue

                    randomness = state.randomness
                    if not randomness:
                        # Window is OPEN but the validator hasn't yet
                        # finished _set_window_randomness. Typically <1 s
                        # — poll as fast as feasible so we fire the
                        # moment randomness lands.
                        await asyncio.sleep(0.03)
                        continue

                    skip = set(state.cooldown_prompts) | submitted_prompts_this_window
                    # Expose cooldown to the pre-gen worker so it can pick
                    # a prompt the main loop won't have to discard.
                    self._last_cooldown_set = set(state.cooldown_prompts)

                    # Drain the pre-gen queue and fire all candidates in
                    # parallel. With sub-second per-submission latency
                    # serialized, we lose the batch-filled race against
                    # miners who pipe many POSTs simultaneously. asyncio
                    # gather lets all enqueued submissions hit the
                    # validator within ~50 ms of each other instead of
                    # ~1 s sequential.
                    pre_gens_to_submit: list[dict] = []
                    remaining = MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW - submitted_this_window
                    while remaining > 0:
                        pg = await self._take_pre_gen(
                            skip_prompts=skip | {p["prompt_idx"] for p in pre_gens_to_submit},
                            expected_local_hash=local_hash,
                        )
                        if pg is None:
                            break
                        pre_gens_to_submit.append(pg)
                        remaining -= 1

                    if pre_gens_to_submit:
                        tasks = [
                            asyncio.create_task(self._attempt_one_submission(
                                url=url,
                                client=client,
                                prompt_idx=pg["prompt_idx"],
                                randomness=randomness,
                                local_hash=local_hash,
                                window_n=state.window_n,
                                local_n=local_n,
                                pre_gen_generations=pg["generations"],
                                pre_gen_problem=pg["problem"],
                            ))
                            for pg in pre_gens_to_submit
                        ]
                        responses = await asyncio.gather(*tasks, return_exceptions=True)
                        for pg, resp in zip(pre_gens_to_submit, responses):
                            if isinstance(resp, Exception) or resp is None:
                                continue
                            submitted_prompts_this_window.add(pg["prompt_idx"])
                            submitted_this_window += 1
                            results.append(resp)
                            if not resp.accepted and resp.reason == RejectReason.BATCH_FILLED:
                                batch_full = True
                        continue  # back to /state poll to see batch-filled state

                    # No pre-gens staged: fall back to the slow main-loop
                    # gen path. One submission this iteration; pre-gen
                    # will refill the queue while we run.
                    try:
                        prompt_idx = pick_prompt_idx(
                            self.env, skip, rng=rng, cache=self._prompt_cache,
                        )
                    except RuntimeError:
                        logger.info("no eligible prompt (cooldown ∪ already-submitted); sleeping")
                        await asyncio.sleep(5)
                        continue

                    resp = await self._attempt_one_submission(
                        url=url,
                        client=client,
                        prompt_idx=prompt_idx,
                        randomness=randomness,
                        local_hash=local_hash,
                        window_n=state.window_n,
                        local_n=local_n,
                        pre_gen_generations=None,
                        pre_gen_problem=None,
                    )
                    if resp is None:
                        continue

                    submitted_prompts_this_window.add(prompt_idx)
                    submitted_this_window += 1
                    results.append(resp)

                    if not resp.accepted and resp.reason == RejectReason.BATCH_FILLED:
                        batch_full = True
            finally:
                verdicts_task.cancel()
                pre_gen_task.cancel()

        return results

    async def _pre_gen_loop(self) -> None:
        """Pre-generate one rollout group while the validator is idle.

        Token sampling only depends on the model checkpoint, not the
        validator's per-window randomness — so we can generate rollouts
        in advance and just plug in the actual randomness for the GRAIL
        commit + signature at window OPEN. Each cycle of the main loop
        consumes the stored group, then this worker refills it. vLLM's
        continuous batcher schedules main + pre-gen requests on the same
        GPU at near-zero extra latency.

        The stored group is tagged with the checkpoint hash it was
        generated against, so a checkpoint advance invalidates stale
        pre-gens via ``_invalidate_pre_gen`` from ``_load_checkpoint``.
        """
        import random as _random

        from reliquary.constants import M_ROLLOUTS

        rng = _random.Random()
        while True:
            try:
                # Pre-gen bails ONLY when the window is OPEN AND the main
                # loop is actively trying to land submissions. When main
                # is in its idle-sleep paths inside OPEN (batch already
                # filled, or per-hotkey submission cap reached) the vLLM
                # lock is free and pre-gen can use it to stage groups
                # for the NEXT window. Without this relaxation, pre-gen
                # never fires on busy subnets where OPEN is continuous —
                # losing every batch_filled race because we never have a
                # group ready at the OPEN flip.
                main_in_critical_path = (
                    self._window_is_open
                    and not self._batch_full_in_window
                    and not self._main_at_submission_cap
                )
                if main_in_critical_path:
                    await asyncio.sleep(0.5)
                    continue

                async with self._pre_gen_lock:
                    queue_full = len(self._pre_gen_queue) >= self._pre_gen_queue_max
                if queue_full:
                    await asyncio.sleep(1.0)
                    continue

                ckpt_at_start = getattr(self, "_loaded_checkpoint_path", None)
                if ckpt_at_start is None or self.vllm_model is None:
                    await asyncio.sleep(1.0)
                    continue

                skip = set(self._last_cooldown_set)
                # Don't re-pick prompts that are already staged in the queue.
                async with self._pre_gen_lock:
                    for entry in self._pre_gen_queue:
                        skip.add(entry["prompt_idx"])
                try:
                    prompt_idx = pick_prompt_idx(
                        self.env, skip, rng=rng, cache=self._prompt_cache,
                    )
                except RuntimeError:
                    await asyncio.sleep(5.0)
                    continue

                problem = self.env.get_problem(prompt_idx)
                logger.info(
                    "pre-gen: starting gen prompt=%d (window_open=%s batch_full=%s cap_hit=%s)",
                    prompt_idx, self._window_is_open,
                    self._batch_full_in_window, self._main_at_submission_cap,
                )
                # Run vLLM gen in a worker thread — it's a blocking
                # synchronous call. Pass an empty randomness because the
                # token sampling path doesn't read it (only GRAIL does,
                # and GRAIL runs later at submission time).
                generations = await asyncio.to_thread(
                    self._generate_m_rollouts, problem, "",
                )
                if len(generations) < M_ROLLOUTS:
                    # Treat length-finish as σ=0 in the cache so prompts
                    # that consistently exceed 8192 tokens accumulate
                    # toward ban_after_n_zero and get filtered out of
                    # future sampling. Without this the cache only sees
                    # "successful" gens and never learns about the
                    # too-hard tail.
                    if self._prompt_cache is not None:
                        self._prompt_cache.record(prompt_idx, 0.0)
                    logger.info(
                        "pre-gen: dropped prompt=%d (got %d/%d rollouts — "
                        "likely length-finish; recorded as σ=0)",
                        prompt_idx, len(generations), M_ROLLOUTS,
                    )
                    continue

                # If a checkpoint advance happened while we were
                # generating, the work is invalid.
                if getattr(self, "_loaded_checkpoint_path", None) != ckpt_at_start:
                    logger.info(
                        "pre-gen: dropped prompt=%d (checkpoint advanced during gen)",
                        prompt_idx,
                    )
                    continue

                # σ pre-screen: pre-gen σ-filter follows the CLI flag.
                # When ``--sigma-filter`` is on, drop σ-low groups (they
                # would only burn a submission slot). When ``--no-sigma-
                # filter`` is on (visibility mode), keep staging them so
                # the queue stays warm — empty-queue at OPEN means main
                # loop has to do fresh gens which lose the race. Cache
                # still records σ regardless.
                rewards = self._annotate_rewards(generations, problem)
                sigma = self._group_sigma(rewards)
                if self._prompt_cache is not None:
                    self._prompt_cache.record(prompt_idx, sigma, int(sum(rewards)))
                if sigma < SIGMA_MIN and self._sigma_filter_enabled:
                    logger.info(
                        "pre-gen: σ-drop prompt=%d sigma=%.3f rewards_sum=%.1f/8 (not staged)",
                        prompt_idx, sigma, sum(rewards),
                    )
                    continue

                # Pre-compute the GRAIL forward passes now (during
                # validator idle) so the at-submission cycle is
                # r_vec → sketch → sign → POST only. The 8 hf_model
                # passes are the second-biggest cost in the cycle after
                # token gen; moving them here shaves ~1-2 s.
                try:
                    await asyncio.to_thread(
                        self._precompute_grail_proof_inputs, generations,
                    )
                except Exception:
                    logger.exception(
                        "pre-gen GRAIL precompute failed for prompt=%d; "
                        "dropping (main path will recompute on the fly)",
                        prompt_idx,
                    )
                    continue

                # Final freshness check — if hf_model was swapped while
                # we were computing, the cached hidden_states are stale.
                if getattr(self, "_loaded_checkpoint_path", None) != ckpt_at_start:
                    logger.info(
                        "pre-gen: dropped prompt=%d (checkpoint advanced "
                        "during GRAIL precompute)",
                        prompt_idx,
                    )
                    for gen in generations:
                        gen.pop("_cached_hidden_states", None)
                        gen.pop("_cached_token_logprobs", None)
                        gen.pop("_cached_buckets_f", None)
                    continue

                async with self._pre_gen_lock:
                    if len(self._pre_gen_queue) < self._pre_gen_queue_max:
                        self._pre_gen_queue.append({
                            "prompt_idx": prompt_idx,
                            "generations": generations,
                            "problem": problem,
                            "checkpoint_path": ckpt_at_start,
                            "sigma": sigma,
                        })
                        logger.info(
                            "pre-gen ready: prompt=%d sigma=%.3f (%d rollouts) "
                            "queue=%d/%d",
                            prompt_idx, sigma, len(generations),
                            len(self._pre_gen_queue), self._pre_gen_queue_max,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("pre-gen err: %s", e)
                await asyncio.sleep(1.0)

    async def _take_pre_gen(
        self,
        *,
        skip_prompts: set[int],
        expected_local_hash: str,
    ) -> dict | None:
        """Consume the first usable staged group, or None.

        Walks the queue from head to tail; the first entry whose prompt
        isn't skipped and whose checkpoint matches the active one is
        popped and returned. Stale entries encountered along the way
        are discarded (their GPU tensors freed).
        """
        active_ckpt = getattr(self, "_loaded_checkpoint_path", None)
        async with self._pre_gen_lock:
            i = 0
            while i < len(self._pre_gen_queue):
                entry = self._pre_gen_queue[i]
                if entry["prompt_idx"] in skip_prompts:
                    self._free_cached_grail_tensors(entry)
                    self._pre_gen_queue.pop(i)
                    continue
                if entry.get("checkpoint_path") != active_ckpt:
                    self._free_cached_grail_tensors(entry)
                    self._pre_gen_queue.pop(i)
                    continue
                # Hit — pop and return.
                self._pre_gen_queue.pop(i)
                return entry
            return None

    @staticmethod
    def _free_cached_grail_tensors(entry: dict) -> None:
        """Drop GPU tensors that were cached at pre-gen time on each rollout."""
        for gen in entry.get("generations", []):
            gen.pop("_cached_hidden_states", None)
            gen.pop("_cached_token_logprobs", None)
            gen.pop("_cached_buckets_f", None)

    def _invalidate_pre_gen(self) -> None:
        """Drop all staged pre-gens — called when the checkpoint advances.

        Explicitly nukes the cached GPU tensors on each staged rollout
        so they're eligible for the ``torch.cuda.empty_cache()`` call
        later in ``_load_checkpoint`` — without this they'd stay
        reachable through the dict reference until Python GC runs,
        wasting GPU memory during the reload.
        """
        for entry in self._pre_gen_queue:
            self._free_cached_grail_tensors(entry)
        self._pre_gen_queue.clear()

    async def _poll_verdicts_loop(self, url: str, client) -> None:
        """Stream the validator's real post-worker verdicts to the log.

        /submit's response is only an enqueue ACK ("submitted") — the actual
        ACCEPTED / GRAIL_FAIL / WRONG_RANDOMNESS / etc. verdict lands
        seconds later via GET /verdicts/{hotkey}. Without this poller the
        miner can only see what was queued, not what passed. Runs as a
        background task; cancelled when mine_window exits.
        """
        from reliquary.miner.submitter import get_verdicts_v2

        hotkey = self.wallet.hotkey.ss58_address
        # Start from now so each miner restart doesn't replay the full
        # validator-side verdict history (which is noisy and only
        # confuses post-mortem analysis of recent runs).
        since = time.time()
        while True:
            try:
                resp = await get_verdicts_v2(url, hotkey, since, client=client)
                for v in resp.verdicts:
                    reason = v.reason.value if hasattr(v.reason, "value") else v.reason
                    if v.accepted:
                        logger.info(
                            "verdict ACCEPTED win=%s mr=%s",
                            v.window_n, v.merkle_root[:12],
                        )
                    else:
                        logger.warning(
                            "verdict REJECTED win=%s mr=%s reason=%s",
                            v.window_n, v.merkle_root[:12], reason,
                        )
                    since = max(since, v.ts)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("verdicts poll failed: %s", e)
            await asyncio.sleep(5)

    async def _attempt_one_submission(
        self,
        *,
        url: str,
        client,
        prompt_idx: int,
        randomness: str,
        local_hash: str,
        window_n: int,
        local_n: int,
        pre_gen_generations: list[dict] | None = None,
        pre_gen_problem: dict | None = None,
    ):
        """One generate → freshness re-check → sign → POST attempt.

        When pre-generated rollouts are supplied, vLLM gen is skipped —
        the cycle becomes GRAIL build + sign + POST (~sub-second). The
        validator's per-window randomness only seeds the GRAIL r_vec,
        not token sampling, so tokens generated in advance against the
        same checkpoint are still valid.

        Returns the BatchSubmissionResponse on a completed POST, or None
        if the attempt was aborted before submit (state drifted under
        us, generation underran).
        """
        import os

        from reliquary.constants import M_ROLLOUTS
        from reliquary.miner.submitter import (
            SubmissionError, get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import (
            BatchSubmissionRequest, WindowState,
        )

        if pre_gen_generations is not None and pre_gen_problem is not None:
            problem = pre_gen_problem
            generations = pre_gen_generations
            logger.info(
                "using pre-generated rollouts for prompt=%d", prompt_idx,
            )
        else:
            problem = self.env.get_problem(prompt_idx)
            # Run vLLM gen in a worker thread so the asyncio loop stays
            # free to drive other tasks — most importantly the pre-gen
            # worker, which needs the loop to fire its next iteration
            # while a main-loop gen is still in flight. Without this,
            # one slow main-loop gen freezes pre-gen for its full
            # duration (often 5+ minutes).
            generations = await asyncio.to_thread(
                self._generate_m_rollouts, problem, randomness,
            )
        if len(generations) < M_ROLLOUTS:
            # Length-finish drop: mark prompt as bad in the cache so it
            # accumulates toward ban_after_n_zero. Same reasoning as
            # the pre-gen path — keeps the cache honest about prompts
            # that exceed 8192 tokens at this checkpoint.
            if self._prompt_cache is not None and pre_gen_generations is None:
                self._prompt_cache.record(prompt_idx, 0.0)
            logger.warning(
                "generated %d/%d for prompt %d; skipping (length-finish; "
                "recorded as σ=0 in cache)",
                len(generations), M_ROLLOUTS, prompt_idx,
            )
            return None

        # Local σ filter: the validator's OUT_OF_ZONE check is
        # ``σ < SIGMA_MIN``, and OUT_OF_ZONE still burns one of our
        # 8 per-window submission slots (rate-limit increments before
        # the worker even runs). Computing σ here is just decoding +
        # env.compute_reward × 8 — sub-second — and lets us bail
        # before consuming a slot on a prompt the validator will
        # certainly reject.
        rewards = self._annotate_rewards(generations, problem)
        sigma = self._group_sigma(rewards)
        # Record the σ outcome once per group so future sampling can bias
        # toward zone-hitters. We record here (not earlier) so pre-gen
        # hits are recorded on commit rather than on stage, which keeps
        # per-(prompt, ckpt) record counts honest with how many submission
        # slots they actually burned.
        if self._prompt_cache is not None and pre_gen_generations is None:
            self._prompt_cache.record(prompt_idx, sigma, int(sum(rewards)))
        if sigma < SIGMA_MIN:
            if self._sigma_filter_enabled:
                logger.info(
                    "σ-skip prompt=%d sigma=%.3f rewards_sum=%.1f/%d (below SIGMA_MIN=%.2f)",
                    prompt_idx, sigma, sum(rewards), len(rewards), SIGMA_MIN,
                )
                return None
            logger.info(
                "σ-low (filter off) prompt=%d sigma=%.3f rewards_sum=%.1f/%d — submitting anyway",
                prompt_idx, sigma, sum(rewards), len(rewards),
            )
        else:
            logger.info(
                "σ-IN-ZONE prompt=%d sigma=%.3f rewards_sum=%.1f/%d — submitting (ACCEPTED candidate)",
                prompt_idx, sigma, sum(rewards), len(rewards),
            )

        rollout_submissions = [
            self._build_rollout_submission(gen, problem, randomness)
            for gen in generations
        ]
        merkle_root = _compute_merkle_root(rollout_submissions)

        # Skip pre-POST /state re-check on pre-gen path. The pre-gen
        # queue already validated checkpoint_path at consumption time;
        # window-advance / randomness-change risk is small relative to
        # the ~100-200 ms /state round-trip cost — which puts us behind
        # other miners in the batch_filled race. Worst case: we POST
        # to a closed window, validator returns NO_ACTIVE_WINDOW, we
        # log and move on. Same outcome, just one fewer round trip.
        # For the slow main-loop gen path (200-500 s of gen) the
        # freshness re-check is still worth it because state really
        # could have drifted during the gen.
        if pre_gen_generations is None:
            try:
                fresh = await get_window_state_v2(url, client=client)
            except Exception as e:
                logger.warning("pre-POST /state re-check failed (%s); skipping", e)
                return None

            if fresh.state != WindowState.OPEN:
                logger.info(
                    "window closed during gen (prompt=%d, state=%s); dropping",
                    prompt_idx, fresh.state,
                )
                return None
            if fresh.window_n != window_n:
                logger.info(
                    "window advanced %d→%d during gen; dropping prompt=%d",
                    window_n, fresh.window_n, prompt_idx,
                )
                return None
            if fresh.checkpoint_n != local_n or fresh.randomness != randomness:
                logger.info(
                    "checkpoint/randomness advanced during gen (ckpt %d→%d); "
                    "dropping prompt=%d",
                    local_n, fresh.checkpoint_n, prompt_idx,
                )
                return None
            if prompt_idx in set(fresh.cooldown_prompts):
                logger.info(
                    "prompt %d entered cooldown during gen; dropping",
                    prompt_idx,
                )
                return None

        # Compute the drand round at the latest possible moment — the
        # validator gates with zero tolerance on FUTURE_ROUND and one
        # round on STALE_ROUND. We claim ``current - 1`` when we can
        # (the validator iterates rounds ascending at seal time, so the
        # earlier-claimed round sorts ahead of submissions claiming the
        # current round).
        current_round, _drand_strategy = _competitive_drand_round_at_send()
        nonce = os.urandom(16).hex()
        envelope_sig = sign_envelope(
            wallet=self.wallet,
            miner_hotkey=self.wallet.hotkey.ss58_address,
            window_start=window_n,
            prompt_idx=prompt_idx,
            merkle_root=merkle_root,
            checkpoint_hash=local_hash,
            drand_round=current_round,
            randomness=randomness,
            nonce=nonce,
        ).hex()

        request = BatchSubmissionRequest(
            miner_hotkey=self.wallet.hotkey.ss58_address,
            prompt_idx=prompt_idx,
            window_start=window_n,
            merkle_root=merkle_root,
            rollouts=rollout_submissions,
            checkpoint_hash=local_hash,
            drand_round=current_round,
            nonce=nonce,
            envelope_signature=envelope_sig,
        )

        try:
            resp = await submit_batch_v2(url, request, client=client)
        except SubmissionError as exc:
            logger.error("submit failed for prompt=%d: %s", prompt_idx, exc)
            return None

        reason_str = resp.reason.value if hasattr(resp.reason, "value") else resp.reason
        logger.info(
            "submitted window=%d prompt=%d drand_round=%d(%s) accepted=%s reason=%s",
            window_n, prompt_idx, current_round, _drand_strategy,
            resp.accepted, reason_str,
        )
        return resp

    def _load_checkpoint(self, local_path: str):
        """Reload hf_model and rebuild the vLLM engine from *local_path*.

        On checkpoint change we have to recreate the vLLM ``LLM`` instance
        — vLLM doesn't expose a hot weight-swap in all versions, so the
        cleanest, most version-portable path is teardown + rebuild from
        ``self.vllm_kwargs``. The cost is ~30-60 s of generation downtime
        per checkpoint roll (steady-state cadence ~20-30 min), which is
        well under one window.

        Order on single-GPU is important: tear down vLLM first (free
        most VRAM), then swap hf_model, then re-init vLLM so it sizes
        its KV cache against the remaining headroom.
        """
        import gc
        import os
        import time
        import torch
        from transformers import AutoModelForCausalLM

        from reliquary.constants import ATTN_IMPLEMENTATION

        loaded = getattr(self, "_loaded_checkpoint_path", None)
        # Compare via realpath — snapshot_download with vs without
        # allow_patterns can return logically-equivalent paths that
        # differ as strings (symlinks, trailing slashes).
        try:
            same = loaded and os.path.realpath(loaded) == os.path.realpath(local_path)
        except OSError:
            same = loaded == local_path
        if same:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)
        # Any rollouts staged by the pre-gen worker were generated
        # against the previous checkpoint and are no longer valid for
        # GRAIL verification — drop them now.
        self._invalidate_pre_gen()

        # Fast path: in-place vLLM weight reload via collective_rpc.
        # ``reload_weights(weights_path=...)`` swaps tensors layer-by-layer
        # inside the existing EngineCore — no subprocess teardown, no CUDA
        # graph recapture, no KV cache discard. ~5 s vs ~45 s for full
        # rebuild. We still reload hf_model the slow way (no equivalent
        # in-place API). On any failure we fall through to the full
        # teardown path below.
        if self.vllm_model is not None:
            try:
                self.vllm_model.collective_rpc(
                    "reload_weights",
                    kwargs={
                        "weights_path": local_path,
                        "is_checkpoint_format": True,
                    },
                )
            except Exception:
                logger.exception(
                    "vLLM in-place reload_weights failed; falling back "
                    "to full teardown + rebuild."
                )
            else:
                # In-place succeeded — swap hf_model only, return early.
                try:
                    new_hf = AutoModelForCausalLM.from_pretrained(
                        local_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=ATTN_IMPLEMENTATION,
                    ).to(f"cuda:{self.proof_gpu}").eval()
                except Exception:
                    logger.exception(
                        "hf_model reload failed after in-place vLLM swap; "
                        "GRAIL proofs will be inconsistent until next pull.",
                    )
                    return self.hf_model
                old_hf = self.hf_model
                self.hf_model = new_hf
                del old_hf
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self._loaded_checkpoint_path = local_path
                logger.info(
                    "Checkpoint %s loaded (in-place vLLM reload + hf_model swap)",
                    local_path,
                )
                return self.hf_model

        old_vllm = self.vllm_model
        self.vllm_model = None
        # vLLM's ``LLM`` has no ``__del__`` — the spawned EngineCore
        # subprocess will keep holding GPU memory after the Python
        # reference is dropped. Call its shutdown explicitly so the
        # subprocess exits and releases its KV-cache allocation before
        # we try to load the new model.
        if old_vllm is not None:
            try:
                old_vllm.llm_engine.engine_core.shutdown(timeout=10.0)
            except Exception:
                logger.exception("old vLLM shutdown raised; continuing teardown")
        del old_vllm
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            new_hf = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.proof_gpu}").eval()
        except Exception:
            logger.exception(
                "Failed to reload hf_model from %s; vLLM is also torn "
                "down. Miner is BROKEN until the next successful pull.",
                local_path,
            )
            return self.hf_model

        old_hf = self.hf_model
        self.hf_model = new_hf
        del old_hf
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        if self.vllm_kwargs is None:
            logger.warning(
                "_load_checkpoint: vllm_kwargs not set — cannot rebuild "
                "vLLM. Generation will be broken until miner restart."
            )
            self._loaded_checkpoint_path = None
            return self.hf_model

        # Give the old EngineCore subprocess time to fully release its
        # GPU allocations — without this gap, vLLM's "free memory on
        # device" check at startup fires and the reload fails with
        # ValueError even though most of the released memory is just
        # CUDA-allocator cache that will recompact within a second or two.
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        time.sleep(2.0)
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        from vllm import LLM
        kwargs = dict(self.vllm_kwargs)
        kwargs["model"] = local_path

        last_exc: Exception | None = None
        # Reload retries with progressively smaller KV cache budgets — if
        # the first attempt fails because the OS hasn't reclaimed the
        # old EngineCore's GPU yet, dropping gpu_memory_utilization buys
        # headroom without restarting the miner.
        for mem_frac in (kwargs.get("gpu_memory_utilization", 0.5), 0.35, 0.25):
            kwargs["gpu_memory_utilization"] = mem_frac
            try:
                self.vllm_model = LLM(**kwargs)
                # Update kwargs so subsequent reloads start from the value
                # that actually worked this time, not the original.
                self.vllm_kwargs = dict(kwargs)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                logger.warning(
                    "vLLM re-init at gpu_memory_utilization=%.2f failed: %s",
                    mem_frac, str(e)[:200],
                )
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(2.0)

        if last_exc is not None:
            logger.exception(
                "vLLM re-init failed at all memory fractions; generation "
                "BROKEN until next successful pull. Last error: %s",
                last_exc,
            )
            self.vllm_model = None
            self._loaded_checkpoint_path = None
            return self.hf_model

        self._loaded_checkpoint_path = local_path
        logger.info("Checkpoint %s loaded (hf_model + vLLM)", local_path)
        return self.hf_model

    def _annotate_rewards(self, generations: list[dict], problem) -> list[float]:
        """Decode each rollout and stash its reward back on the generation
        dict so :py:meth:`_build_rollout_submission` doesn't re-decode the
        same tokens later. Returns the reward list (in the same order).
        """
        rewards: list[float] = []
        for gen in generations:
            tokens = gen["tokens"]
            prompt_length = gen["prompt_length"]
            completion_text = self.tokenizer.decode(tokens[prompt_length:])
            reward = self.env.compute_reward(problem, completion_text)
            gen["_cached_reward"] = reward
            rewards.append(reward)
        return rewards

    @staticmethod
    def _group_sigma(rewards: list[float]) -> float:
        """Population standard deviation of the rollout-group rewards.

        Matches the validator's OUT_OF_ZONE check: ``σ < SIGMA_MIN``
        rejects. For binary {0, 1} rewards on 8 samples this is
        equivalent to fewer than 2 (or more than 6) correct.
        """
        import statistics
        if len(rewards) < 2:
            return 0.0
        return statistics.pstdev(rewards)

    def _generate_m_rollouts(self, problem, randomness) -> list[dict]:
        """Generate M_ROLLOUTS completions at T_PROTO via vLLM.

        Uses ``SamplingParams(n=M_ROLLOUTS)`` so vLLM does the prompt
        prefill once and emits M independent samples — continuous-batched
        across the rollouts and (when other generations are in flight)
        across requests. The rollout-token format matches the prior HF
        path: each entry is ``{"tokens": prompt + completion, "prompt_length": P}``,
        with the EOS token appended when generation stopped on it
        (vLLM omits stop tokens from ``token_ids`` by default but the
        validator's BAD_TERMINATION check expects EOS terminal).
        """
        import time as _time
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        sp = SamplingParams(
            n=M_ROLLOUTS,
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            # HF's ``top_k=0`` means "disable top-k filtering"; vLLM
            # uses ``-1`` for the same semantic. Translate to be safe.
            top_k=-1 if TOP_K_PROTO == 0 else TOP_K_PROTO,
            max_tokens=self.max_new_tokens,
        )
        prompt_id = problem.get("id", "")[:10]
        logger.info(
            "gen start prompt_id=%s prompt_len=%d max_new=%d",
            prompt_id, prompt_length, self.max_new_tokens,
        )
        t0 = _time.perf_counter()
        # Serialize against concurrent gen from the pre-gen worker —
        # see _vllm_lock comment in __init__ for the deadlock story.
        with self._vllm_lock:
            outputs = self.vllm_model.generate(
                [TokensPrompt(prompt_token_ids=prompt_tokens)],
                sp,
                use_tqdm=False,
            )
        elapsed = _time.perf_counter() - t0
        out_lens = [len(o.token_ids) for o in outputs[0].outputs]
        finish_reasons = [o.finish_reason for o in outputs[0].outputs]
        logger.info(
            "gen done prompt_id=%s elapsed=%.1fs lens=%s finish=%s",
            prompt_id, elapsed, out_lens, finish_reasons,
        )
        ro = outputs[0]
        eos = self.tokenizer.eos_token_id
        # If any rollout hit the max_tokens length cap, the group is a
        # guaranteed BAD_TERMINATION reject at the validator (per-rollout
        # last-token-is-EOS check). Returning [] lets the caller skip the
        # submission slot entirely instead of burning it on a doomed
        # group. We still log the group so the user sees that gen
        # happened and what shape it was.
        if any(c.finish_reason == "length" for c in ro.outputs):
            return []
        rollouts = []
        for completion in ro.outputs:
            gen = list(completion.token_ids)
            if completion.finish_reason == "stop" and (not gen or gen[-1] != eos):
                gen.append(eos)
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
            })
        return rollouts

    def _build_rollout_submission(self, generation, problem, randomness):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit."""
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        # Use the cached reward stashed by ``_annotate_rewards`` to avoid
        # decoding the same tokens + recomputing the env reward twice.
        if "_cached_reward" in generation:
            reward = generation["_cached_reward"]
        else:
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

    def _precompute_grail_proof_inputs(self, generations: list[dict]) -> None:
        """Run the hf_model forward pass on each rollout and cache the
        randomness-independent outputs on the generation dict.

        Three randomness-independent quantities are cached per rollout:

        - ``_cached_hidden_states``: GRAIL layer activations.
        - ``_cached_token_logprobs``: per-completion-token log-probs.
        - ``_cached_buckets_f``: ``[seq_len, topk]`` float32 bucket matrix
          ready for a single dot product with ``r_vec`` at submission
          time (the only randomness-dependent step). This moves the
          per-token topk + sort + bucketize work — the dominant cost in
          GRAIL commit construction — out of the time-critical OPEN race.

        The at-submission cycle then collapses to:
          ``r_vec = prf(randomness)`` (microseconds) →
          ``sketches = buckets_f @ r_vec_f`` (one matvec per rollout) →
          ``sign_commit_binding`` (one sr25519 sig per rollout) → POST.

        On checkpoint advance ``_invalidate_pre_gen`` clears the staged
        group, releasing these GPU tensors before the new hf_model loads.
        """
        import torch
        from reliquary.shared.forward import forward_single_layer
        from reliquary.protocol.grail_verifier import log_magnitude_bucket_vectorized

        for gen in generations:
            if (
                "_cached_hidden_states" in gen
                and "_cached_token_logprobs" in gen
                and "_cached_buckets_f" in gen
            ):
                continue
            all_tokens: list[int] = gen["tokens"]
            prompt_length: int = gen["prompt_length"]

            proof_input = torch.tensor(
                [all_tokens], device=f"cuda:{self.proof_gpu}",
            )
            with torch.no_grad():
                hidden_states, logits = forward_single_layer(
                    self.hf_model, proof_input, None, LAYER_INDEX,
                )
            hidden_states = hidden_states[0]  # [seq_len, hidden_dim]
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
            token_logprobs: list[float] = []
            for i in range(prompt_length, len(all_tokens)):
                token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

            # Pre-compute the randomness-independent half of the GRAIL
            # commitment: per-token topk on |h_layer|, sorted indices,
            # signed-value gather, magnitude bucketize. Result is a
            # float32 [seq_len, topk] matrix that's ready for the final
            # matvec against r_vec at submission time.
            with torch.no_grad():
                abs_h = hidden_states.abs()
                _, topk_indices = torch.topk(
                    abs_h, k=self._verifier.topk, dim=1,
                )
                topk_indices, _ = torch.sort(topk_indices, dim=1)
                signed_values = torch.gather(
                    hidden_states, dim=1, index=topk_indices,
                )
                buckets = log_magnitude_bucket_vectorized(
                    signed_values, self._verifier.num_buckets,
                )
                buckets_f = buckets.to(torch.float32)

            gen["_cached_hidden_states"] = hidden_states.detach()
            gen["_cached_token_logprobs"] = token_logprobs
            gen["_cached_buckets_f"] = buckets_f.detach()

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Construct a GRAIL proof commit dict from a generation dict.

        Reproduces the proof construction:
          - HF forward pass for hidden_states + logits (skipped when
            ``_cached_hidden_states`` / ``_cached_token_logprobs`` are
            present, which is the pre-gen fast path)
          - Commitment batch via GRAILVerifier (uses randomness)
          - log-softmax token log-probs
          - Signature via sign_commit_binding (uses randomness)
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        cached_buckets_f = generation.get("_cached_buckets_f")
        if (
            "_cached_hidden_states" in generation
            and "_cached_token_logprobs" in generation
        ):
            hidden_states = generation["_cached_hidden_states"]
            token_logprobs = generation["_cached_token_logprobs"]
        else:
            proof_input = torch.tensor(
                [all_tokens], device=f"cuda:{self.proof_gpu}"
            )
            with torch.no_grad():
                hidden_states, logits = forward_single_layer(
                    self.hf_model, proof_input, None, LAYER_INDEX
                )
            hidden_states = hidden_states[0]  # [seq_len, hidden_dim]
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
            token_logprobs = []
            for i in range(prompt_length, len(all_tokens)):
                token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        # Build commitments. Fast path: when ``_cached_buckets_f`` is
        # present (pre-gen has done the topk+sort+bucketize ahead of
        # time), we only need the single ``buckets_f @ r_vec_f`` matvec.
        # Slow path: full create_commitments_batch.
        r_vec = self._verifier.generate_r_vec(randomness)
        if cached_buckets_f is not None:
            from reliquary.protocol.grail_verifier import PRIME_Q
            r_vec_f = r_vec.to(torch.float32).to(cached_buckets_f.device)
            sketches = (cached_buckets_f @ r_vec_f).to(torch.int64)
            sketch_vals = [int(s) % PRIME_Q for s in sketches.tolist()]
            commitments = [{"sketch": s} for s in sketch_vals]
        else:
            commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        # Sign (depends on randomness)
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
