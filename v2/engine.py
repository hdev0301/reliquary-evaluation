"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

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


def _is_vllm_instance(obj: Any) -> bool:
    """Duck-check whether *obj* is a ``vllm.LLM`` (avoids importing vllm)."""
    cls = type(obj)
    return cls.__name__ == "LLM" and cls.__module__.startswith("vllm")

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
    blacklist: set[int] | None = None,
) -> int:
    """Pick a random prompt index not in cooldown or the difficulty blacklist.

    Uniform-random selection with rejection sampling. ``blacklist`` lets
    the miner remember prompts that recently went OUT_OF_ZONE (all-correct
    or all-incorrect under the current policy) so it stops wasting compute
    re-trying them every window.

    Raises ``RuntimeError`` if no eligible prompt can be found.
    """
    rng = rng or _random
    n = len(env)
    skip = cooldown_prompts if blacklist is None else cooldown_prompts | blacklist
    # Fast path: small skip set → reject sample is cheap and avoids the
    # O(n) eligible-list build (n is 880k+ for OpenMathInstruct).
    if len(skip) < n / 2:
        for _ in range(max_attempts):
            idx = rng.randrange(n)
            if idx not in skip:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(n) if i not in skip]
    if not eligible:
        raise RuntimeError("no eligible prompt — env fully in cooldown / blacklist")
    return rng.choice(eligible)


def _rewards_std(rewards: list[float]) -> float:
    """Population std — must match validator's ``rewards_std`` exactly."""
    n = len(rewards)
    if n < 2:
        return 0.0
    mean = sum(rewards) / n
    return (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5


@dataclasses.dataclass
class _PreGenEntry:
    """A submission that's been pre-generated and pre-proven during the
    non-OPEN phase, ready to be finalized with the next window's randomness
    and POSTed the moment the validator flips to OPEN.

    Everything in here was computed against the local model snapshot
    identified by ``checkpoint_hash``. If the validator publishes a new
    checkpoint before this entry is submitted, the entry is discarded —
    the validator's checkpoint-hash gate (WRONG_CHECKPOINT) would reject
    a stale submission.

    Memory cost: ``hidden_states_list`` dominates — M_ROLLOUTS tensors of
    [seq_len, hidden_dim] in bfloat16 (e.g. Qwen3-4B + seq_len 2048 ≈
    14 MiB/rollout × 8 = ~110 MiB/entry). Pre-gen buffer sized accordingly.
    """

    prompt_idx: int
    generations: list[dict]               # tokens + prompt_length per rollout
    rewards: list[float]                  # already-validated against sigma_min
    completion_texts: list[str]           # for debug + future telemetry
    hidden_states_list: list[Any]         # torch.Tensor on proof GPU, one per rollout
    token_logprobs_list: list[list[float]]
    checkpoint_hash: str
    sigma: float
    enqueued_at: float                    # monotonic timestamp


class _PreGenBuffer:
    """FIFO of pre-generated entries, capped at ``max_size``.

    Drained when the validator OPENs; restocked during TRAINING / PUBLISHING /
    READY. Bounded by ``max_size`` AND by the per-hotkey submission cap
    (MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8) — making it larger than 8
    is pointless because we can't submit more than 8 per window anyway.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max(0, int(max_size))
        self._entries: list[_PreGenEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def is_full(self) -> bool:
        return len(self._entries) >= self.max_size

    def add(self, entry: _PreGenEntry) -> None:
        if self.max_size == 0:
            return
        # Dedup on prompt_idx — same prompt twice would race with itself
        # and only the first submission would land a slot.
        if any(e.prompt_idx == entry.prompt_idx for e in self._entries):
            return
        self._entries.append(entry)
        while len(self._entries) > self.max_size:
            self._entries.pop(0)

    def clear(self) -> None:
        self._entries.clear()

    def discard_stale(self, current_checkpoint_hash: str) -> int:
        """Drop entries whose checkpoint_hash no longer matches.

        Called after a checkpoint pull — old entries would be rejected as
        WRONG_CHECKPOINT by the validator. Returns count discarded.
        """
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if e.checkpoint_hash == current_checkpoint_hash
        ]
        return before - len(self._entries)

    def pop_eligible(
        self,
        cooldown_set: set[int],
        used_prompts_this_window: set[int],
    ) -> _PreGenEntry | None:
        """Pop the oldest entry whose prompt is still eligible.

        Eligibility = not in this window's cooldown set AND not already
        submitted by us in this window (per-prompt dedup). Drops ineligible
        entries as it scans so the buffer stays clean.
        """
        while self._entries:
            entry = self._entries[0]
            if entry.prompt_idx in cooldown_set:
                # Validator-side cooldown — would be rejected as
                # PROMPT_IN_COOLDOWN. Drop and try next.
                self._entries.pop(0)
                continue
            if entry.prompt_idx in used_prompts_this_window:
                self._entries.pop(0)
                continue
            return self._entries.pop(0)
        return None

    def prompt_indices(self) -> set[int]:
        return {e.prompt_idx for e in self._entries}


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
        min_sigma: float = SIGMA_MIN,
        prescreen_k: int = 4,
        prescreen_max_tokens: int = 1024,
        difficulty_blacklist_size: int = 4096,
        pregen_buffer_size: int = 4,
        state_poll_ms: int = 100,
        vllm_gpu_memory_utilization: float = 0.85,
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

        # Generation backend — vLLM is ~5-10× faster than HF .generate()
        # for our batched M=8 rollout case. We accept either a vllm.LLM
        # instance (preferred) or a HF AutoModelForCausalLM (fallback)
        # under the same constructor arg so existing call sites keep
        # working. vLLM is detected by class name + module prefix to
        # avoid a hard import dependency.
        self._use_vllm = _is_vllm_instance(vllm_model)
        self._vllm_gpu_memory_utilization = float(vllm_gpu_memory_utilization)
        if self._use_vllm:
            logger.info("MiningEngine: using vLLM for rollout generation")
        else:
            logger.info(
                "MiningEngine: using HuggingFace .generate() for rollouts "
                "(install vllm + pass --use-vllm for ~5-10× speedup)"
            )

        # Difficulty filter — keep submissions inside the validator's
        # zone (σ ≥ min_sigma). For 8 binary rewards, σ=0.433 corresponds
        # to exactly 2 (or 6) successes — the tightest in-zone configuration.
        # Pushing min_sigma above 0.43 narrows the accepted band; pushing
        # below 0.43 risks OUT_OF_ZONE rejections in steady state.
        self.min_sigma = float(min_sigma)
        # Probe rollouts: generate a small batch first, skip the prompt if
        # all-correct or all-incorrect (extreme difficulty). 0 disables
        # the probe and goes straight to full M=8 generation.
        if prescreen_k < 0 or prescreen_k >= M_ROLLOUTS:
            raise ValueError(
                f"prescreen_k must be in [0, {M_ROLLOUTS - 1}], got {prescreen_k}"
            )
        self.prescreen_k = int(prescreen_k)
        # 0 → match max_new_tokens (safe default: no length heterogeneity
        # between probe and tail rollouts). A smaller value trades some
        # false-negative skips for a faster probe on extreme prompts.
        self.prescreen_max_tokens = (
            int(prescreen_max_tokens) if prescreen_max_tokens > 0 else int(max_new_tokens)
        )
        # FIFO bounded set of prompt_idx that recently went OUT_OF_ZONE
        # or all-same in the probe. Avoids re-trying obviously-extreme
        # prompts every window. Reset on checkpoint pull (policy changed).
        self._difficulty_blacklist: list[int] = []
        self._difficulty_blacklist_set: set[int] = set()
        self._difficulty_blacklist_size = int(difficulty_blacklist_size)

        # Pre-generation buffer — entries are computed during the validator's
        # non-OPEN phase (TRAINING / PUBLISHING / READY) and finalized +
        # POSTed in a burst the instant OPEN flips. Capped at 8 because
        # MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8 — extra entries would
        # just be RATE_LIMITED. Larger buffer = more memory + more compute
        # wasted on entries that get superseded by checkpoint pulls.
        self._pregen = _PreGenBuffer(max_size=min(pregen_buffer_size, 8))
        # Poll interval during non-OPEN when we're waiting for OPEN with
        # ready entries. 100ms is a good balance between detection latency
        # (faster = earlier drand round = bigger slot share) and validator
        # /state load.
        self._state_poll_secs = max(0.01, state_poll_ms / 1000.0)

        # Lazy imports for heavy deps — keep module import cheap.
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    def _blacklist_prompt(self, prompt_idx: int) -> None:
        """Add a prompt to the bounded difficulty blacklist (FIFO eviction)."""
        if self._difficulty_blacklist_size <= 0:
            return
        if prompt_idx in self._difficulty_blacklist_set:
            return
        self._difficulty_blacklist.append(prompt_idx)
        self._difficulty_blacklist_set.add(prompt_idx)
        while len(self._difficulty_blacklist) > self._difficulty_blacklist_size:
            evicted = self._difficulty_blacklist.pop(0)
            self._difficulty_blacklist_set.discard(evicted)

    def _reset_difficulty_blacklist(self) -> None:
        """Wipe the blacklist — call after a checkpoint pull since the
        new policy may solve prompts the old one couldn't (and vice versa)."""
        self._difficulty_blacklist.clear()
        self._difficulty_blacklist_set.clear()

    # ------------------------------------------------------------------
    # Pre-generation pipeline (non-OPEN phase)
    # ------------------------------------------------------------------

    def _try_pregen_one(
        self,
        cooldown_set: set[int],
        local_hash: str,
        rng: _random.Random,
    ) -> bool:
        """Pre-generate one entry and stash it in the buffer.

        Returns True on a successful add (buffer grew), False on any skip
        (no eligible prompt, all-same probe, out-of-zone, generation
        failure). All work — generation, reward, σ-gate, HF forward pass
        for hidden_states/logprobs — happens here, so the OPEN burst only
        pays for randomness binding + signature + POST.
        """
        # Don't pre-gen for prompts already cached — those would race
        # against themselves at submit time. Treat the cache's prompt
        # indices as an extra blacklist for this attempt.
        cache_prompts = self._pregen.prompt_indices()
        avoid = self._difficulty_blacklist_set | cache_prompts
        try:
            prompt_idx = pick_prompt_idx(
                self.env, cooldown_set, rng=rng, blacklist=avoid,
            )
        except RuntimeError:
            return False

        problem = self.env.get_problem(prompt_idx)

        # Optional probe — same logic as the inline path.
        probe_gens: list[dict] = []
        probe_rewards: list[float] = []
        probe_texts: list[str] = []
        if self.prescreen_k > 0:
            probe_gens = self._generate_rollouts(
                problem, "",
                m=self.prescreen_k,
                max_new_tokens=self.prescreen_max_tokens,
            )
            for gen in probe_gens:
                text = self.tokenizer.decode(gen["tokens"][gen["prompt_length"]:])
                probe_texts.append(text)
                probe_rewards.append(self.env.compute_reward(problem, text))
            if len(set(probe_rewards)) <= 1:
                self._blacklist_prompt(prompt_idx)
                logger.debug(
                    "pregen skip prompt %d (probe all=%s)",
                    prompt_idx,
                    probe_rewards[0] if probe_rewards else None,
                )
                return False

        remaining = M_ROLLOUTS - len(probe_gens)
        if remaining > 0:
            tail_gens = self._generate_rollouts(
                problem, "",
                m=remaining,
                max_new_tokens=self.max_new_tokens,
            )
        else:
            tail_gens = []
        generations = probe_gens + tail_gens
        if len(generations) < M_ROLLOUTS:
            logger.warning(
                "pregen partial gen %d/%d for prompt %d",
                len(generations), M_ROLLOUTS, prompt_idx,
            )
            return False

        completion_texts = list(probe_texts)
        rewards = list(probe_rewards)
        for gen in tail_gens:
            text = self.tokenizer.decode(gen["tokens"][gen["prompt_length"]:])
            completion_texts.append(text)
            rewards.append(self.env.compute_reward(problem, text))

        sigma = _rewards_std(rewards)
        if sigma < self.min_sigma:
            self._blacklist_prompt(prompt_idx)
            logger.debug(
                "pregen skip prompt %d out_of_zone (σ=%.3f, %d/%d successes)",
                prompt_idx, sigma, int(sum(rewards)), M_ROLLOUTS,
            )
            return False

        # σ ≥ min_sigma → this entry is worth proving. Run the 8 HF
        # forward passes now (the slow part) so the OPEN burst is fast.
        try:
            hidden_states_list = []
            token_logprobs_list = []
            for gen in generations:
                hs, lp = self._grail_prepare(gen)
                hidden_states_list.append(hs)
                token_logprobs_list.append(lp)
        except Exception:
            logger.exception("pregen grail_prepare failed for prompt %d", prompt_idx)
            return False

        self._pregen.add(_PreGenEntry(
            prompt_idx=prompt_idx,
            generations=generations,
            rewards=rewards,
            completion_texts=completion_texts,
            hidden_states_list=hidden_states_list,
            token_logprobs_list=token_logprobs_list,
            checkpoint_hash=local_hash,
            sigma=sigma,
            enqueued_at=time.monotonic(),
        ))
        logger.info(
            "pregen READY prompt=%d σ=%.3f successes=%d/%d cache=%d/%d",
            prompt_idx, sigma, int(sum(rewards)), M_ROLLOUTS,
            len(self._pregen), self._pregen.max_size,
        )
        return True

    # ------------------------------------------------------------------
    # OPEN-burst submission path
    # ------------------------------------------------------------------

    async def _submit_pregen_entry(
        self,
        entry: _PreGenEntry,
        state,
        url: str,
        client,
        used_prompts_this_window: set[int],
    ):
        """Finalize a pre-gen entry against this window's randomness + POST.

        Returns the BatchSubmissionResponse if the POST resolved (accepted
        or rejected), or None on transport failure / pre-flight skip.
        """
        from reliquary.miner.submitter import (
            SubmissionError, get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import WindowState

        # CRITICAL: the validator verifies the envelope signature against
        # its CURRENT batcher's randomness (see server.py's
        # ``_randomness_for_sig``), not against whatever randomness we
        # signed with. If the window rolled between our last /state poll
        # and now (cheap GRAIL finalize + RTT), the verify will fail with
        # bad_envelope_signature even though our bytes are correct.
        # Re-poll right before signing so randomness/window are fresh.
        try:
            fresh = await get_window_state_v2(url, client=client)
        except Exception as e:
            logger.warning("pregen pre-submit /state refresh failed: %s", e)
            return None
        if fresh.state != WindowState.OPEN or not fresh.randomness:
            logger.info(
                "pregen skip submit prompt=%d: state rolled to %s",
                entry.prompt_idx, fresh.state.value if hasattr(fresh.state, "value") else fresh.state,
            )
            return None
        if entry.prompt_idx in set(fresh.cooldown_prompts):
            logger.info(
                "pregen skip submit prompt=%d: now in cooldown",
                entry.prompt_idx,
            )
            return None
        state = fresh
        randomness = state.randomness or ""
        rollout_submissions = [
            self._build_rollout_submission_from_prepared(
                gen, reward, hs, lp, randomness,
            )
            for gen, reward, hs, lp in zip(
                entry.generations,
                entry.rewards,
                entry.hidden_states_list,
                entry.token_logprobs_list,
            )
        ]
        merkle_root = _compute_merkle_root(rollout_submissions)
        current_round = _current_drand_round_at_send()
        import os as _os
        _nonce = _os.urandom(16).hex()
        _envelope_sig = sign_envelope(
            wallet=self.wallet,
            miner_hotkey=self.wallet.hotkey.ss58_address,
            window_start=state.window_n,
            prompt_idx=entry.prompt_idx,
            merkle_root=merkle_root,
            checkpoint_hash=entry.checkpoint_hash,
            drand_round=current_round,
            randomness=randomness,
            nonce=_nonce,
        ).hex()
        request = BatchSubmissionRequest(
            miner_hotkey=self.wallet.hotkey.ss58_address,
            prompt_idx=entry.prompt_idx,
            window_start=state.window_n,
            merkle_root=merkle_root,
            rollouts=rollout_submissions,
            checkpoint_hash=entry.checkpoint_hash,
            drand_round=current_round,
            nonce=_nonce,
            envelope_signature=_envelope_sig,
        )
        try:
            resp = await submit_batch_v2(url, request, client=client)
        except SubmissionError as exc:
            logger.error("pregen submit failed prompt=%d: %s", entry.prompt_idx, exc)
            return None
        # Track per-prompt dedup so we don't re-submit the same idx from
        # the inline path later in the same window.
        used_prompts_this_window.add(entry.prompt_idx)
        latency_ms = (time.monotonic() - entry.enqueued_at) * 1000
        logger.info(
            "pregen SUBMIT window=%d prompt=%d round=%d accepted=%s reason=%s "
            "(cached %.0fms)",
            state.window_n, entry.prompt_idx, current_round, resp.accepted,
            resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
            latency_ms,
        )
        return resp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,  # v2.0 param kept for CLI compat; ignored
        use_drand: bool = True,
    ) -> list:
        """Poll /state; pre-generate during non-OPEN; burst-submit on OPEN.

        State machine:
          * Read /state and pull a new checkpoint if needed. A pull
            invalidates pre-gen state (the cached entries' checkpoint_hash
            no longer matches what the validator expects).
          * OPEN + randomness ready → drain the pre-gen buffer at the
            fastest rate the validator will accept. Each pre-gen entry
            costs only commitments + signature + POST. Fall back to inline
            generation only after the buffer is empty.
          * Anything else (TRAINING / PUBLISHING / READY / OPEN-without-
            randomness) → fill the pre-gen buffer until it's at capacity
            or the env is fully blacklisted. Tight poll loop (state_poll_ms)
            so OPEN is detected within ~100 ms.

        Returns the list of BatchSubmissionResponse objects collected
        across the loop. The loop exits only on external cancellation.
        """
        import httpx
        import random

        from reliquary.constants import POLL_INTERVAL_SECONDS
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
            get_window_state_v2,
        )
        from reliquary.protocol.submission import WindowState

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
        results: list = []
        # Optional seed from main.py — when the caller already
        # downloaded + loaded the validator's current checkpoint at boot,
        # it sets these attrs so the first ``maybe_pull_checkpoint`` skips
        # a redundant download + load (which would otherwise OOM the GPU
        # because the old vLLM still holds its memory).
        local_n = int(getattr(self, "_initial_local_n", 0))
        local_hash = str(getattr(self, "_initial_local_hash", ""))
        # Per-window dedup: prompts we've already submitted in THIS
        # window. Resets every time window_n advances. Bounds us against
        # accidentally submitting the same prompt twice from buffer +
        # inline paths.
        current_window_n: int | None = None
        used_prompts_this_window: set[int] = set()

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

                # Reset per-window dedup at the window roll.
                if state.window_n != current_window_n:
                    current_window_n = state.window_n
                    used_prompts_this_window.clear()

                # Pull new checkpoint if needed (works at any state). A
                # successful pull invalidates the pregen buffer because
                # the cached entries' checkpoint_hash no longer matches
                # the validator's expectation.
                try:
                    prev_n = local_n
                    local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
                        state=state, local_n=local_n, local_hash=local_hash,
                        local_model=self.hf_model,
                        download_fn=_hf_download,
                        load_fn=self._load_checkpoint,
                    )
                    if local_n != prev_n:
                        self._reset_difficulty_blacklist()
                        dropped = self._pregen.discard_stale(local_hash)
                        if dropped:
                            logger.info(
                                "pregen invalidated %d entries on checkpoint pull "
                                "(now @ %d/%s)",
                                dropped, local_n, local_hash[:12],
                            )
                except Exception:
                    logger.exception("checkpoint pull failed; keeping local")

                cooldown_set = set(state.cooldown_prompts)
                window_open = (
                    state.state == WindowState.OPEN and bool(state.randomness)
                )

                # ------------------------------------------------------
                # OPEN burst: drain the pre-gen buffer first.
                # ------------------------------------------------------
                if window_open:
                    burst_count = 0
                    while True:
                        entry = self._pregen.pop_eligible(
                            cooldown_set, used_prompts_this_window,
                        )
                        if entry is None:
                            break
                        if entry.checkpoint_hash != local_hash:
                            # Sanity check — should already be filtered
                            # by discard_stale, but belt-and-braces.
                            continue
                        resp = await self._submit_pregen_entry(
                            entry, state, url, client,
                            used_prompts_this_window,
                        )
                        if resp is not None:
                            results.append(resp)
                            burst_count += 1
                    if burst_count:
                        logger.info(
                            "OPEN burst: submitted %d pregen entries for "
                            "window=%d", burst_count, state.window_n,
                        )

                    # Inline fallback: if the buffer was empty (or we
                    # haven't hit the per-hotkey cap yet), generate one
                    # submission from scratch. The validator's
                    # MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8 caps us,
                    # so anything we've already burst this window counts.
                    if (
                        len(used_prompts_this_window) < 8
                        and not self._pregen.is_full()
                    ):
                        resp = await self._inline_generate_and_submit(
                            state, url, client, rng,
                            cooldown_set, local_hash,
                            used_prompts_this_window,
                        )
                        if resp is not None:
                            results.append(resp)
                    # After serving OPEN, loop again immediately to catch
                    # any state transitions (TRAINING starts).
                    continue

                # ------------------------------------------------------
                # Non-OPEN: fill the pre-gen buffer.
                # ------------------------------------------------------
                if not self._pregen.is_full():
                    self._try_pregen_one(cooldown_set, local_hash, rng)
                    # Loop again immediately to fill the next slot.
                    continue

                # Buffer full and not OPEN → poll fast so we catch OPEN
                # within ~state_poll_secs.
                await asyncio.sleep(self._state_poll_secs)

        return results

    async def _inline_generate_and_submit(
        self,
        state,
        url: str,
        client,
        rng: _random.Random,
        cooldown_set: set[int],
        local_hash: str,
        used_prompts_this_window: set[int],
    ):
        """Fallback path when the pre-gen buffer is empty during OPEN.

        Same flow as the pre-gen pipeline + finalize, but inline so the
        miner still produces *something* on warm start or after a
        cache-invalidating checkpoint pull. Slow path — earlier-drand-
        round slots will have been won by anyone with a primed buffer.
        """
        from reliquary.miner.submitter import SubmissionError, submit_batch_v2

        randomness = state.randomness
        avoid = self._difficulty_blacklist_set | used_prompts_this_window
        try:
            prompt_idx = pick_prompt_idx(
                self.env, cooldown_set, rng=rng, blacklist=avoid,
            )
        except RuntimeError:
            logger.info("inline: env fully in cooldown / blacklist; skipping")
            return None

        problem = self.env.get_problem(prompt_idx)

        probe_gens: list[dict] = []
        probe_rewards: list[float] = []
        probe_texts: list[str] = []
        if self.prescreen_k > 0:
            probe_gens = self._generate_rollouts(
                problem, randomness,
                m=self.prescreen_k,
                max_new_tokens=self.prescreen_max_tokens,
            )
            for gen in probe_gens:
                text = self.tokenizer.decode(gen["tokens"][gen["prompt_length"]:])
                probe_texts.append(text)
                probe_rewards.append(self.env.compute_reward(problem, text))
            if len(set(probe_rewards)) <= 1:
                self._blacklist_prompt(prompt_idx)
                return None

        remaining = M_ROLLOUTS - len(probe_gens)
        tail_gens = (
            self._generate_rollouts(
                problem, randomness,
                m=remaining, max_new_tokens=self.max_new_tokens,
            )
            if remaining > 0 else []
        )
        generations = probe_gens + tail_gens
        if len(generations) < M_ROLLOUTS:
            return None

        completion_texts = list(probe_texts)
        rewards = list(probe_rewards)
        for gen in tail_gens:
            text = self.tokenizer.decode(gen["tokens"][gen["prompt_length"]:])
            completion_texts.append(text)
            rewards.append(self.env.compute_reward(problem, text))

        sigma = _rewards_std(rewards)
        if sigma < self.min_sigma:
            self._blacklist_prompt(prompt_idx)
            return None

        # Same timing-bug guard as ``_submit_pregen_entry``: the inline
        # path's generation took 30-60 s, and the validator verifies the
        # envelope sig against its CURRENT batcher randomness. If the
        # window rolled during gen, our commits are bound to the OLD
        # randomness (and the validator's WRONG_RANDOMNESS gate would
        # reject anyway). Re-poll and abort cleanly if anything moved.
        from reliquary.miner.submitter import get_window_state_v2
        from reliquary.protocol.submission import WindowState
        try:
            fresh = await get_window_state_v2(url, client=client)
        except Exception as e:
            logger.warning("inline pre-submit /state refresh failed: %s", e)
            return None
        if fresh.state != WindowState.OPEN or not fresh.randomness:
            logger.info("inline skip submit: state rolled out of OPEN")
            return None
        if fresh.randomness != randomness:
            logger.info(
                "inline skip submit prompt=%d: randomness rolled "
                "during gen (was %s..., now %s...)",
                prompt_idx, randomness[:12], fresh.randomness[:12],
            )
            return None
        if prompt_idx in set(fresh.cooldown_prompts):
            self._blacklist_prompt(prompt_idx)
            logger.info("inline skip submit prompt=%d: now in cooldown", prompt_idx)
            return None
        state = fresh

        rollout_submissions = [
            self._build_rollout_submission_with_reward(
                gen, problem, randomness, reward, text,
            )
            for gen, reward, text in zip(generations, rewards, completion_texts)
        ]
        merkle_root = _compute_merkle_root(rollout_submissions)
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
            randomness=randomness or "",
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
        except SubmissionError as exc:
            logger.error("inline submit failed prompt=%d: %s", prompt_idx, exc)
            return None
        used_prompts_this_window.add(prompt_idx)
        logger.info(
            "inline SUBMIT window=%d prompt=%d round=%d accepted=%s reason=%s",
            state.window_n, prompt_idx, current_round, resp.accepted,
            resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
        )
        return resp

    def _load_checkpoint(self, local_path: str):
        """Reload both generation and proof models from *local_path*.

        The generation model (``self.vllm_model``) may be either a HF
        AutoModelForCausalLM or a ``vllm.LLM``. The vLLM path destroys
        the engine and rebuilds — slow (~30-60 s) but rare (checkpoints
        publish every ``CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`` ≈ 10
        windows). Crucially, the GRAIL pre-gen buffer is invalidated by
        the caller on every successful pull, so we don't ship stale
        proofs after the swap.

        Returns the new ``hf_model`` (also stored on ``self.hf_model``).
        If the proof-model reload fails, the old model is kept and pre-
        gen continues against it (stable but stale).
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

        # 2. Reload the generation model on cuda:vllm_gpu. vLLM has no
        # hot-reload, so we destroy + rebuild. HF gets the usual
        # from_pretrained swap.
        if self._use_vllm:
            try:
                from vllm import LLM
                import gc
                # Tear down the old engine before allocating the new one
                # — vLLM holds ~gpu_memory_utilization fraction of GPU
                # memory and won't release it via ``del`` alone. v1
                # engine exposes ``llm_engine.engine_core.shutdown()``
                # (uniproc) or ``shutdown_engine_loop()`` (multiproc) on
                # the EngineCoreProc subprocess. We call whichever is
                # present and fall back to GC.
                old_gen = self.vllm_model
                self.vllm_model = None
                if old_gen is not None:
                    for shutdown_path in (
                        # v1 engine, uniproc executor
                        lambda g: g.llm_engine.engine_core.shutdown(),
                        # v1 engine, sync-MP client wrapper
                        lambda g: g.llm_engine.engine_core.shutdown_engine_loop(),
                        # legacy v0 engine
                        lambda g: g.llm_engine.shutdown(),
                    ):
                        try:
                            shutdown_path(old_gen)
                            logger.info("vLLM engine shutdown via %s", shutdown_path)
                            break
                        except (AttributeError, Exception) as e:
                            logger.debug("vLLM shutdown path failed: %s", e)
                            continue
                del old_gen
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass
                # Same max_model_len cap as main.py — without this, vLLM
                # uses the model's config max_position_embeddings (262144
                # for Qwen3-4B) and OOMs the KV cache.
                new_gen = LLM(
                    model=local_path,
                    dtype="bfloat16",
                    gpu_memory_utilization=getattr(
                        self, "_vllm_gpu_memory_utilization", 0.85
                    ),
                    enforce_eager=False,
                    trust_remote_code=True,
                    max_model_len=min(16384, self.max_new_tokens + 4096),
                    max_num_seqs=16,
                )
                self.vllm_model = new_gen
                self._use_vllm = True  # still vLLM after reload
            except Exception:
                logger.exception(
                    "Failed to rebuild vLLM engine from %s; falling back to "
                    "HuggingFace generation on the same checkpoint", local_path,
                )
                try:
                    self.vllm_model = AutoModelForCausalLM.from_pretrained(
                        local_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=ATTN_IMPLEMENTATION,
                    ).to(f"cuda:{self.vllm_gpu}").eval()
                    self._use_vllm = False
                except Exception:
                    logger.exception(
                        "HF generation fallback also failed; miner generation is "
                        "BROKEN until the next successful pull.",
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
                    "Failed to reload HF generation model from %s; miner "
                    "generation is BROKEN until the next successful pull. "
                    "hf_model was swapped so GRAIL proofs will be inconsistent.",
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

    def _generate_rollouts(
        self,
        problem,
        randomness,
        *,
        m: int,
        max_new_tokens: int | None = None,
    ) -> list[dict]:
        """Generate *m* completions at T_PROTO. Dispatches to vLLM or HF."""
        if m <= 0:
            return []
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        max_new_tokens = min(max_new_tokens, MAX_NEW_TOKENS_PROTOCOL_CAP)

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        if self._use_vllm:
            return self._generate_rollouts_vllm(
                prompt_tokens, m=m, max_new_tokens=max_new_tokens,
            )
        return self._generate_rollouts_hf(
            prompt_tokens, m=m, max_new_tokens=max_new_tokens,
        )

    def _generate_rollouts_hf(
        self,
        prompt_tokens: list[int],
        *,
        m: int,
        max_new_tokens: int,
    ) -> list[dict]:
        """HF-backed generation: one batched ``.generate()`` call of size m.

        A single batched call is ~5-7× faster than m serial calls — the
        matmul tiling uses far more GPU compute. Each row samples
        independently (do_sample=True), so GRPO-group semantics are
        preserved. Each output row is truncated at its first post-prompt
        EOS so trailing batch-padding (which HF pads with pad_token_id =
        eos_token_id) is not carried downstream — otherwise the validator's
        GRAIL forward pass would see extra EOS tokens the miner didn't
        "generate" in the usual sense.
        """
        import torch

        prompt_length = len(prompt_tokens)
        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * m,
                device=getattr(self.vllm_model, "device", "cpu"),
            )
            outputs = self.vllm_model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=TOP_K_PROTO,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        eos = self.tokenizer.eos_token_id
        rollouts = []
        for i in range(m):
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

    def _generate_rollouts_vllm(
        self,
        prompt_tokens: list[int],
        *,
        m: int,
        max_new_tokens: int,
    ) -> list[dict]:
        """vLLM-backed generation: ``n=m`` samples in a single request.

        vLLM's continuous batching + paged attention is dramatically faster
        than HF's static-batch generate(): a single prompt with n=8 samples
        finishes in seconds rather than tens of seconds for typical
        4-8B models on a single H100/H200.

        Token bookkeeping mirrors the HF path: we truncate each completion
        at its first EOS so trailing padding doesn't leak downstream into
        the validator's GRAIL forward pass.
        """
        from vllm import SamplingParams

        # vLLM uses ``top_k=-1`` to mean "disabled"; HF uses 0. Translate.
        vllm_top_k = TOP_K_PROTO if TOP_K_PROTO > 0 else -1
        sampling_params = SamplingParams(
            n=m,
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            top_k=vllm_top_k,
            max_tokens=max_new_tokens,
            # vLLM honours model's EOS by default; setting stop_token_ids
            # here would override it. Leave alone.
        )
        # Pre-tokenized input avoids vLLM re-running the chat template /
        # tokenizer and guarantees byte-equality with the validator's
        # canonical_prompt_tokens check (PROMPT_MISMATCH gate). vLLM 0.21
        # dropped the ``prompt_token_ids=`` kwarg; the first positional
        # arg now accepts ``list[int]`` or ``TokensPrompt`` directly.
        prompt_length = len(prompt_tokens)
        try:
            from vllm import TokensPrompt
            vllm_prompts = [TokensPrompt(prompt_token_ids=prompt_tokens)]
        except ImportError:
            vllm_prompts = [prompt_tokens]
        request_outputs = self.vllm_model.generate(
            vllm_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if not request_outputs:
            return []
        request_output = request_outputs[0]
        eos = self.tokenizer.eos_token_id
        rollouts: list[dict] = []
        for completion in request_output.outputs:
            gen = list(completion.token_ids)
            try:
                first_eos = gen.index(eos)
                gen = gen[: first_eos + 1]
            except ValueError:
                pass
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
            })
        # vLLM normally returns exactly n completions, but defend against
        # truncation under preemption.
        return rollouts[:m]

    def _generate_m_rollouts(self, problem, randomness) -> list[dict]:
        """Back-compat wrapper for the full M_ROLLOUTS at full max_new_tokens."""
        return self._generate_rollouts(
            problem, randomness, m=M_ROLLOUTS, max_new_tokens=self.max_new_tokens,
        )

    def _build_rollout_submission(self, generation, problem, randomness):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit.

        Kept for back-compat / tests; the live mining loop now uses
        ``_build_rollout_submission_with_reward`` to skip a second
        tokenizer.decode / env.compute_reward roundtrip.
        """
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        reward = self.env.compute_reward(problem, completion_text)
        return self._build_rollout_submission_with_reward(
            generation, problem, randomness, reward, completion_text,
        )

    def _build_rollout_submission_with_reward(
        self,
        generation,
        problem,
        randomness,
        reward: float,
        completion_text: str,  # noqa: ARG002 — accepted for symmetry/future use
    ):
        """Inline-path builder: HF forward + commit + sign in one shot.

        Reward + completion_text are pre-computed in the zone-gate path so
        we don't pay the tokenizer.decode + reward parsing twice per
        rollout. The pre-gen path uses ``_build_rollout_submission_from_prepared``
        instead, which feeds in cached hidden_states / logprobs and skips
        the forward pass.
        """
        commit = self._build_grail_commit(generation, randomness)
        return RolloutSubmission(
            tokens=generation["tokens"],
            reward=reward,
            commit=commit,
        )

    def _build_rollout_submission_from_prepared(
        self,
        generation: dict,
        reward: float,
        hidden_states: Any,
        token_logprobs: list[float],
        randomness: str,
    ) -> RolloutSubmission:
        """OPEN-burst path: bind cached pre-gen state to this window's randomness."""
        commit = self._grail_finalize(
            generation, hidden_states, token_logprobs, randomness,
        )
        return RolloutSubmission(
            tokens=generation["tokens"],
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

    def _grail_prepare(self, generation: dict) -> tuple[Any, list[float]]:
        """Run the HF forward pass and extract per-token logprobs.

        Returns ``(hidden_states, token_logprobs)``. Neither depends on
        per-window randomness, so this work is safe to do during the
        validator's non-OPEN phase as part of pre-generation.

        ``hidden_states`` is left on the proof GPU — at OPEN time we feed
        it directly into ``_grail_finalize`` which uses it once and
        releases the reference. The buffer's memory budget is bounded by
        the pre-gen buffer size; see ``_PreGenEntry`` for accounting.
        """
        import torch

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

        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]

        # fp32 log_softmax to match the validator and reduce tail-token drift.
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs: list[float] = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())
        # Free the logits tensor; only hidden_states is needed downstream.
        del logits, log_probs

        return hidden_states, token_logprobs

    def _grail_finalize(
        self,
        generation: dict,
        hidden_states: Any,
        token_logprobs: list[float],
        randomness: str,
    ) -> dict:
        """Combine pre-computed forward output with per-window randomness.

        Fast path: just a topk + bucketing + matmul (GRAIL commitments)
        and a sr25519 sign. No HF forward pass — the heavy work was done
        upstream in ``_grail_prepare``. Called inline during the OPEN
        burst, so it must stay sub-100 ms per rollout to keep the per-
        entry submit under one drand round.
        """
        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

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

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Construct a GRAIL proof commit dict from a generation dict.

        Convenience wrapper used by the legacy inline-submit path and by
        tests. Production now splits into ``_grail_prepare`` (no
        randomness, runs during non-OPEN pre-gen) +
        ``_grail_finalize`` (uses randomness, runs in the OPEN burst).
        """
        hidden_states, token_logprobs = self._grail_prepare(generation)
        return self._grail_finalize(
            generation, hidden_states, token_logprobs, randomness,
        )
