"""Pregeneration pool — prepare valid submissions ahead of the next window.

Why this is legitimate and why it works
---------------------------------------
Generation and the GRAIL forward pass are functions of ``(checkpoint, prompt,
tokens)`` *only*; the per-window drand randomness enters solely through
``r_vec`` (a 16-element coefficient vector) at the very end. So for the
**currently published checkpoint** a miner can, ahead of time:

  1. sample 8 rollouts per candidate prompt with vLLM,
  2. run the bit-identical HF forward to cache the randomness-free proof
     artifacts (per-token *buckets* + *token_logprobs*),
  3. compute the reward and the population σ, and keep only **in-zone** groups
     (σ ≥ SIGMA_MIN ⇔ 2..6 of 8 correct),
  4. pre-screen every rollout against the validator's behavioural gates
     (termination p_stop, token-authenticity floor, boxed-answer prob,
     sampling-distribution stats) with safety margins, dropping any risky group.

The rollouts are genuine current-checkpoint samples; nothing about them is
fabricated or replayed. When a window opens the engine only has to project the
cached buckets with the freshly revealed ``r_vec``, sign, and POST — microsecond
work that lets it fire in the first drand round and beat the BATCH_FILLED race.

All caches are keyed by the checkpoint HF revision. When the validator publishes
a new checkpoint the pool for the old revision is dropped (it would be
``WRONG_CHECKPOINT``) and refilled against the new weights.

Threading model: this class OWNS the vLLM + HF models and does *all* GPU work on
its own daemon thread. The async submit loop only reads the thread-safe
``PregenStore`` (CPU tensors) and the current-checkpoint scalars — it never
touches CUDA, so there is no cross-thread CUDA contention.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from reliquary.constants import (
    ATTN_IMPLEMENTATION,
    BOXED_ANSWER_MIN_PROB,
    CHALLENGE_K,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    MIN_EOS_PROBABILITY,
    M_ROLLOUTS,
    SAMPLING_LOW_Q10_MAX,
    SAMPLING_MEDIAN_LOW_MAX,
    SAMPLING_MIN_STEPS,
    TOKEN_AUTH_ARGMAX_CONF,
    TOKEN_AUTH_THRESHOLD,
)

logger = logging.getLogger(__name__)

# --- Safety margins above the validator's hard thresholds. We discard a
# rollout *before submitting* if it would land near any boundary, so the live
# submit path has effectively zero behavioural rejections. ---
SAFE_TOKEN_AUTH_MIN = max(TOKEN_AUTH_THRESHOLD * 100, 1e-8)  # vs 1e-10 floor
SAFE_P_STOP_MIN = max(MIN_EOS_PROBABILITY * 2, 0.02)          # vs 0.01 floor
SAFE_BOXED_MIN = max(BOXED_ANSWER_MIN_PROB * 5, 0.005)        # vs 0.001 floor
SAFE_SAMPLING_MEDIAN_MIN = SAMPLING_MEDIAN_LOW_MAX + 0.05     # vs 0.30 floor
SAFE_SAMPLING_Q10_MIN = SAMPLING_LOW_Q10_MAX * 2              # vs 0.025 floor


# ---------------------------------------------------------------------------
# Prepared artifacts
# ---------------------------------------------------------------------------

@dataclass
class PreparedRollout:
    """The randomness-free, cacheable half of one rollout's submission."""

    all_tokens: list[int]
    prompt_length: int
    completion_length: int
    reward: float
    token_logprobs: list[float]   # completion-only (length == completion_length)
    buckets: Any                  # CPU int8 tensor [seq_len, topk] from grail_cache.compute_buckets


@dataclass
class PreparedGroup:
    """A full GRPO group ready to submit on one prompt for one checkpoint."""

    prompt_idx: int
    checkpoint_hash: str
    sigma: float
    rollouts: list[PreparedRollout]


# ---------------------------------------------------------------------------
# Thread-safe store
# ---------------------------------------------------------------------------

class PregenStore:
    """Thread-safe pool of prepared groups, keyed by checkpoint revision.

    Producer: the pregeneration thread (``add``). Consumer: the async submit
    loop (``pop_groups``). Only the active checkpoint's groups are retained.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_ckpt: dict[str, dict[int, PreparedGroup]] = {}

    def set_active_checkpoint(self, checkpoint_hash: str) -> None:
        """Drop every group not belonging to *checkpoint_hash* (stale weights)."""
        with self._lock:
            self._by_ckpt = {checkpoint_hash: self._by_ckpt.get(checkpoint_hash, {})}

    def add(self, group: PreparedGroup) -> None:
        with self._lock:
            self._by_ckpt.setdefault(group.checkpoint_hash, {})[group.prompt_idx] = group

    def ready_count(self, checkpoint_hash: str) -> int:
        with self._lock:
            return len(self._by_ckpt.get(checkpoint_hash, {}))

    def prepared_idxs(self, checkpoint_hash: str) -> set[int]:
        with self._lock:
            return set(self._by_ckpt.get(checkpoint_hash, {}).keys())

    def pop_groups(
        self,
        checkpoint_hash: str,
        *,
        exclude_idxs: set[int],
        n: int,
    ) -> list[PreparedGroup]:
        """Remove and return up to *n* ready groups whose prompt_idx is not in
        *exclude_idxs* (cooldown set ∪ already-submitted-this-window)."""
        with self._lock:
            pool = self._by_ckpt.get(checkpoint_hash, {})
            out: list[PreparedGroup] = []
            for idx in list(pool.keys()):
                if idx in exclude_idxs:
                    continue
                out.append(pool.pop(idx))
                if len(out) >= n:
                    break
            return out


# ---------------------------------------------------------------------------
# Bounded "recently attempted" set so the sampler keeps exploring fresh prompts
# ---------------------------------------------------------------------------

class _BoundedSeen:
    def __init__(self, maxlen: int = 1_000_000) -> None:
        self._dq: deque[int] = deque(maxlen=maxlen)
        self._set: set[int] = set()

    def add_many(self, items) -> None:
        for it in items:
            if it not in self._set:
                if len(self._dq) == self._dq.maxlen:
                    self._set.discard(self._dq[0])
                self._dq.append(it)
                self._set.add(it)

    def __contains__(self, item) -> bool:
        return item in self._set


# ---------------------------------------------------------------------------
# Pregenerator
# ---------------------------------------------------------------------------

class Pregenerator:
    """Owns the models + a daemon thread that keeps the pool full."""

    def __init__(
        self,
        *,
        vllm_gen,
        hf_model,
        tokenizer,
        env,
        verifier,
        proof_device: str,
        checkpoint_n: int,
        checkpoint_hash: str,
        model_path: str,
        model_name: str,
        repo_id: str | None,
        target_ready: int = 64,
        gen_batch_size: int = 16,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        candidate_sampler: Callable[[int, set[int]], list[int]] | None = None,
        prompt_sources: set[str] | None = None,
        use_frontier: bool = False,
        winners_path: str | None = None,
        controls_path: str | None = None,
        frontier_save_path: str | None = None,
        seed_positive_idxs: list[int] | None = None,
        explicit_pool_idxs: list[int] | None = None,
        decool_snipe: bool = False,
    ) -> None:
        self.vllm_gen = vllm_gen
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.env = env
        self.verifier = verifier
        self.proof_device = proof_device
        self.store = PregenStore()

        self.target_ready = target_ready
        self.gen_batch_size = gen_batch_size
        self.max_new_tokens = max_new_tokens
        # Restrict the candidate pool to these OMI problem_source values. This
        # checkpoint is a long-CoT reasoner that only terminates (emits EOS) on
        # short, GSM8K-style problems; on hard MATH problems it rambles to the
        # cap and never yields a valid 8-rollout group. Filtering to GSM8K-style
        # sources is where in-zone groups actually come from (validated against
        # winners.jsonl). Prompt selection is explicitly allowed by the protocol.
        self.prompt_sources = prompt_sources
        self._candidate_pool = self._build_candidate_pool(prompt_sources)
        # Explicit curated pool overrides source filter + frontier. Used to mine
        # a vetted set of prompt indices that are confirmed in-zone (2-6/8) on the
        # CURRENT checkpoint (see scripts/winners_replay.py). Prompt selection is
        # explicitly allowed by the protocol; rewards/logprobs are recomputed by
        # the validator, so curating prompts is honest, not a tamper.
        self._explicit_pool = False
        if explicit_pool_idxs:
            self._candidate_pool = list(dict.fromkeys(int(i) for i in explicit_pool_idxs))
            self._explicit_pool = True
            logger.info(
                "EXPLICIT prompt-idx pool: %d prompts (source filter + frontier bypassed)",
                len(self._candidate_pool),
            )
        # Decool-sniping: a priority provider (set by the engine) supplies idxs
        # that just exited the validator's cooldown. The sampler mines those
        # first, then falls back to broad exploration over the candidate pool.
        self._priority_provider = None
        self._decool_snipe = decool_snipe
        if candidate_sampler is not None:
            self.candidate_sampler = candidate_sampler
        elif self._explicit_pool:
            self.candidate_sampler = self._default_sampler
        elif decool_snipe:
            self.candidate_sampler = self._decool_sampler
            logger.info("DECOOL-SNIPE sampler enabled (priority=cooldown exits, fallback=broad)")
        elif use_frontier and self._candidate_pool:
            from reliquary.miner.frontier import build_frontier_sampler
            self.candidate_sampler = build_frontier_sampler(
                env, self._candidate_pool,
                winners_path=winners_path, controls_path=controls_path,
                save_path=frontier_save_path, seed_positive_idxs=seed_positive_idxs,
            )
            logger.info(
                "frontier predictor ENABLED over %d-prompt pool (online learning)",
                len(self._candidate_pool),
            )
        else:
            self.candidate_sampler = self._default_sampler

        # Checkpoint state guarded by ``_meta_lock``.
        self._meta_lock = threading.Lock()
        self._cur_n = checkpoint_n
        self._cur_hash = checkpoint_hash
        self._cur_model_path = model_path
        self._cur_model_name = model_name
        self._repo_id = repo_id
        self._desired: tuple[str, str, int] | None = None  # (repo_id, revision, n)

        self._eos_set = self._resolve_eos_set()
        self._seen = _BoundedSeen()
        self._cooldown_provider: Callable[[], set[int]] = lambda: set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.store.set_active_checkpoint(self._cur_hash)

    # ---- public API used by the engine (async thread) ----

    def current(self) -> tuple[int, str, str]:
        """Return ``(checkpoint_n, checkpoint_hash, model_name)`` atomically."""
        with self._meta_lock:
            return self._cur_n, self._cur_hash, self._cur_model_name

    def request_checkpoint(self, repo_id: str | None, revision: str, n: int) -> None:
        """Ask the pregen thread to switch to a newly published checkpoint."""
        with self._meta_lock:
            self._desired = (repo_id, revision, n)

    def set_cooldown_provider(self, fn: Callable[[], set[int]]) -> None:
        self._cooldown_provider = fn

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="pregen", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---- daemon loop ----

    def _run(self) -> None:
        logger.info("pregen thread started (target_ready=%d, batch=%d)", self.target_ready, self.gen_batch_size)
        while not self._stop.is_set():
            try:
                if self._maybe_reload():
                    continue
                ckpt_hash = self._cur_hash
                self.store.set_active_checkpoint(ckpt_hash)
                if self.store.ready_count(ckpt_hash) >= self.target_ready:
                    self._stop.wait(0.5)
                    continue
                cooldown = self._safe_cooldown()
                exclude = cooldown | self.store.prepared_idxs(ckpt_hash)
                candidates = self.candidate_sampler(self.gen_batch_size, exclude)
                candidates = [c for c in candidates if c not in self._seen][: self.gen_batch_size]
                if not candidates:
                    # Small curated / decool pools drain _seen after one pass;
                    # reset it so prompts whose groups were consumed/submitted (or
                    # that decool again later) get re-mined for the next window
                    # (cooldown + prepared_idxs still exclude ineligible ones).
                    if self._explicit_pool or self._decool_snipe:
                        self._seen = _BoundedSeen()
                    self._stop.wait(0.5)
                    continue
                self._seen.add_many(candidates)
                groups = self.build_groups(candidates, ckpt_hash)
                # Only store if still on the same checkpoint we built against.
                if self._cur_hash == ckpt_hash:
                    for g in groups:
                        self.store.add(g)
                logger.info(
                    "pregen: +%d/%d in-zone groups (pool=%d, ckpt=%s)",
                    len(groups), len(candidates), self.store.ready_count(ckpt_hash), ckpt_hash[:10],
                )
            except Exception:
                logger.exception("pregen loop iteration failed; backing off")
                self._stop.wait(2.0)

    def _safe_cooldown(self) -> set[int]:
        try:
            return set(self._cooldown_provider())
        except Exception:
            return set()

    # ---- checkpoint reload (runs on the pregen thread → no cross-thread CUDA) ----

    def _maybe_reload(self) -> bool:
        with self._meta_lock:
            desired = self._desired
            if desired is None or desired[1] == self._cur_hash:
                self._desired = None
                return False
            repo_id, revision, n = desired

        logger.info("pregen: reloading checkpoint -> n=%d rev=%s", n, revision[:10])
        try:
            local_path = self._download(repo_id, revision)
            new_hf = self._load_hf(local_path)
        except Exception:
            logger.exception("pregen: checkpoint download/load failed; keeping current")
            with self._meta_lock:
                self._desired = None
            return True

        # Swap HF proof model, then reload the vLLM weights.
        old_hf = self.hf_model
        self.hf_model = new_hf
        del old_hf
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            self.vllm_gen.reload(local_path)
        except Exception:
            logger.exception("pregen: vLLM reload failed; generation halted until next pull")

        self._eos_set = self._resolve_eos_set()
        with self._meta_lock:
            self._cur_n = n
            self._cur_hash = revision
            self._cur_model_path = local_path
            self._cur_model_name = getattr(self.hf_model, "name_or_path", local_path)
            self._repo_id = repo_id
            self._desired = None
        self.store.set_active_checkpoint(revision)
        self._seen = _BoundedSeen()
        return True

    def _download(self, repo_id: str | None, revision: str) -> str:
        from huggingface_hub import snapshot_download

        if repo_id is None:
            # No remote repo (bootstrap / local path mode): reuse current path.
            return self._cur_model_path
        return snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=["model.safetensors", "config.json", "tokenizer*", "*.json"],
        )

    def _load_hf(self, local_path: str):
        import torch
        from transformers import AutoModelForCausalLM

        return (
            AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            )
            .to(self.proof_device)
            .eval()
        )

    # ---- group construction ----

    def _record_outcome(self, prompt_idx: int, n_correct: int | None, terminated: int) -> None:
        """Feed a group's outcome to the frontier sampler for online learning."""
        rec = getattr(self.candidate_sampler, "record", None)
        if rec is None:
            return
        try:
            rec(prompt_idx, n_correct, terminated)
        except Exception:
            logger.exception("frontier record failed")

    def build_groups(self, prompt_idxs: list[int], checkpoint_hash: str) -> list[PreparedGroup]:
        from reliquary.validator.verifier import is_in_zone, rewards_std

        prompts: list[tuple[int, list[int]]] = []
        problems: dict[int, dict] = {}
        for idx in prompt_idxs:
            problem = self.env.get_problem(idx)
            problems[idx] = problem
            ptoks = self.tokenizer.encode(problem["prompt"], add_special_tokens=False)
            prompts.append((idx, ptoks))

        gen_groups = self.vllm_gen.generate_groups(prompts, max_new_tokens=self.max_new_tokens)

        prepared: list[PreparedGroup] = []
        n_lt8 = n_oz = n_safety = 0
        comp_counts: list[int] = []
        correct_hist: list[int] = []   # #correct (0..8) among groups with 8 completions
        for gg in gen_groups:
            comp_counts.append(len(gg.completions))
            if len(gg.completions) < M_ROLLOUTS:
                n_lt8 += 1
                self._record_outcome(gg.prompt_idx, None, len(gg.completions))
                continue  # need 8 genuine EOS-terminated samples — no cherry-picking
            comps = gg.completions[:M_ROLLOUTS]
            problem = problems[gg.prompt_idx]
            plen = len(gg.prompt_token_ids)

            rewards: list[float] = []
            rollout_tokens: list[list[int]] = []
            for comp in comps:
                all_tokens = gg.prompt_token_ids + comp
                completion_text = self.tokenizer.decode(comp)
                rewards.append(float(self.env.compute_reward(problem, completion_text)))
                rollout_tokens.append(all_tokens)
            correct_hist.append(int(sum(rewards)))
            self._record_outcome(gg.prompt_idx, int(sum(rewards)), len(gg.completions))

            sigma = rewards_std(rewards)
            if not is_in_zone(sigma):
                n_oz += 1
                continue  # OUT_OF_ZONE pre-filter — the whole point of the pool

            prepared_rollouts: list[PreparedRollout] = []
            ok = True
            for all_tokens, reward in zip(rollout_tokens, rewards):
                pr = self._build_prepared_rollout(all_tokens, plen, reward)
                if pr is None:
                    ok = False
                    break
                prepared_rollouts.append(pr)
            if not ok or len(prepared_rollouts) != M_ROLLOUTS:
                n_safety += 1
                continue

            prepared.append(
                PreparedGroup(
                    prompt_idx=gg.prompt_idx,
                    checkpoint_hash=checkpoint_hash,
                    sigma=sigma,
                    rollouts=prepared_rollouts,
                )
            )

        if gen_groups:
            import statistics as _st
            avg_comp = round(_st.mean(comp_counts), 1) if comp_counts else 0.0
            hist = {k: correct_hist.count(k) for k in range(M_ROLLOUTS + 1) if correct_hist.count(k)}
            logger.info(
                "pregen batch detail: kept=%d drop[<8comp=%d out_of_zone=%d safety=%d] "
                "avg_completions_returned=%.1f correct_hist(8-comp groups)=%s",
                len(prepared), n_lt8, n_oz, n_safety, avg_comp, hist,
            )
        return prepared

    def _build_prepared_rollout(
        self, all_tokens: list[int], prompt_length: int, reward: float
    ) -> PreparedRollout | None:
        """Run the HF forward, cache buckets+logprobs, and pre-screen against
        every behavioural gate with margin. Returns None to reject the rollout
        (which drops the whole group)."""
        import torch

        from reliquary.miner.grail_cache import compute_buckets
        from reliquary.shared.forward import forward_single_layer

        seq_len = len(all_tokens)
        completion_length = seq_len - prompt_length
        # verify_logprobs_claim needs completion_length >= CHALLENGE_K, else it
        # returns (False, inf) → LOGPROB_MISMATCH. Reject short completions now.
        if completion_length < CHALLENGE_K:
            return None
        if seq_len < 2:
            return None

        proof_input = torch.tensor([all_tokens], device=self.proof_device)
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )
        hidden_states = hidden_states[0]               # [seq, hidden]
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)   # matches reference

        buckets = compute_buckets(hidden_states).to("cpu")

        rows = torch.arange(prompt_length - 1, seq_len - 1, device=log_probs.device)
        toks = torch.tensor(all_tokens[prompt_length:seq_len], device=log_probs.device)
        comp_logprobs = log_probs[rows, toks]                      # [comp]
        chosen_probs = comp_logprobs.exp()                         # [comp]
        # max over vocab once ([seq]) then index, avoiding a [comp, vocab] copy.
        argmax_probs = log_probs.max(dim=-1).values[rows].exp()    # [comp]

        token_logprobs = [float(x) for x in comp_logprobs.tolist()]
        chosen = [float(x) for x in chosen_probs.tolist()]
        amax = [float(x) for x in argmax_probs.tolist()]

        # p_stop: EOS mass at the step that predicted the final token.
        eos_idx = torch.tensor(sorted(self._eos_set), device=log_probs.device)
        p_stop = float(log_probs[seq_len - 2].exp()[eos_idx].sum().item())

        # free the big fp32 logits tensor early
        del log_probs, logits, hidden_states

        if not self._passes_safety(all_tokens, prompt_length, completion_length, chosen, amax, p_stop):
            return None

        return PreparedRollout(
            all_tokens=all_tokens,
            prompt_length=prompt_length,
            completion_length=completion_length,
            reward=reward,
            token_logprobs=token_logprobs,
            buckets=buckets,
        )

    def _passes_safety(
        self,
        all_tokens: list[int],
        prompt_length: int,
        completion_length: int,
        chosen: list[float],
        amax: list[float],
        p_stop: float,
    ) -> bool:
        # Termination (verify_termination Path 2).
        if all_tokens[-1] not in self._eos_set or p_stop < SAFE_P_STOP_MIN:
            return False
        # Token authenticity (TOKEN_TAMPERED): no emitted token below the floor.
        if min(chosen) < SAFE_TOKEN_AUTH_MIN:
            return False
        # Sampling distribution (soft, but avoid): median/q10 must clear margins.
        if completion_length >= SAMPLING_MIN_STEPS:
            import numpy as np

            x = np.asarray(chosen, dtype=np.float64)
            if float(np.median(x)) < SAFE_SAMPLING_MEDIAN_MIN:
                return False
            if float(np.quantile(x, 0.10)) < SAFE_SAMPLING_Q10_MIN:
                return False
        # Boxed-answer probability (BOXED_ANSWER_TAMPERED): every boxed token
        # must clear the floor unless the model wasn't confident there.
        if not self._boxed_ok(all_tokens, prompt_length, completion_length, chosen, amax):
            return False
        return True

    def _boxed_ok(
        self,
        all_tokens: list[int],
        prompt_length: int,
        completion_length: int,
        chosen: list[float],
        amax: list[float],
    ) -> bool:
        try:
            from reliquary.validator.verifier import _find_last_boxed_token_range
        except Exception:
            return True  # defence-in-depth only; zone+auth already strong
        completion_tokens = all_tokens[prompt_length: prompt_length + completion_length]
        rng = _find_last_boxed_token_range(completion_tokens, self.tokenizer)
        if rng is None:
            return True
        start, end = rng
        for i in range(start, end + 1):
            if 0 <= i < len(chosen):
                if chosen[i] < SAFE_BOXED_MIN and i < len(amax) and amax[i] >= TOKEN_AUTH_ARGMAX_CONF:
                    return False
        return True

    # ---- helpers ----

    def _resolve_eos_set(self) -> set[int]:
        try:
            from reliquary.validator.verifier import _eos_set_from_model

            s = _eos_set_from_model(self.hf_model, self.tokenizer)
            if s:
                return s
        except Exception:
            pass
        eos = getattr(self.tokenizer, "eos_token_id", None)
        return {int(eos)} if eos is not None else set()

    def _build_candidate_pool(self, sources: set[str] | None) -> list[int] | None:
        """Return the list of prompt indices whose OMI problem_source is in
        *sources*, or None to sample uniformly over the whole env."""
        if not sources:
            return None
        try:
            col = self.env._dataset["problem_source"]
            pool = [i for i, s in enumerate(col) if s in sources]
            logger.info(
                "candidate pool: %d/%d prompts with problem_source in %s",
                len(pool), len(col), sorted(sources),
            )
            if pool:
                return pool
            logger.warning("no prompts matched %s; falling back to uniform sampling", sorted(sources))
        except Exception:
            logger.exception("could not build source-filtered candidate pool; using uniform")
        return None

    def set_priority_provider(self, fn) -> None:
        """Register a callable returning a list of high-priority prompt idxs
        (e.g. freshly cooldown-exited prompts for decool-sniping)."""
        self._priority_provider = fn

    def _decool_sampler(self, n: int, exclude: set[int]) -> list[int]:
        """Mine freshly-decooled prompts first (recently rewarded = likely
        in-zone and now submittable), then fall back to broad exploration."""
        out: list[int] = []
        if self._priority_provider is not None:
            try:
                for idx in self._priority_provider():
                    idx = int(idx)
                    if idx not in exclude and idx not in out:
                        out.append(idx)
                        if len(out) >= n:
                            return out
            except Exception:
                logger.exception("priority provider failed; using fallback only")
        if len(out) < n:
            out += self._default_sampler(n - len(out), exclude | set(out))
        return out

    def _default_sampler(self, n: int, exclude: set[int]) -> list[int]:
        import random as _random

        pool = self._candidate_pool
        out: list[int] = []
        attempts = 0
        if pool is not None:
            size = len(pool)
            while len(out) < n and attempts < n * 100:
                idx = pool[_random.randrange(size)]
                attempts += 1
                if idx not in exclude:
                    out.append(idx)
            return out
        size = len(self.env)
        while len(out) < n and attempts < n * 50:
            idx = _random.randrange(size)
            attempts += 1
            if idx not in exclude:
                out.append(idx)
        return out
