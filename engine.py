"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from typing import TYPE_CHECKING

import random as _random

from reliquary.constants import (
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
    UPLOAD_BUFFER,
    WINDOW_LENGTH,
)
from reliquary.infrastructure import chain
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


def _has_complete_boxed(text: str) -> bool:
    """True iff ``text`` contains a balanced ``\\boxed{...}`` or ``\\fbox{...}``.

    Mirrors the brace-walk in ``reliquary.environment.math._last_boxed_only_string``
    so that we stop generation at exactly the point the env's reward extractor
    would have a scoreable answer — nothing more.
    """
    for marker in ("\\boxed{", "\\fbox{"):
        idx = text.rfind(marker)
        if idx < 0:
            continue
        depth = 1
        j = idx + len(marker)
        while j < len(text):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return True
            j += 1
    return False


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

        # Per-prompt σ memory for frontier-band selection. Cleared on every
        # checkpoint advance because the policy has shifted and old σ
        # observations no longer predict the new in-zone band.
        self._prompt_sigma_history: dict[int, deque] = {}
        # Prompts observed with σ < threshold under the current checkpoint —
        # excluded from both exploit and fallback paths until checkpoint
        # advances. With 8 Bernoulli samples at T=0.9, a single σ=0 outcome
        # (all-correct or all-wrong) is strong evidence the prompt sits
        # outside the learnable band for this policy — retrying just wastes
        # another full rollout group on a guaranteed OUT_OF_ZONE.
        self._blacklisted_prompts: set[int] = set()
        # Per-difficulty-level σ history. MATH problems have levels 1–5;
        # Qwen3-4B-Instruct typically dominates level 1–2 (σ=0 all-correct)
        # and struggles with 5 (σ=0 all-wrong) — the in-zone band tends to
        # live in 3–4. We accumulate sigma observations per level and use
        # the level distribution to bias fallback picks toward the
        # currently most-learnable level.
        self._level_sigma_history: dict[int, deque] = {}
        # Pre-built level → list[prompt_idx] map, built lazily on first use.
        self._prompts_by_level: dict[int, list[int]] | None = None
        self._stats_checkpoint_n: int = -1
        self._STATS_MAXLEN = 5
        # Single-observation is enough for inclusion: with ~12,500 prompts
        # and ~50 picks/hour, repeat picks on the same prompt are rare, so
        # waiting for 2 observations would keep the picker in fallback
        # indefinitely. The blacklist (σ < threshold) already filters bad
        # prompts; remaining observations are reliable enough.
        self._STATS_MIN_OBS = 1
        self._SIGMA_THRESHOLD = 0.43
        self._EXPLORE_RATE = 0.20
        # When True, the model is allowed to ramble to the protocol cap.
        # This is the strategy observed in top-miner submission logs:
        # long completions with multiple boxed answers manufacture variance
        # on otherwise-easy prompts (env extracts the *last* boxed, which
        # varies across sampling paths), keeping σ in-zone where
        # early-stopping would land σ=0 / all-correct. Trade-off: ~8× slower
        # generation per attempt. Worthwhile when windows stay open long
        # (light competition); harmful when windows fill in seconds. Set
        # via RELIQUARY_LET_MODEL_RAMBLE=1 environment variable.
        import os as _os
        self._LET_MODEL_RAMBLE = _os.environ.get(
            "RELIQUARY_LET_MODEL_RAMBLE", "0"
        ).strip() == "1"
        # Exploit fires after 3 in-zone observations rather than 10 — the
        # candidate pool grows slowly given the prompt count, and we'd
        # rather concentrate on a thin pool of known-good than dilute back
        # to uniform-random fallback.
        self._MIN_FRONTIER_POOL = 3

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

        from reliquary.constants import M_ROLLOUTS, POLL_INTERVAL_SECONDS
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

        # Compute randomness (once — v2.1 uses it only for GRAIL sketch seed)
        randomness = await self._compute_randomness(subtensor, 0, use_drand)

        rng = random.Random()
        results = []
        local_n = 0
        local_hash = ""

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

                # Pick prompt, generate, submit.
                cooldown_set = set(state.cooldown_prompts)
                try:
                    prompt_idx, pick_mode, pick_sigma, pick_pool = (
                        self._pick_prompt_idx_smart(cooldown_set, local_n, rng)
                    )
                except RuntimeError:
                    logger.info("env fully in cooldown; sleeping")
                    await asyncio.sleep(5)
                    continue

                logger.info(
                    "pick window=%d prompt=%d mode=%s mean_sigma=%s pool=%d ckpt_n=%d",
                    state.window_n, prompt_idx, pick_mode,
                    f"{pick_sigma:.3f}" if pick_sigma is not None else "n/a",
                    pick_pool, local_n,
                )

                problem = self.env.get_problem(prompt_idx)

                t_gen_start = time.perf_counter()
                generations = self._generate_m_rollouts(problem, randomness)
                t_gen_ms = int((time.perf_counter() - t_gen_start) * 1000)

                if len(generations) < M_ROLLOUTS:
                    logger.warning(
                        "generated %d/%d for prompt %d; skipping",
                        len(generations), M_ROLLOUTS, prompt_idx,
                    )
                    continue

                t_proof_start = time.perf_counter()
                rollout_submissions = [
                    self._build_rollout_submission(gen, problem, randomness)
                    for gen in generations
                ]
                merkle_root = _compute_merkle_root(rollout_submissions)
                t_proof_ms = int((time.perf_counter() - t_proof_start) * 1000)

                rewards_list = [r.reward for r in rollout_submissions]
                sigma_obs = statistics.pstdev(rewards_list)
                self._record_sigma(prompt_idx, sigma_obs, local_n)

                # Pre-submit freshness check: generation took 10s–200s+; in
                # that time the window may have sealed (→ window_not_active /
                # window_mismatch reject) or the validator may have bounced
                # (→ HTTP submit hangs until our 10s wait_for fires). A cheap
                # /state poll with a 3s ceiling catches both cases and lets
                # us skip a doomed submit before it ties up the loop.
                try:
                    fresh_state = await asyncio.wait_for(
                        get_window_state_v2(url, client=client),
                        timeout=3.0,
                    )
                    stale = (
                        fresh_state.state != WindowState.OPEN
                        or fresh_state.window_n != state.window_n
                    )
                    if stale:
                        logger.warning(
                            "skip stale submit window=%d prompt=%d rewards=%s "
                            "sigma_obs=%.3f (validator window=%d state=%s)",
                            state.window_n, prompt_idx, rewards_list, sigma_obs,
                            fresh_state.window_n, fresh_state.state.value
                            if hasattr(fresh_state.state, "value")
                            else fresh_state.state,
                        )
                        continue
                except (asyncio.TimeoutError, SubmissionError, Exception) as exc:
                    logger.warning(
                        "skip submit window=%d prompt=%d rewards=%s sigma_obs=%.3f "
                        "(/state check failed: %s)",
                        state.window_n, prompt_idx, rewards_list, sigma_obs,
                        type(exc).__name__,
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
                t_submit_start = time.perf_counter()
                try:
                    # Hard outer ceiling: submitter.py retries 3× with 60s
                    # per-attempt timeout, so a hung connection (e.g. window
                    # already sealed under load) can burn ~180s. We cap the
                    # whole retry loop at 10s so the engine moves on to the
                    # next window instead of starving on a doomed submission.
                    resp = await asyncio.wait_for(
                        submit_batch_v2(url, request, client=client),
                        timeout=10.0,
                    )
                    t_submit_ms = int((time.perf_counter() - t_submit_start) * 1000)
                    reason_str = (
                        resp.reason.value if hasattr(resp.reason, "value") else resp.reason
                    )
                    logger.info(
                        "submit window=%d prompt=%d accepted=%s reason=%s "
                        "rewards=%s sigma_obs=%.3f gen_ms=%d proof_ms=%d submit_ms=%d total_ms=%d "
                        "blacklist=%d",
                        state.window_n, prompt_idx, resp.accepted, reason_str,
                        rewards_list, sigma_obs, t_gen_ms, t_proof_ms, t_submit_ms,
                        t_gen_ms + t_proof_ms + t_submit_ms,
                        len(self._blacklisted_prompts),
                    )
                    results.append(resp)
                except asyncio.TimeoutError:
                    t_submit_ms = int((time.perf_counter() - t_submit_start) * 1000)
                    logger.error(
                        "submit window=%d prompt=%d timeout submit_ms=%d rewards=%s sigma_obs=%.3f (abandoned)",
                        state.window_n, prompt_idx, t_submit_ms, rewards_list, sigma_obs,
                    )
                except SubmissionError as exc:
                    t_submit_ms = int((time.perf_counter() - t_submit_start) * 1000)
                    logger.error(
                        "submit window=%d prompt=%d error submit_ms=%d rewards=%s sigma_obs=%.3f: %s",
                        state.window_n, prompt_idx, t_submit_ms, rewards_list, sigma_obs, exc,
                    )

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
        from transformers import StoppingCriteria, StoppingCriteriaList

        prompt_tokens = self.tokenizer.encode(
            problem["prompt"], add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        # Stop as soon as every sample in the batch has emitted (a) a balanced
        # \boxed{...} (or \fbox{...}) AND (b) any of the model's natural EOS
        # tokens. The EOS set must match what validator's verify_termination
        # accepts (validator/verifier.py:94-119): generation_config.eos_token_id
        # if available (Qwen3-Instruct = [151645, 151643]), else tokenizer.eos_token_id.
        # Checking only tokenizer.eos_token_id misses 151643 (endoftext) emissions
        # — those samples then never stop early and run to the 8192-token cap,
        # producing the 200+ second slow-tail batches.
        tokenizer = self.tokenizer
        eos_ids: set[int] = set()
        gen_cfg = getattr(self.vllm_model, "generation_config", None)
        if gen_cfg is not None:
            cfg_eos = getattr(gen_cfg, "eos_token_id", None)
            if isinstance(cfg_eos, int):
                eos_ids = {cfg_eos}
            elif isinstance(cfg_eos, (list, tuple)):
                eos_ids = {int(e) for e in cfg_eos if e is not None}
        if not eos_ids and tokenizer.eos_token_id is not None:
            eos_ids = {int(tokenizer.eos_token_id)}

        class _BoxedComplete(StoppingCriteria):
            def __init__(self):
                self._step = 0
                self._done = [False] * M_ROLLOUTS

            def __call__(self, input_ids, scores, **kwargs):
                self._step += 1
                if self._step % 16 != 0:
                    return False
                for i in range(input_ids.shape[0]):
                    if self._done[i]:
                        continue
                    new_tokens = input_ids[i, prompt_length:].tolist()
                    if not any(t in eos_ids for t in new_tokens):
                        continue
                    if _has_complete_boxed(tokenizer.decode(new_tokens)):
                        self._done[i] = True
                return all(self._done)

        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * M_ROLLOUTS,
                device=getattr(self.vllm_model, "device", "cpu"),
            )
            generate_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=T_PROTO,
                top_p=TOP_P_PROTO,
                top_k=TOP_K_PROTO,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if not self._LET_MODEL_RAMBLE:
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [_BoxedComplete()]
                )
            outputs = self.vllm_model.generate(input_tensor, **generate_kwargs)
        rollouts = []
        for i in range(M_ROLLOUTS):
            seq = outputs[i].tolist()
            gen = seq[prompt_length:]
            # Truncate at the first occurrence of ANY EOS token in the
            # validator-recognised set, not just tokenizer.eos_token_id —
            # otherwise samples that ended with 151643 keep trailing
            # padding tokens in the submission.
            first_eos = next((j for j, t in enumerate(gen) if t in eos_ids), -1)
            if first_eos >= 0:
                gen = gen[: first_eos + 1]
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

        # Token log-probs from HF (bit-identical with validator)
        log_probs = torch.log_softmax(logits[0], dim=-1)
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

    def _reset_stats_if_stale(self, local_n: int) -> None:
        if local_n != self._stats_checkpoint_n:
            self._prompt_sigma_history.clear()
            self._blacklisted_prompts.clear()
            self._level_sigma_history.clear()
            self._stats_checkpoint_n = local_n

    def _get_level(self, prompt_idx: int) -> int | None:
        """Read the MATH difficulty level (1–5) for a prompt, or None if
        the underlying dataset doesn't expose it."""
        dataset = getattr(self.env, "_dataset", None)
        if dataset is None:
            return None
        try:
            row = dataset[prompt_idx % len(dataset)]
            level_str = str(row.get("level", "")).strip()
            if not level_str:
                return None
            return int(level_str.split()[-1])
        except (KeyError, ValueError, IndexError, TypeError):
            return None

    def _build_prompts_by_level(self) -> dict[int, list[int]]:
        if self._prompts_by_level is None:
            mapping: dict[int, list[int]] = {}
            for i in range(len(self.env)):
                level = self._get_level(i)
                if level is not None:
                    mapping.setdefault(level, []).append(i)
            self._prompts_by_level = mapping
        return self._prompts_by_level

    def _record_sigma(self, prompt_idx: int, sigma_obs: float, local_n: int) -> None:
        self._reset_stats_if_stale(local_n)
        hist = self._prompt_sigma_history.get(prompt_idx)
        if hist is None:
            hist = deque(maxlen=self._STATS_MAXLEN)
            self._prompt_sigma_history[prompt_idx] = hist
        hist.append(sigma_obs)
        if sigma_obs < self._SIGMA_THRESHOLD:
            self._blacklisted_prompts.add(prompt_idx)
        level = self._get_level(prompt_idx)
        if level is not None:
            level_hist = self._level_sigma_history.get(level)
            if level_hist is None:
                level_hist = deque(maxlen=50)
                self._level_sigma_history[level] = level_hist
            level_hist.append(sigma_obs)

    def _pick_prompt_idx_smart(
        self,
        cooldown_set: set[int],
        local_n: int,
        rng: _random.Random,
    ) -> tuple[int, str, float | None, int]:
        """Frontier-band picker with ε-exploration.

        Returns ``(prompt_idx, mode, mean_sigma_or_None, pool_size)``. Mode
        is one of ``"explore"`` (ε-coin), ``"exploit"`` (frontier sample),
        ``"level_biased"`` (no exploit pool yet but level priors available),
        or ``"fallback"`` (uniform random). Exclusion set is the union of
        the validator's cooldown set and our local blacklist.
        """
        self._reset_stats_if_stale(local_n)
        excluded = cooldown_set | self._blacklisted_prompts

        if rng.random() < self._EXPLORE_RATE:
            return pick_prompt_idx(self.env, excluded, rng=rng), "explore", None, 0

        candidates: list[int] = []
        weights: list[float] = []
        for idx, hist in self._prompt_sigma_history.items():
            if idx in excluded or len(hist) < self._STATS_MIN_OBS:
                continue
            mean_sigma = sum(hist) / len(hist)
            if mean_sigma >= self._SIGMA_THRESHOLD:
                candidates.append(idx)
                weights.append(mean_sigma)

        pool_size = len(candidates)
        if pool_size >= self._MIN_FRONTIER_POOL:
            chosen = rng.choices(candidates, weights=weights, k=1)[0]
            chosen_mean = weights[candidates.index(chosen)]
            return chosen, "exploit", chosen_mean, pool_size

        # Level-biased fallback: weight levels by their mean σ. A level
        # with mean σ = 0 (consistently all-correct or all-wrong) gets
        # weight 0; a level near σ=0.5 gets the highest weight. This
        # concentrates picks where the policy is most learnable instead
        # of wasting attempts on levels Qwen3-4B has already mastered.
        level_choice = self._pick_by_level_prior(excluded, rng)
        if level_choice is not None:
            return level_choice, "level_biased", None, pool_size
        return pick_prompt_idx(self.env, excluded, rng=rng), "fallback", None, pool_size

    def _pick_by_level_prior(
        self, excluded: set[int], rng: _random.Random
    ) -> int | None:
        """Pick a prompt by difficulty-level prior.

        Combines a static prior (level 5 hardest → highest weight) with the
        observed mean σ per level. The static prior dominates when we have
        few observations; observations take over as they accumulate. Without
        the static prior, all picks return σ=0 initially → all level weights
        stay at 0 → picker falls back to uniform-random over all prompts,
        which keeps hitting trivially-easy level 1–2 prompts that Qwen3-4B
        solves 8/8.
        """
        levels_by_pid = self._build_prompts_by_level()
        if not levels_by_pid:
            return None

        STATIC_PRIOR = {1: 0.10, 2: 0.20, 3: 0.50, 4: 0.90, 5: 1.00}
        levels = sorted(levels_by_pid.keys())
        weights: list[float] = []
        for lv in levels:
            hist = self._level_sigma_history.get(lv)
            n_obs = len(hist) if hist else 0
            observed_mean = (sum(hist) / n_obs) if n_obs else 0.0
            # Observation weight ramps from 0 → 0.7 over ~14 obs. Below that
            # the static prior keeps the picker pointed at hard levels.
            alpha = min(0.7, n_obs / 20.0)
            prior = STATIC_PRIOR.get(lv, 0.5)
            weights.append(alpha * observed_mean + (1.0 - alpha) * prior)

        if not any(w > 0 for w in weights):
            return None
        chosen_level = rng.choices(levels, weights=weights, k=1)[0]
        level_prompts = levels_by_pid[chosen_level]
        eligible = [p for p in level_prompts if p not in excluded]
        if not eligible:
            return None
        return rng.choice(eligible)
