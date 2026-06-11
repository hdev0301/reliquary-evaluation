"""vLLM 0.20.x generation backend for the miner.

vLLM is used for **token sampling only**. The GRAIL proof and the submitted
``token_logprobs`` are always computed by the bit-identical HuggingFace forward
pass (see ``MiningEngine._build_grail_commit``), because the validator
recomputes both with *its own* HF forward on the submitted token ids — it never
inspects the sampler. The verifier code even guards against "vLLM->HF drift"
explicitly (validator/verifier.py), so vLLM-gen + HF-proof is a supported miner
architecture. vLLM's only job is to make the expensive autoregressive sampling
fast (continuous batching + paged attention), so pregeneration can stay far
ahead of the windows.

Single-GPU note: the proof HF model shares the GPU with vLLM. Initialise the
vLLM engine *first* with a bounded ``gpu_memory_utilization`` (default 0.55),
then load the HF proof copy into the remainder — see cli/main.py.
"""

from __future__ import annotations

import logging
import os as _os
from dataclasses import dataclass

# vLLM V1 launches its EngineCore in a child process. The miner parent has
# already initialised CUDA (HF proof model, device_count probe), so a *forked*
# child fails with "Cannot re-initialize CUDA in forked subprocess". Force the
# spawn start method (set before vllm is ever imported).
_os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from reliquary.constants import (
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratedGroup:
    """One prompt's group of sampled completions.

    ``completions`` is a list of completion-token-id lists. Each is normalised
    to end with **exactly one** EOS token (matching the reference miner's
    ``gen[:first_eos + 1]``) so the validator's ``verify_termination`` /
    ``has_eos_padding`` invariants hold. Cap-truncated (non-EOS) completions are
    dropped before this point.
    """

    prompt_idx: int
    prompt_token_ids: list[int]
    completions: list[list[int]]


class VLLMGenerator:
    """Thin wrapper over ``vllm.LLM`` for batched GRPO-group sampling."""

    def __init__(
        self,
        model_path: str,
        *,
        eos_token_ids: set[int] | None = None,
        oversample: int = M_ROLLOUTS,
        gpu_memory_utilization: float = 0.55,
        max_model_len: int | None = None,
        dtype: str = "bfloat16",
        seed: int | None = None,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = True,
    ) -> None:
        # Sample this many completions per prompt; we keep the first M_ROLLOUTS
        # that terminate naturally with EOS. Over-sampling (oversample > 8) is
        # needed because this checkpoint is a long-CoT reasoner with a low
        # per-rollout EOS-termination rate — selecting on FORMAT (termination),
        # not reward, so it is not sigma-shaping.
        self.oversample = max(int(oversample), M_ROLLOUTS)
        try:
            from vllm import LLM  # noqa: F401
        except ImportError as e:  # pragma: no cover - exercised only at runtime
            raise ImportError(
                "vLLM 0.20.x is required for the generation backend "
                "(`pip install 'vllm>=0.20,<0.21'`)."
            ) from e

        self.model_path = model_path
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._seed = seed
        self._enforce_eager = enforce_eager
        self._enable_prefix_caching = enable_prefix_caching
        # Concurrency cap. vLLM defaults max_num_seqs=128 here, but the KV cache
        # supports ~188 concurrent at our request length — raising the cap lets
        # continuous batching saturate the engine for ~1.5x discovery throughput.
        # 0/None = leave vLLM default.
        import os as _o
        _mns = int(_o.environ.get("RELIQUARY_MAX_NUM_SEQS", "192"))
        self._max_num_seqs = _mns if _mns > 0 else None

        self.llm = self._build_llm(model_path)
        # The validator resolves its EOS set from generation_config first, then
        # tokenizer (verifier._eos_set_from_model). vLLM stops on the model's
        # generation_config EOS, which may differ from tokenizer.eos_token_id
        # (Qwen3-Instruct: <|im_end|> vs <|endoftext|>). Honour the caller's
        # resolved set so we re-append the exact token the model stopped on.
        tok_eos = self._resolve_tokenizer_eos_id()
        self.eos_token_ids: set[int] = set(eos_token_ids) if eos_token_ids else {tok_eos}
        self.eos_token_ids.add(tok_eos)
        self.default_eos_id = tok_eos
        logger.info(
            "vLLM generator ready: model=%s gpu_mem_util=%.2f eos_set=%s",
            model_path, gpu_memory_utilization, sorted(self.eos_token_ids),
        )

    def _build_llm(self, model_path: str):
        from vllm import LLM

        kwargs: dict = dict(
            model=model_path,
            tokenizer=model_path,
            dtype=self._dtype,
            gpu_memory_utilization=self._gpu_memory_utilization,
            enforce_eager=self._enforce_eager,
            enable_prefix_caching=self._enable_prefix_caching,
            tensor_parallel_size=1,
        )
        if self._max_model_len is not None:
            kwargs["max_model_len"] = self._max_model_len
        if self._seed is not None:
            kwargs["seed"] = self._seed
        if self._max_num_seqs is not None:
            kwargs["max_num_seqs"] = self._max_num_seqs
        kwargs.update(self._multimodal_kwargs(model_path))
        return LLM(**kwargs)

    @staticmethod
    def _multimodal_kwargs(model_path: str) -> dict:
        """Adjustments for multimodal-wrapper checkpoints (e.g. Qwen3.5) used text-only.

        (1) limit_mm_per_prompt={image:0,video:0} -> vLLM skips loading an image
            processor the text-only snapshot lacks, and runs the language model only.
        (2) enforce_eager=True. vLLM's CUDA-graph capture path reads config.vocab_size,
            which a multimodal wrapper nests under text_config (top-level absent ->
            AttributeError). Surfacing it via hf_overrides made vLLM MIS-IDENTIFY the
            model as plain text-Qwen3 and mis-load the weights (vocab mismatch -> garbage
            generation). Eager mode skips that capture path, loads the correct multimodal
            class, and generates correctly (verified: solves prompts at reward 1.0).
            Tradeoff: no CUDA graphs, but long thinking decodes are compute-bound anyway.
        (Qwen3.5 linear-attn kernels also require `ninja` on PATH for the load JIT.)
        """
        import json as _json, os as _os
        try:
            cfg = _json.load(open(_os.path.join(model_path, "config.json")))
        except Exception:
            return {}
        if not (("vision_config" in cfg) or ("image_token_id" in cfg)):
            return {}
        return {"limit_mm_per_prompt": {"image": 0, "video": 0}, "enforce_eager": True}

    def _resolve_tokenizer_eos_id(self) -> int:
        tok = self.llm.get_tokenizer()
        eos = getattr(tok, "eos_token_id", None)
        if eos is None:
            raise RuntimeError("tokenizer exposes no eos_token_id; cannot enforce termination")
        return int(eos)

    def _sampling_params(self, max_new_tokens: int, oversample: int | None = None):
        from vllm import SamplingParams

        # Protocol TOP_K_PROTO=0 means "disabled"; vLLM spells that -1.
        top_k = -1 if TOP_K_PROTO <= 0 else TOP_K_PROTO
        return SamplingParams(
            n=(oversample if oversample is not None else self.oversample),
            temperature=T_PROTO,
            top_p=TOP_P_PROTO,
            top_k=top_k,
            max_tokens=max_new_tokens,
            # Stop on the FULL validator EOS set (both <|im_end|> and <|endoftext|>),
            # not just the model's generation_config default — otherwise the model
            # can emit the "other" EOS mid-stream, vLLM keeps going, and that interior
            # EOS survives into the submission -> validator has_eos_padding -> reject.
            stop_token_ids=sorted(self.eos_token_ids),
            # Natural EOS termination only (ignore_eos=False is the default);
            # we keep cap-hit completions out in the post-filter.
            detokenize=False,
        )

    def _normalise_completion(self, token_ids: list[int], stop_token: int | None) -> list[int]:
        """Force the completion to end with exactly one EOS token.

        vLLM strips the natural EOS that triggered the stop; the protocol wants
        it present as the final token (and forbids any tokens after it). Drop
        any trailing EOS-set tokens, then append the *exact* token the model
        stopped on (``stop_token`` from vLLM's ``stop_reason``) so p_stop is
        measured on the EOS the model truly emitted.
        """
        gen = list(token_ids)
        while gen and gen[-1] in self.eos_token_ids:
            gen.pop()
        eos = stop_token if (isinstance(stop_token, int) and stop_token in self.eos_token_ids) else self.default_eos_id
        gen.append(eos)
        return gen

    def generate_groups(
        self,
        prompts: list[tuple[int, list[int]]],
        *,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        oversample: int | None = None,
    ) -> list[GeneratedGroup]:
        """Sample ``M_ROLLOUTS`` completions for each ``(prompt_idx, token_ids)``.

        All prompts are submitted in a single ``llm.generate`` call so vLLM's
        continuous batching saturates the GPU. Only EOS-terminated completions
        are returned (``finish_reason == "stop"`` on the model EOS); cap
        truncations and any completion shorter than is useful are dropped — the
        pregeneration zone/safety filter handles the rest.
        """
        if not prompts:
            return []
        sp = self._sampling_params(max_new_tokens, oversample)
        # The {"prompt_token_ids": [...]} dict IS vLLM's TokensPrompt TypedDict;
        # passing it directly avoids version-specific import paths.
        vllm_prompts = [{"prompt_token_ids": ids} for _, ids in prompts]
        outputs = self.llm.generate(vllm_prompts, sp)

        groups: list[GeneratedGroup] = []
        for (prompt_idx, prompt_ids), out in zip(prompts, outputs):
            completions: list[list[int]] = []
            for comp in out.outputs:
                # Keep only completions the model actually ended (EOS), not cap
                # truncations — those risk BAD_TERMINATION and waste a slot.
                if comp.finish_reason == "length":
                    continue
                ids = list(comp.token_ids)
                if not ids:
                    continue
                stop_token = comp.stop_reason if isinstance(comp.stop_reason, int) else None
                completions.append(self._normalise_completion(ids, stop_token))
            groups.append(
                GeneratedGroup(
                    prompt_idx=prompt_idx,
                    prompt_token_ids=list(prompt_ids),
                    completions=completions,
                )
            )
        return groups

    def reload(self, model_path: str) -> None:
        """Swap to a new checkpoint by rebuilding the engine.

        Called when the validator publishes a new ``checkpoint_n``. A full
        rebuild is the robust path (vLLM's in-place ``load_weights`` RPC is
        faster but version-sensitive); it only happens every
        ``CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`` windows.
        """
        import gc

        logger.info("vLLM reload: %s -> %s", self.model_path, model_path)
        old = self.llm
        self.llm = None
        del old
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        self.model_path = model_path
        self.llm = self._build_llm(model_path)
        # Re-establish the EOS set against the rebuilt engine, mirroring __init__
        # (there is no _resolve_eos_id / self.eos_id — the class uses the
        # eos_token_ids SET + default_eos_id; the old line referenced names that
        # never existed and crashed every checkpoint reload). The reload target is
        # the same repo/revision (new checkpoint_n only), so the tokenizer EOS is
        # unchanged; re-resolve + re-merge to stay robust if it ever differs.
        tok_eos = self._resolve_tokenizer_eos_id()
        self.eos_token_ids.add(tok_eos)
        self.default_eos_id = tok_eos
        logger.info(
            "vLLM reload complete: model=%s eos_set=%s",
            model_path, sorted(self.eos_token_ids),
        )
