"""vLLM adapter that exposes HuggingFace's ``.generate()`` API.

The Reliquary miner engine calls ``self.vllm_model.generate(input_tensor,
max_new_tokens=..., do_sample=True, temperature=..., top_p=..., top_k=...,
pad_token_id=...)`` — the HuggingFace ``AutoModelForCausalLM.generate``
signature — and expects back a 2-D ``torch.Tensor`` of shape
``(n, prompt_len + max_gen_len)`` padded with ``pad_token_id``.

This adapter wraps a real vLLM ``LLM`` instance and translates that
call into vLLM's batched-sampling API (``n=k`` shares the prompt KV
cache across all ``k`` samples — much more efficient than HF's
batch-of-identical-prompts approach). The result is repacked into the
tensor shape MiningEngine expects, so engine code is **unchanged on the
call site** — only ``_load_checkpoint`` needs a small detect-and-dispatch
addition (handled by ``_is_vllm_adapter``).

Why an adapter rather than rewriting the engine: keeps the HF path
fully functional as a fallback (delete the launcher / pass an HF model
and the engine reverts to HF generation), and contains the vLLM-specific
quirks (top_k semantics, output unpacking, checkpoint reload) in one file.

Compatibility notes:
- vLLM uses ``top_k=-1`` to disable; HF uses ``top_k=0``. Translated here.
- vLLM's ``CompletionOutput.token_ids`` already excludes the prompt and
  stops at EOS / max_tokens. The engine's post-EOS truncation is a no-op
  on vLLM output but kept for HF-fallback safety.
- ``reload(local_path)`` rebuilds the LLM in-place. vLLM doesn't expose
  a clean hot-reload primitive, so we destroy and recreate; the
  ~30-60s cost is amortised over CHECKPOINT_PUBLISH_INTERVAL_WINDOWS=10
  windows (the validator only republishes every 10 windows).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _patch_transformers_for_vllm() -> None:
    """Restore ``all_special_tokens_extended`` on the tokenizer base class.

    vLLM ≤ 0.7.x's ``get_cached_tokenizer`` accesses
    ``tokenizer.all_special_tokens_extended``, which transformers removed
    in 4.50+. Without this shim, vLLM crashes during ``LLM(...)`` init::

        AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended

    Critical: we patch ``PreTrainedTokenizerBase`` (in
    ``transformers.tokenization_utils_base``), NOT ``PreTrainedTokenizer``.
    In newer transformers, slow tokenizers like ``Qwen2Tokenizer`` don't
    have ``PreTrainedTokenizer`` in their MRO directly — the shim only
    takes effect when attached to ``PreTrainedTokenizerBase``, which IS
    the deepest common base (and the same class whose ``__getattr__``
    raised the AttributeError in the user's traceback). Patching the
    wrong class silently appears to succeed but doesn't actually inject
    the property into instances.

    Patches BOTH slow and fast tokenizer code paths via the shared base
    so Qwen2Tokenizer, Qwen2TokenizerFast, etc. all pick it up.

    Why a shim and not a transformers downgrade: the user's Qwen3-4B
    model needs transformers ≥ 4.51 for the ``Qwen3ForCausalLM``
    architecture to register with ``AutoModelForCausalLM``. Pinning
    ``transformers < 4.50`` would unbreak vLLM but break the HF GRAIL
    proof model load — a worse trade.

    The fallback value is the plain ``all_special_tokens`` list — what
    vLLM iterates anyway. The setter stashes assigned values in a
    private slot so subsequent reads return what was written (vLLM's
    ``get_cached_tokenizer`` both reads from the original tokenizer AND
    writes to the cached copy, so the property needs both directions).

    Wrapped in defensive try/except — ``self.all_special_tokens`` itself
    could theoretically raise on stripped-down tokenizers; the fallback
    to ``[]`` keeps vLLM init alive at the cost of an empty special-token
    cache (vLLM still works, just doesn't get the cache-fast-path
    optimisation for those tokens).

    Idempotent: no-op when the attribute already exists.
    """
    def _get(self):
        try:
            return getattr(
                self, "_vllm_shim_extended", list(self.all_special_tokens),
            )
        except Exception:
            return []

    def _set(self, value):
        self._vllm_shim_extended = value

    patched: list[str] = []

    # Primary target: the actual deepest base class. This is what's in
    # every tokenizer's MRO, slow or fast, and is the class whose
    # __getattr__ raised the error in the user's trace.
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(_get, _set)
            patched.append("PreTrainedTokenizerBase")
    except Exception:
        logger.exception("could not patch PreTrainedTokenizerBase")

    # Belt-and-suspenders: also patch the slow/fast subclass bases in
    # case some weird transformers version has shadowed the base's
    # property in a subclass. Cheap and idempotent.
    for mod_path, cls_name in (
        ("transformers", "PreTrainedTokenizer"),
        ("transformers", "PreTrainedTokenizerFast"),
    ):
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            if "all_special_tokens_extended" not in cls.__dict__:
                cls.all_special_tokens_extended = property(_get, _set)
                patched.append(cls_name)
        except Exception:
            pass

    if patched:
        logger.info(
            "patched all_special_tokens_extended on: %s "
            "(vLLM 0.7.x compat with transformers ≥ 4.50)",
            ", ".join(patched),
        )


# Apply the patch at module import — before any vLLM imports happen
# transitively. Safe to run multiple times (idempotent).
_patch_transformers_for_vllm()


class VLLMAdapter:
    """Wrap a vLLM ``LLM`` instance behind the HF ``.generate()`` API.

    Constructed once per miner session. Subsequent checkpoint pulls go
    through ``reload(local_path)`` which destroys the underlying LLM and
    builds a new one on the same GPU.
    """

    # Sentinel attribute the engine checks via ``hasattr`` to decide whether
    # to use the adapter's ``reload()`` (vLLM rebuild) or the standard HF
    # ``AutoModelForCausalLM.from_pretrained`` reload path in
    # ``_load_checkpoint``. Don't rename without updating the engine.
    _is_vllm_adapter: bool = True

    def __init__(
        self,
        model_path: str,
        *,
        gpu_id: int = 0,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
    ) -> None:
        self._model_path = model_path
        self._gpu_id = gpu_id
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dtype = dtype
        self._enforce_eager = enforce_eager

        # The engine reads ``model.device`` to position the input tensor
        # before calling .generate(). vLLM doesn't expose this attribute,
        # so we pin it manually. By convention vLLM uses cuda:0 (the first
        # device visible to the process), so for a 2-GPU box where vLLM is
        # on the physical GPU mapped to cuda:0 the launcher should ensure
        # the proof model lives on cuda:1.
        self.device = f"cuda:{gpu_id}"

        self._llm: Any | None = None
        self._build(model_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build(self, model_path: str) -> None:
        from vllm import LLM
        logger.info(
            "vLLM build: model=%s dtype=%s max_model_len=%d gpu_mem_util=%.2f enforce_eager=%s",
            model_path, self._dtype, self._max_model_len,
            self._gpu_memory_utilization, self._enforce_eager,
        )
        self._llm = LLM(
            model=model_path,
            dtype=self._dtype,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_model_len=self._max_model_len,
            enforce_eager=self._enforce_eager,
            disable_log_stats=True,
        )
        self._model_path = model_path
        logger.info("vLLM ready on %s", self.device)

    def reload(self, local_path: str) -> None:
        """Rebuild the LLM pointing at a new checkpoint.

        Called by ``MiningEngine._load_checkpoint`` when the validator
        publishes a new checkpoint revision. vLLM doesn't support
        in-place weight swap cleanly, so we tear down and rebuild. The
        ~30–60s cost is acceptable because checkpoint pulls happen at
        most every CHECKPOINT_PUBLISH_INTERVAL_WINDOWS windows.
        """
        logger.info("vLLM reload: %s → %s", self._model_path, local_path)
        if self._llm is not None:
            # vLLM holds references to CUDA memory in the executor; explicit
            # del + empty_cache before the new build keeps the next
            # allocation from running into fragmentation on the same GPU.
            old = self._llm
            self._llm = None
            del old
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
        self._build(local_path)

    # ------------------------------------------------------------------
    # HuggingFace-compatible generate
    # ------------------------------------------------------------------

    def generate(
        self,
        input_tensor,
        *,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        pad_token_id: int | None = None,
        **_ignored,
    ):
        """Run vLLM batched sampling and return an HF-shaped tensor.

        The engine sends ``input_tensor`` of shape ``(n, prompt_len)``
        with every row containing the SAME prompt token ids — HF batches
        identical prompts because that's how it expresses n-sample
        generation. vLLM expresses this natively via ``SamplingParams(n=k)``
        which shares the prompt's KV cache across all k samples (far
        more memory-efficient than k duplicated prompts).

        Returns a torch.Tensor of shape ``(n, prompt_len + max_gen_len)``
        right-padded with ``pad_token_id`` so per-row slicing
        (``outputs[i].tolist()[prompt_length:]``) works unchanged in the
        engine.

        Emits start/end INFO logs around the vLLM call itself with
        timing + token counts. The vLLM call is the longest single
        operation in the miner pipeline (10–60s typical) and emits no
        progress output of its own (``use_tqdm=False``), so without
        these markers the loop appears to hang for the entire duration.
        """
        import time
        import torch
        from vllm import SamplingParams

        if self._llm is None:
            raise RuntimeError("vLLM not built — reload() failed?")

        # Engine always sends identical rows. Take the first row as the
        # canonical prompt. Defensive: also check the assumption.
        if input_tensor.dim() != 2:
            raise ValueError(
                f"input_tensor must be 2D (n, prompt_len); got shape {tuple(input_tensor.shape)}"
            )
        prompt_tokens = input_tensor[0].tolist()
        n = input_tensor.shape[0]

        # vLLM disables top-k with -1; HF disables with 0. Translate.
        vllm_top_k = top_k if top_k > 0 else -1

        params = SamplingParams(
            n=n,
            temperature=float(temperature) if do_sample else 0.0,
            top_p=float(top_p),
            top_k=vllm_top_k,
            max_tokens=int(max_new_tokens),
        )

        logger.info(
            "vllm.generate begin n=%d prompt_len=%d max_new_tokens=%d "
            "temperature=%.2f top_p=%.2f top_k=%d",
            n, len(prompt_tokens), max_new_tokens,
            temperature, top_p, vllm_top_k,
        )
        t0 = time.monotonic()
        outputs = self._llm.generate(
            prompt_token_ids=[prompt_tokens],
            sampling_params=params,
            use_tqdm=False,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        # outputs is list[RequestOutput] with one element (one prompt).
        request_output = outputs[0]
        gen_lens = [len(c.token_ids) for c in request_output.outputs]
        finish_reasons = [c.finish_reason for c in request_output.outputs]
        total_tokens = sum(gen_lens)
        toks_per_sec = (
            total_tokens / (elapsed_ms / 1000.0)
            if elapsed_ms > 0 else 0.0
        )
        logger.info(
            "vllm.generate done n=%d elapsed_ms=%.0f total_tokens=%d "
            "throughput=%.0f tok/s gen_lens=%s finish=%s",
            n, elapsed_ms, total_tokens, toks_per_sec,
            gen_lens, finish_reasons,
        )

        # Reconstruct (n, seq_len) tensor with right-padding so the
        # engine's per-row EOS slicing works unchanged.
        rows: list[list[int]] = []
        for completion in request_output.outputs:
            rows.append(prompt_tokens + list(completion.token_ids))

        if len(rows) != n:
            logger.warning(
                "vLLM returned %d samples for n=%d request; padding short rows",
                len(rows), n,
            )
            # Pad with empty completions (engine will see them as 0-length
            # and skip in the truncation step). Shouldn't happen in practice.
            while len(rows) < n:
                rows.append(list(prompt_tokens))

        max_len = max(len(r) for r in rows)
        pad_id = pad_token_id if pad_token_id is not None else 0
        padded = [r + [pad_id] * (max_len - len(r)) for r in rows]
        return torch.tensor(padded, device=self.device)
