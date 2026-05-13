"""vLLM adapter (v4 logging-hardened, v4.2 log-trim).

DEPLOYMENT:

    cp vllm_adapter-v4.py /root/reliquary/vllm_adapter.py

Drop-in replacement for v3's ``vllm_adapter.py``. Behaviour-compatible:
the engine still detects this adapter via the ``_is_vllm_adapter``
sentinel and calls ``generate(...)`` / ``reload(...)`` exactly the same
way. Only the logging changed.

What v4 fixes (vs v3)
=====================

1. **No build heartbeat.** v3's ``LLM(...)`` constructor (60-90 s on
   H200 + Qwen3-4B) emitted nothing between begin and done — a miner
   box looked dead for over a minute every checkpoint reload, often
   triggering operators to ``kill -9`` healthy processes. v4 spawns a
   heartbeat thread (2 s ticks for the first 20 s, 5 s thereafter)
   while ``LLM(...)`` runs.

v4.2 log-trim (current)
=======================

The engine-v4 path now wraps every ``generate(...)`` call in
``asyncio.to_thread`` and emits its own ``GEN prompt=N gen=Ms`` line
with the same timing info. The adapter's per-generate logging is
therefore redundant and was actively hurting time-to-submit on
chatty windows (each duplicate log line costs ~50-200 µs of CPU and
fights for the same single-threaded asyncio reactor that's trying
to ship the previous submit).

This release REMOVES:

- ``vllm.generate begin/done`` log lines (engine's GEN line replaces).
- The ``generate()`` heartbeat thread (the call is now off-loop, so
  there's nothing to ``kill -9`` mid-generate).
- All ``_stderr_print`` raw-fd duplicates on hot paths — bittensor
  no longer clobbers handlers in the supported versions, and the
  ``FlushingStreamHandler`` configured by ``setup_logging`` reliably
  reaches stderr on its own.
- The transformers-shim "patched ..." INFO line (moved to DEBUG; the
  shim is idempotent and an operator only needs to see it when
  debugging vLLM init).

This release KEEPS:

- ``vLLM build start/done`` INFO (once per session + per reload).
- ``vLLM reload: A → B`` INFO (once per checkpoint).
- The build/reload heartbeat (operators need a sign of life during
  the 60-90 s ``LLM(...)`` construct).
- Error / exception logging on the generate path.

All other paths (transformers shim, n=k batched sampling, padding)
are unchanged.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


# Build/reload heartbeat cadence. Two phases:
#   - "fresh" : first ``_HEARTBEAT_FRESH_DURATION_S`` seconds use the fast
#               interval so an operator immediately sees motion.
#   - "settled" : afterwards we back off to a slower cadence to avoid
#                 log spam during legitimately long completions.
#
# Only used during ``LLM(...) construct`` (build/reload paths). The
# generate() path runs off-loop via asyncio.to_thread and is logged by
# the engine's GEN line — no per-generate heartbeat thread is started.
_HEARTBEAT_FRESH_INTERVAL_S = 2.0
_HEARTBEAT_SETTLED_INTERVAL_S = 5.0
_HEARTBEAT_FRESH_DURATION_S = 20.0


class _Heartbeat:
    """Thread that emits a periodic progress line while a blocking call runs.

    Usage::

        with _Heartbeat("LLM(...) construct", fresh_interval_s=2.0):
            self._llm = LLM(...)

    Emits::

        heartbeat label='LLM(...) construct' tick=1 elapsed=2.0s
        heartbeat label='LLM(...) construct' tick=2 elapsed=4.0s
        ...

    Single emit channel per tick: the module logger at ``tick_level``.
    v4.2 dropped the raw-stderr-fd duplicate path; the
    ``FlushingStreamHandler`` configured by ``setup_logging`` reliably
    reaches stderr on its own under the supported bittensor versions.
    """

    def __init__(
        self,
        label: str,
        *,
        fresh_interval_s: float = _HEARTBEAT_FRESH_INTERVAL_S,
        settled_interval_s: float = _HEARTBEAT_SETTLED_INTERVAL_S,
        fresh_duration_s: float = _HEARTBEAT_FRESH_DURATION_S,
        extra_fn=None,
        tick_level: int = logging.INFO,
        done_ok_level: int = logging.INFO,
    ) -> None:
        self.label = label
        self.fresh_interval_s = fresh_interval_s
        self.settled_interval_s = settled_interval_s
        self.fresh_duration_s = fresh_duration_s
        self.extra_fn = extra_fn  # optional () -> str for trailing context
        self.tick_level = tick_level
        self.done_ok_level = done_ok_level
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def __enter__(self) -> "_Heartbeat":
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    "heartbeat thread for %r did not join after 5s",
                    self.label,
                )
        elapsed = time.monotonic() - self._t0
        if exc_type is None:
            logger.log(
                self.done_ok_level,
                "heartbeat label='%s' state=done elapsed=%.1fs",
                self.label, elapsed,
            )
        else:
            logger.error(
                "heartbeat label='%s' state=fail elapsed=%.1fs exc=%s: %s",
                self.label, elapsed, exc_type.__name__, exc,
            )

    def _interval(self, elapsed: float) -> float:
        if elapsed < self.fresh_duration_s:
            return self.fresh_interval_s
        return self.settled_interval_s

    def _run(self) -> None:
        tick = 0
        while True:
            elapsed = time.monotonic() - self._t0
            interval = self._interval(elapsed)
            # Wait returns True if stop was set; False on timeout.
            if self._stop.wait(interval):
                return
            tick += 1
            elapsed = time.monotonic() - self._t0
            extra = ""
            if self.extra_fn is not None:
                try:
                    extra_str = self.extra_fn()
                    if extra_str:
                        extra = " " + extra_str
                except Exception:
                    pass
            logger.log(
                self.tick_level,
                "heartbeat label='%s' tick=%d elapsed=%.1fs%s",
                self.label, tick, elapsed, extra,
            )


# ---------------------------------------------------------------------------
# Transformers shim (carried verbatim from v3)
# ---------------------------------------------------------------------------

def _patch_transformers_for_vllm() -> None:
    """Restore ``all_special_tokens_extended`` on tokenizer base classes.

    vLLM ≤ 0.7.x calls ``tokenizer.all_special_tokens_extended``, which
    transformers removed in 4.50+. Without this shim, vLLM crashes during
    LLM(...) init with AttributeError. Idempotent.
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

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(_get, _set)
            patched.append("PreTrainedTokenizerBase")
    except Exception:
        logger.exception("could not patch PreTrainedTokenizerBase")

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

    try:
        from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
        if not hasattr(PreTrainedTokenizerFast, "all_special_tokens_extended"):
            PreTrainedTokenizerFast.all_special_tokens_extended = property(_get, _set)
            patched.append("PreTrainedTokenizerFast")
    except Exception:
        pass

    for mod_path, cls_names in [
        ("transformers.models.qwen2", ["Qwen2Tokenizer", "Qwen2TokenizerFast"]),
        ("transformers.models.qwen", ["QwenTokenizer"]),
    ]:
        for cls_name in cls_names:
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
        logger.debug(
            "patched all_special_tokens_extended on: %s "
            "(vLLM 0.7.x compat with transformers ≥ 4.50)",
            ", ".join(patched),
        )
    else:
        logger.debug(
            "all_special_tokens_extended patch not needed "
            "(already available or not used by vLLM)"
        )


_patch_transformers_for_vllm()


# ---------------------------------------------------------------------------
# GPU memory helper (used by heartbeat extra context)
# ---------------------------------------------------------------------------

def _gpu_mem_str(gpu_id: int) -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        free, total = torch.cuda.mem_get_info(gpu_id)
        used = total - free
        return (
            f"gpu_mem=used:{used / (1024 ** 3):.1f}GB/"
            f"free:{free / (1024 ** 3):.1f}GB/"
            f"total:{total / (1024 ** 3):.1f}GB"
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# VLLMAdapter
# ---------------------------------------------------------------------------

class VLLMAdapter:
    """Wrap a vLLM ``LLM`` instance behind the HF ``.generate()`` API.

    Behaviour-compatible with v3: same constructor signature, same
    ``generate()`` shape, same ``_is_vllm_adapter`` sentinel detected
    by ``MiningEngine._load_checkpoint``.
    """

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
        kv_cache_dtype: str | None = None,
        speculative_model: str | None = None,
        num_speculative_tokens: int = 0,
        attention_backend: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._gpu_id = gpu_id
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dtype = dtype
        self._enforce_eager = enforce_eager
        # Tier 1: FP8 KV cache halves attention memory and lets a larger
        # batch fit in the pre-allocated KV pool. "fp8" works on Hopper/
        # Blackwell; "auto"/None falls back to model dtype.
        self._kv_cache_dtype = kv_cache_dtype
        # Passed to LLM(...) directly; setting the legacy
        # VLLM_ATTENTION_BACKEND env var is silently ignored in vLLM v1.
        self._attention_backend = attention_backend
        # Tier 2: speculative decoding via a small draft model. The draft
        # proposes `num_speculative_tokens` tokens per step; the target
        # model verifies all of them in one forward pass. Typical
        # speedup 1.5-2.5x on long decodes. Set both to enable; either
        # missing disables the feature.
        self._speculative_model = speculative_model
        self._num_speculative_tokens = num_speculative_tokens

        self.device = f"cuda:{gpu_id}"

        self._llm: Any | None = None
        self._build(model_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build(self, model_path: str) -> None:
        from vllm import LLM

        spec_enabled = bool(
            self._speculative_model and self._num_speculative_tokens > 0
        )
        logger.info(
            "vLLM build start: model=%s dtype=%s max_model_len=%d "
            "gpu_mem_util=%.2f enforce_eager=%s kv_cache_dtype=%s "
            "attn=%s spec=%s gpu=cuda:%d %s",
            model_path, self._dtype, self._max_model_len,
            self._gpu_memory_utilization, self._enforce_eager,
            self._kv_cache_dtype or "auto",
            self._attention_backend or "auto",
            (
                f"{self._speculative_model.split('/')[-1]}@k={self._num_speculative_tokens}"
                if spec_enabled
                else "off"
            ),
            self._gpu_id, _gpu_mem_str(self._gpu_id),
        )

        gpu_id = self._gpu_id

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "dtype": self._dtype,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "max_model_len": self._max_model_len,
            "enforce_eager": self._enforce_eager,
            "disable_log_stats": True,
            # Math env reuses the same chat template + system prompt across
            # all 12,500 prompts, so prefix caching turns repeated prefill
            # work into a one-time cost. Chunked prefill is required for
            # FlashInfer to interleave prefill with decode efficiently and
            # is a no-op on workloads without long prompts.
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        }
        if self._kv_cache_dtype and self._kv_cache_dtype.lower() != "auto":
            llm_kwargs["kv_cache_dtype"] = self._kv_cache_dtype
        if self._attention_backend:
            llm_kwargs["attention_backend"] = self._attention_backend
        # Speculative decoding API depends on vLLM version:
        #   - vLLM <0.7: flat kwargs ``speculative_model`` + ``num_speculative_tokens``
        #   - vLLM >=0.7: nested dict ``speculative_config={"model":..., "num_speculative_tokens":...}``
        # We attempt the modern dict-based form first; if vLLM raises a
        # TypeError signalling the old API, fall back to flat kwargs.
        spec_kwargs_modern: dict[str, Any] | None = None
        spec_kwargs_legacy: dict[str, Any] | None = None
        if spec_enabled:
            spec_kwargs_modern = {
                "speculative_config": {
                    "model": self._speculative_model,
                    "num_speculative_tokens": self._num_speculative_tokens,
                },
            }
            spec_kwargs_legacy = {
                "speculative_model": self._speculative_model,
                "num_speculative_tokens": self._num_speculative_tokens,
            }

        with _Heartbeat(
            f"LLM(...) construct model={model_path.split('/')[-1]} "
            f"gpu=cuda:{gpu_id}",
            extra_fn=lambda: _gpu_mem_str(gpu_id),
        ):
            if spec_enabled and spec_kwargs_modern is not None:
                try:
                    self._llm = LLM(**llm_kwargs, **spec_kwargs_modern)
                except TypeError as e:
                    if "speculative_config" in str(e):
                        logger.warning(
                            "vLLM rejected speculative_config dict; "
                            "falling back to legacy flat kwargs. err=%s", e,
                        )
                        self._llm = LLM(**llm_kwargs, **spec_kwargs_legacy)
                    else:
                        raise
            else:
                self._llm = LLM(**llm_kwargs)

        self._model_path = model_path
        logger.info(
            "vLLM build done: ready on %s %s",
            self.device, _gpu_mem_str(self._gpu_id),
        )

    def reload(self, local_path: str) -> None:
        """Rebuild the LLM pointing at a new checkpoint.

        vLLM has no in-place weight swap, so we tear down and rebuild.
        The ~30-60s cost is amortised over CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
        windows (the validator only republishes every 10 windows).
        """
        logger.info("vLLM reload: %s → %s", self._model_path, local_path)

        if self._llm is not None:
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

        Same call surface as v3. The engine sends ``(n, prompt_len)``
        rows with identical prompts; we collapse via ``SamplingParams(n=k)``
        which shares the prompt's KV cache across all k samples. Output
        is repacked into ``(n, prompt_len + max_gen_len)`` right-padded
        with ``pad_token_id`` so the engine's per-row EOS truncation
        works unchanged.
        """
        import torch
        from vllm import SamplingParams

        if self._llm is None:
            raise RuntimeError("vLLM not built — reload() failed?")

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

        # v4.2: per-generate begin/done/heartbeat logging removed. The
        # engine emits its own ``GEN prompt=N gen=Ms`` line with the
        # same timing info, and ``asyncio.to_thread`` keeps the asyncio
        # reactor responsive while the call runs — no more "operator
        # thinks the process is dead and kill -9's it" failure mode that
        # the v4 heartbeat was guarding against.
        t0 = time.monotonic()
        try:
            # vLLM ≥ 0.9 removed the ``prompt_token_ids=`` kwarg; pass
            # prompt as a TokensPrompt-shaped dict. Dict form works
            # across 0.8.x–0.20.x without importing TokensPrompt
            # (its module path changed between versions).
            outputs = self._llm.generate(
                [{"prompt_token_ids": prompt_tokens}],
                sampling_params=params,
                use_tqdm=False,
            )
        except Exception:
            logger.exception(
                "vllm.generate raised after %.1fs",
                time.monotonic() - t0,
            )
            raise

        request_output = outputs[0]

        rows: list[list[int]] = []
        for completion in request_output.outputs:
            rows.append(prompt_tokens + list(completion.token_ids))

        if len(rows) != n:
            logger.warning(
                "vLLM returned %d samples for n=%d request; padding short rows",
                len(rows), n,
            )
            while len(rows) < n:
                rows.append(list(prompt_tokens))

        max_len = max(len(r) for r in rows)
        pad_id = pad_token_id if pad_token_id is not None else 0
        padded = [r + [pad_id] * (max_len - len(r)) for r in rows]
        return torch.tensor(padded, device=self.device)

