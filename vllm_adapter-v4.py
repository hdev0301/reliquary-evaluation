"""vLLM adapter (v4 logging-hardened) — heartbeats during build/reload/generate.

DEPLOYMENT:

    cp vllm_adapter-v4.py /root/reliquary/vllm_adapter.py

Drop-in replacement for v3's ``vllm_adapter.py``. Behaviour-compatible:
the engine still detects this adapter via the ``_is_vllm_adapter``
sentinel and calls ``generate(...)`` / ``reload(...)`` exactly the same
way. Only the logging changed.

What v4 fixes
=============

The v3 adapter logged only at the *boundary* of each blocking call:

    vLLM build: model=... dtype=... ...
    [60-90 seconds of silence while LLM(...) captures CUDA graphs]
    vLLM ready on cuda:N

    vllm.generate begin n=8 prompt_len=412 max_new_tokens=8192
    [heartbeat every 10s during generate — fine]
    vllm.generate done elapsed_ms=24580 ...

There were two problems:

1. **No build heartbeat.** The ``LLM(...)`` constructor is the longest
   single blocking call in the entire miner pipeline (60-90 s on H200 +
   Qwen3-4B). v3 emitted nothing between begin and done — a miner box
   looked dead for over a minute every checkpoint reload, often
   triggering operators to ``kill -9`` perfectly healthy processes.

2. **Heartbeat too coarse for fast generation.** v3's 10 s heartbeat
   misses the short-end of the latency distribution: 12-15 s generations
   look identical to 8 s ones at INFO level. Fast race-mode picks (~5 s)
   never heartbeat at all.

v4:

- ``_build()`` and ``reload()`` spawn a heartbeat thread that emits
  every 2 s for the first 20 s, every 5 s thereafter. Includes the
  stage label ("LLM(...) constructor") so an operator sees exactly
  what's blocking.
- ``generate()`` heartbeat fires every 5 s (was 10 s) with the same
  back-off pattern. Once a generate is >30 s in, it slows to 10 s.
- All log lines reach a flushing handler (configured by main-v4's
  ``setup_logging``). Optional raw-fd duplicates: set
  ``RELIQUARY_RAW_STDERR=1`` if logging is still invisible (same flag
  as ``engine-v4`` ``_emit``).
- Routine ``vllm.generate`` heartbeats and begin/done noise are at
  DEBUG or trimmed so a normal PICK→GEN→SUB cycle stays ~3 INFO lines.

All other paths (transformers shim, n=k batched sampling, padding) are
unchanged.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


# Heartbeat cadence. Two phases:
#   - "fresh" : first ``_HEARTBEAT_FRESH_DURATION_S`` seconds use the fast
#               interval so an operator immediately sees motion.
#   - "settled" : afterwards we back off to a slower cadence to avoid
#                 log spam during legitimately long completions.
_HEARTBEAT_FRESH_INTERVAL_S = 2.0
_HEARTBEAT_SETTLED_INTERVAL_S = 5.0
_HEARTBEAT_FRESH_DURATION_S = 20.0

# Generate-call heartbeat is much looser than build/reload. Generations
# are routine work and the engine emits its own per-attempt PICK/GEN/SUB
# lines bracketing them, so a heartbeat is only useful when a generation
# is suspiciously slow. We skip the fresh-phase entirely (fresh_duration=0
# means we go straight to settled), so for typical 10-30 s generations
# the heartbeat thread emits zero ticks — only for >30 s slowdowns does
# the operator start seeing tick lines.
_GENERATE_HEARTBEAT_FRESH_INTERVAL_S = 30.0
_GENERATE_HEARTBEAT_SETTLED_INTERVAL_S = 30.0
_GENERATE_HEARTBEAT_FRESH_DURATION_S = 0.0


# Bulletproof direct-write fd captured at module import time — i.e.
# before bittensor / vllm / transformers had any chance to replace
# sys.stderr or sys.__stderr__. ``os.dup(2)`` returns a NEW file
# descriptor pointing at the underlying kernel stderr. Even if a
# third-party lib does ``dup2(pipe_write, 2)`` to redirect stderr (as
# bittensor's btlogging is known to do), our captured fd STILL points
# at the original stderr, so ``os.write(_FD, ...)`` cannot be
# intercepted or replayed. This is the only Python-level technique
# that survives arbitrary userspace stream wrapping.
try:
    _RAW_STDERR_FD: int | None = os.dup(2)
except OSError:
    _RAW_STDERR_FD = None

# Dedupe ring — drop identical (logger, msg) pairs emitted within
# this window. Defends against any duplication mechanism we haven't
# diagnosed yet (e.g. an upstream lib that hooks stderr at the C
# layer BEFORE our os.dup runs).
_DEDUPE_WINDOW_S = 0.5
_dedupe_last: dict[str, float] = {}

# Mirror ``engine-v4``: raw-fd duplicate is opt-in only.
_RAW_STDERR_FLUSH = os.environ.get(
    "RELIQUARY_RAW_STDERR", "",
).strip().lower() in ("1", "true", "yes")


def _stderr_print(msg: str) -> None:
    """Optional direct write to the kernel-level stderr fd.

    Disabled unless ``RELIQUARY_RAW_STDERR=1`` — the root
    ``FlushingStreamHandler`` is the normal path; duplicates were only
    useful when btlogging clobbered handlers.
    """
    if not _RAW_STDERR_FLUSH:
        return
    now = time.monotonic()
    last_t = _dedupe_last.get(msg)
    if last_t is not None and (now - last_t) < _DEDUPE_WINDOW_S:
        _dedupe_last[msg] = now
        return
    _dedupe_last[msg] = now
    if len(_dedupe_last) > 128:
        cutoff = now - 60.0
        for k in [k for k, t in _dedupe_last.items() if t < cutoff]:
            del _dedupe_last[k]
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        if _RAW_STDERR_FD is not None:
            os.write(_RAW_STDERR_FD, line.encode("utf-8", "replace"))
        else:
            stream = getattr(sys, "__stderr__", None) or sys.stderr
            stream.write(line)
            stream.flush()
    except Exception:
        pass


class _Heartbeat:
    """Thread that emits a periodic progress line while a blocking call runs.

    Usage::

        with _Heartbeat("LLM(...) construct", fresh_interval_s=2.0):
            self._llm = LLM(...)

    Emits::

        heartbeat label='LLM(...) construct' tick=1 elapsed=2.0s
        heartbeat label='LLM(...) construct' tick=2 elapsed=4.0s
        ...

    Two emit channels per tick (when raw flush is enabled):
      - ``logger`` at ``tick_level`` (default INFO)
      - optional raw fd write (``RELIQUARY_RAW_STDERR=1``)
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
            line = (
                f"heartbeat label='{self.label}' state=done "
                f"elapsed={elapsed:.1f}s"
            )
            logger.log(self.done_ok_level, line)
            _stderr_print(line)
        else:
            line = (
                f"heartbeat label='{self.label}' state=fail "
                f"elapsed={elapsed:.1f}s exc={exc_type.__name__}: {exc}"
            )
            logger.error(line)
            _stderr_print(line)

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
            line = (
                f"heartbeat label='{self.label}' tick={tick} "
                f"elapsed={elapsed:.1f}s{extra}"
            )
            logger.log(self.tick_level, line)
            _stderr_print(line)


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
        logger.info(
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
    ) -> None:
        self._model_path = model_path
        self._gpu_id = gpu_id
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dtype = dtype
        self._enforce_eager = enforce_eager

        self.device = f"cuda:{gpu_id}"

        self._llm: Any | None = None
        self._build(model_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build(self, model_path: str) -> None:
        from vllm import LLM

        logger.info(
            "vLLM build start: model=%s dtype=%s max_model_len=%d "
            "gpu_mem_util=%.2f enforce_eager=%s gpu=cuda:%d %s",
            model_path, self._dtype, self._max_model_len,
            self._gpu_memory_utilization, self._enforce_eager,
            self._gpu_id, _gpu_mem_str(self._gpu_id),
        )
        _stderr_print(
            f"[vllm_adapter] LLM(...) construct begin model={model_path} "
            f"gpu=cuda:{self._gpu_id}"
        )

        gpu_id = self._gpu_id

        with _Heartbeat(
            f"LLM(...) construct model={model_path.split('/')[-1]} "
            f"gpu=cuda:{gpu_id}",
            extra_fn=lambda: _gpu_mem_str(gpu_id),
        ):
            self._llm = LLM(
                model=model_path,
                dtype=self._dtype,
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_model_len=self._max_model_len,
                enforce_eager=self._enforce_eager,
                disable_log_stats=True,
            )

        self._model_path = model_path
        logger.info(
            "vLLM build done: ready on %s %s",
            self.device, _gpu_mem_str(self._gpu_id),
        )
        _stderr_print(f"[vllm_adapter] LLM(...) construct done on {self.device}")

    def reload(self, local_path: str) -> None:
        """Rebuild the LLM pointing at a new checkpoint.

        vLLM has no in-place weight swap, so we tear down and rebuild.
        The ~30-60s cost is amortised over CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
        windows (the validator only republishes every 10 windows).
        """
        logger.info("vLLM reload: %s → %s", self._model_path, local_path)
        _stderr_print(
            f"[vllm_adapter] reload begin {self._model_path} -> {local_path}"
        )

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
        _stderr_print(f"[vllm_adapter] reload done -> {local_path}")

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

        logger.debug(
            "vllm.generate begin n=%d prompt_len=%d max_new_tokens=%d "
            "temperature=%.2f top_p=%.2f top_k=%d %s",
            n, len(prompt_tokens), max_new_tokens,
            temperature, top_p, vllm_top_k, _gpu_mem_str(self._gpu_id),
        )
        _stderr_print(
            f"[vllm_adapter] generate begin n={n} prompt_len={len(prompt_tokens)} "
            f"max_new={max_new_tokens}"
        )

        t0 = time.monotonic()
        gpu_id = self._gpu_id
        try:
            with _Heartbeat(
                f"vllm.generate n={n} prompt_len={len(prompt_tokens)}",
                fresh_interval_s=_GENERATE_HEARTBEAT_FRESH_INTERVAL_S,
                settled_interval_s=_GENERATE_HEARTBEAT_SETTLED_INTERVAL_S,
                fresh_duration_s=_GENERATE_HEARTBEAT_FRESH_DURATION_S,
                extra_fn=lambda: _gpu_mem_str(gpu_id),
                tick_level=logging.DEBUG,
                done_ok_level=logging.DEBUG,
            ):
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

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        request_output = outputs[0]
        gen_lens = [len(c.token_ids) for c in request_output.outputs]
        finish_reasons = [c.finish_reason for c in request_output.outputs]
        logger.info(
            "vllm.generate done n=%d elapsed_ms=%.0f gen_lens=%s finish=%s",
            n, elapsed_ms, gen_lens, finish_reasons,
        )
        _stderr_print(
            f"[vllm_adapter] generate done n={n} elapsed_ms={elapsed_ms:.0f}"
        )

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
