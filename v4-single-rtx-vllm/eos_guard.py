"""EOS-confidence sampling guard.

The validator's ``verify_termination`` accepts a rollout only if its
last token is in the EOS set AND the raw ``softmax(logits)`` probability
of EOS at the position before it is ≥ ``MIN_EOS_PROBABILITY`` (= 0.01,
constants.py). When a miner samples EOS at a position where the model's
raw EOS probability is below that floor (which happens often with
``temperature=0.9`` + ``top_p=1.0`` + ``top_k=0`` — pure stochastic
sampling), the rollout looks fine to the miner but the validator's
forward pass on the same prefix computes ``p_stop < 0.01`` and rejects
with ``bad_termination``.

This guard sits in the sampling pipeline and masks EOS logits to
``-inf`` at any position where raw ``p(EOS)`` is below ``threshold``.
That guarantees:

  * EOS is NEVER sampled at positions where the validator would
    subsequently reject the rollout for low p_stop.
  * When EOS *is* sampled, the validator's recomputed p_stop sits at
    ``threshold`` or higher — well above the 0.01 floor for a safety
    margin against bf16 drift between miner and validator HF runs.

Side effect: rollouts that would have terminated at marginal-confidence
EOS now continue generating until they find a high-confidence EOS or
hit ``max_new_tokens``. Cap-truncated rollouts still trip the validator's
truncation counter, so this guard doesn't make us submit MORE rollouts —
it just shifts the failure mode from ``bad_termination(low_p_stop)`` to
``bad_termination(cap_truncation)`` on the small fraction of prompts
where the model can't find a confident EOS within budget. Empirically
the net is a large drop in bad_termination since the low-p case
dominates the failure distribution on Qwen3-4B at T=0.9.

Compatible with two callers:

  * HF transformers: ``LogitsProcessor`` interface
    (``__call__(input_ids, scores)``)
  * vLLM 0.20.x:     ``LogitsProcessor`` callable
    (``__call__(prompt_token_ids, output_token_ids, logits)``)
"""
from __future__ import annotations

import torch


class EosConfidenceGuard:
    """Shared core: applies the EOS mask to a [..., vocab] logits tensor."""

    def __init__(self, eos_ids: set[int] | list[int], threshold: float = 0.015):
        ids = sorted(int(e) for e in eos_ids if e is not None)
        if not ids:
            raise ValueError("EosConfidenceGuard requires a non-empty eos_ids set")
        self.eos_ids = ids
        self.threshold = float(threshold)
        # Cache the eos-index tensor per device — guard is called every
        # decode step so re-allocating on each call adds latency.
        self._eos_tensor_cache: dict[str, torch.Tensor] = {}

    def _eos_tensor(self, device: torch.device | str) -> torch.Tensor:
        key = str(device)
        cached = self._eos_tensor_cache.get(key)
        if cached is None:
            cached = torch.tensor(
                self.eos_ids, device=device, dtype=torch.long,
            )
            self._eos_tensor_cache[key] = cached
        return cached

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask EOS logits to ``-inf`` where raw p(EOS) < threshold.

        Accepts either ``[vocab]`` (single-row, vLLM v1 per-request
        path) or ``[batch, vocab]`` (HF batched-generate path). Returns
        the same shape. The returned tensor IS the input tensor with
        in-place modifications on the masked positions (cheaper than
        cloning for tight decode loops).
        """
        # Raw softmax over the un-temperature-scaled logits — matches
        # the validator's _gpu_p_stop exactly.
        probs = torch.softmax(logits.float(), dim=-1)
        eos_idx = self._eos_tensor(logits.device)
        # Sum over all EOS variants (Qwen3-Instruct has two: 151643
        # <|endoftext|> and 151645 <|im_end|>).
        eos_prob = probs.index_select(-1, eos_idx).sum(dim=-1)
        mask = eos_prob < self.threshold
        if not bool(mask.any().item()):
            return logits
        neg_inf = torch.tensor(
            float("-inf"), device=logits.device, dtype=logits.dtype,
        )
        if logits.dim() == 1:
            # Single-row path (vLLM v1). mask is a 0-d tensor.
            if bool(mask):
                logits[eos_idx] = neg_inf
        else:
            # Batched path (HF generate). Broadcast mask across eos cols.
            # mask shape: [batch], expand to [batch, len(eos_idx)]
            row_idx = mask.nonzero(as_tuple=True)[0]
            if row_idx.numel():
                logits.index_put_(
                    (row_idx.unsqueeze(1), eos_idx.unsqueeze(0)),
                    neg_inf.expand(row_idx.numel(), eos_idx.numel()),
                )
        return logits


class HFEosGuardProcessor:
    """HF transformers ``LogitsProcessor`` adapter for ``EosConfidenceGuard``.

    HF's interface is ``__call__(input_ids, scores) -> scores`` where
    ``scores`` is ``[batch, vocab]``. We mutate scores in place and
    return it.
    """

    def __init__(self, guard: EosConfidenceGuard) -> None:
        self.guard = guard

    def __call__(self, input_ids, scores):  # type: ignore[override]
        return self.guard.apply(scores)


class VllmEosGuardProcessor:
    """vLLM 0.20.x ``logits_processors`` adapter for ``EosConfidenceGuard``.

    vLLM's interface is
    ``__call__(prompt_token_ids, output_token_ids, logits) -> logits``
    where ``logits`` is a single-row ``[vocab]`` tensor in the v1 engine.
    """

    def __init__(self, guard: EosConfidenceGuard) -> None:
        self.guard = guard

    def __call__(self, prompt_token_ids, output_token_ids, logits):  # noqa: D401
        return self.guard.apply(logits)


def _import_vllm_lp_base():
    """Defer the vLLM import until the class is actually defined.

    Lets this module stay importable on machines without vLLM (the
    HF path doesn't need the v1 guard) and keeps the import error
    explicit at class-definition time rather than at module load.
    """
    from vllm.v1.sample.logits_processor import LogitsProcessor
    return LogitsProcessor


try:
    _VllmLogitsProcessor = _import_vllm_lp_base()
except Exception:  # pragma: no cover — vLLM not installed
    _VllmLogitsProcessor = object


class VllmV1EosGuard(_VllmLogitsProcessor):
    """vLLM v1 engine-level ``LogitsProcessor`` that masks EOS to -inf
    where raw ``p(EOS)`` is below ``threshold`` — same thresholding
    semantics as the HF guard, applied per decode step across the
    whole active batch.

    Inherits from ``vllm.v1.sample.logits_processor.LogitsProcessor``:
    vLLM's ``_load_logitsprocs_by_fqcns`` does an ``issubclass`` check
    against that ABC at engine init and rejects duck-typed classes
    with ``ValueError("X is not a subclass of LogitsProcessor")``.

    Registration: ``LLM(..., logits_processors=[VllmV1EosGuard])``.
    Reads its config from env vars because vLLM constructs the
    processor in the EngineCore subprocess via ``cls(vllm_config,
    device, is_pin_memory)`` — no extra args are routable.

    Env vars:
      * ``RELIQUARY_VLLM_EOS_IDS``    comma-separated EOS token ids
      * ``RELIQUARY_VLLM_EOS_THRESHOLD`` float, default 0.015
    """

    def __init__(self, vllm_config, device, is_pin_memory):
        import os as _os
        import sys as _sys
        raw_ids = _os.environ.get("RELIQUARY_VLLM_EOS_IDS", "").strip()
        ids = [int(x) for x in raw_ids.split(",") if x.strip()] if raw_ids else []
        if not ids:
            raise ValueError(
                "VllmV1EosGuard requires RELIQUARY_VLLM_EOS_IDS env var "
                "(comma-separated EOS token ids); none set"
            )
        try:
            self._threshold = float(
                _os.environ.get("RELIQUARY_VLLM_EOS_THRESHOLD", "0.015")
            )
        except ValueError:
            self._threshold = 0.015
        self._device = device
        self._eos_tensor = torch.tensor(ids, device=device, dtype=torch.long)
        self._call_count = 0
        self._mask_count = 0
        # Side-effect proof that vLLM is constructing us in the EngineCore
        # subprocess (stdout/stderr from the subprocess goes to the same
        # log file as the parent). Without this we cannot distinguish
        # "guard never instantiated" from "guard instantiated but apply()
        # never reached".
        print(
            f"[VllmV1EosGuard] __init__ ids={ids} threshold={self._threshold} "
            f"device={device}",
            file=_sys.stderr, flush=True,
        )

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update) -> None:
        # No per-request state — mask applies uniformly to every active
        # row. Engine never calls ``apply`` on an empty batch.
        return None

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [batch, vocab] (or [vocab] in edge cases). Raw
        # softmax matches validator _gpu_p_stop. Sum EOS-id columns
        # and mask those columns to -inf wherever the sum falls below
        # threshold — sampling is then forced to pick a non-EOS token
        # at sub-threshold positions, deferring termination to a
        # position the validator will accept.
        self._call_count += 1
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        probs = torch.softmax(logits.float(), dim=-1)
        eos_prob = probs.index_select(-1, self._eos_tensor).sum(dim=-1)
        mask = eos_prob < self._threshold
        masked_this_call = 0
        if bool(mask.any().item()):
            row_idx = mask.nonzero(as_tuple=True)[0]
            masked_this_call = int(row_idx.numel())
            self._mask_count += masked_this_call
            neg_inf = torch.tensor(
                float("-inf"), device=logits.device, dtype=logits.dtype,
            )
            logits.index_put_(
                (row_idx.unsqueeze(1), self._eos_tensor.unsqueeze(0)),
                neg_inf.expand(row_idx.numel(), self._eos_tensor.numel()),
            )
        # Heartbeat every 5000 calls — enough to know apply() is firing
        # without flooding the log. Single-row applies don't get logged
        # to avoid the "one rollout running solo to cap" log storm we
        # saw with smaller intervals.
        if (
            self._call_count % 5000 == 0
            and logits.shape[0] > 1
        ):
            import sys as _sys
            mask_rate = self._mask_count / max(self._call_count, 1)
            print(
                f"[VllmV1EosGuard] apply #{self._call_count} "
                f"batch={tuple(logits.shape)} "
                f"avg_mask_rate={mask_rate:.2f}",
                file=_sys.stderr, flush=True,
            )
        return logits.squeeze(0) if squeeze_back else logits
