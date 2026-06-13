"""GRAIL proof construction, split for pregeneration.

The validator's GRAIL check (``reliquary.protocol.grail_verifier``) compares,
per challenged token position, a *sketch* = ``buckets · r_vec`` where:

  * ``buckets``  — the log-magnitude buckets of the top-k activations of the
    proof model's hidden state at that position. **Depends only on (model,
    tokens)** — NOT on the window.
  * ``r_vec``    — a tiny ±127 coefficient vector derived from the window's
    ``randomness``. **The only window-dependent input.**

So the whole expensive path — the HF forward pass that produces hidden states
and logits, the top-k + bucketing, and the fp32 log-prob extraction — can run
the moment a rollout is sampled, long before the window that will carry it
opens. When the window finally opens and publishes ``randomness`` we only need
the cheap ``buckets @ r_vec`` matvec plus the ed25519 signature.

``mining/tests/test_pregen_sketch.py`` asserts the split is **bit-identical**
to calling ``GRAILVerifier.create_commitments_batch`` on the spot, so a pregen
proof is indistinguishable on the wire from a freshly-built one.

This mirrors ``reliquary.miner.engine.MiningEngine._build_grail_commit`` field
for field; the only change is *when* each part runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Pure split of GRAILVerifier.create_commitments_batch into a
# randomness-free half (buckets) and a randomness-dependent half (sketch).
# mining/tests/test_pregen_sketch.py asserts these compose to the original.
# ----------------------------------------------------------------------
def buckets_from_hidden(hidden_states, topk: int):
    """Randomness-free half: top-k activation log-magnitude buckets.

    ``hidden_states`` is [seq_len, hidden_dim]. Returns an int8 [seq_len, topk]
    tensor (buckets are in [-7, 7]). Mirrors the first half of
    ``GRAILVerifier.create_commitments_batch`` exactly.
    """
    import torch

    from reliquary.protocol.grail_verifier import log_magnitude_bucket_vectorized

    abs_h = hidden_states.abs()
    _, topk_indices = torch.topk(abs_h, k=topk, dim=1)
    topk_indices, _ = torch.sort(topk_indices, dim=1)
    signed_values = torch.gather(hidden_states, dim=1, index=topk_indices)
    return log_magnitude_bucket_vectorized(signed_values).to(torch.int8)


def commitments_from_buckets(buckets, r_vec) -> list[dict]:
    """Randomness-dependent half: project cached buckets through r_vec.

    Produces the exact ``[{"sketch": ...}]`` list the validator recomputes.
    """
    import torch

    from reliquary.constants import PRIME_Q

    buckets_f = buckets.to(torch.float32)
    r_vec_f = r_vec.to(torch.float32).to(buckets_f.device)
    sketches = (buckets_f @ r_vec_f).to(torch.int64)
    return [{"sketch": s % PRIME_Q} for s in sketches.tolist()]


@dataclass
class ProofPayload:
    """Window-independent half of a GRAIL commit, cached until a window opens.

    ``buckets`` is the int8 [seq_len, topk] activation-bucket matrix; applying
    the per-window ``r_vec`` to it reproduces the exact sketch the validator
    will recompute. Tiny (≤ ~130 KB for an 8 k-token rollout) so a deep pool
    costs little RAM.
    """

    tokens: list[int]
    prompt_length: int
    completion_length: int
    token_logprobs: list[float]
    buckets: "object"          # torch.Tensor [seq_len, topk] int8 on CPU
    model_name: str
    finished_with_eos: bool     # False ⇒ length-capped (counts toward truncation guard)


class ProofBuilder:
    """Builds (and later finalizes) GRAIL proofs from an HF proof model.

    One instance wraps the HF model copy used for proofs. ``precompute`` runs
    the heavy forward pass; ``finalize`` applies a window's randomness. Both
    must use the *same* model the validator published, on the same GPU class,
    with ``flash_attention_2`` — exactly as the reference engine requires.
    """

    def __init__(self, hf_model, *, proof_gpu: int) -> None:
        from reliquary.protocol.grail_verifier import GRAILVerifier
        from reliquary.shared.hf_compat import resolve_hidden_size

        self.hf_model = hf_model
        self.proof_gpu = proof_gpu
        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    # ------------------------------------------------------------------
    # Heavy, window-independent half — run during pregen.
    # ------------------------------------------------------------------
    def precompute(self, tokens: list[int], prompt_length: int, *, finished_with_eos: bool) -> ProofPayload:
        """Run the HF forward pass and cache everything except the r_vec matvec.

        Reproduces ``engine._build_grail_commit`` up to (but not including)
        ``generate_r_vec`` / ``create_commitments_batch``. Returns a small CPU
        payload safe to keep in a pool across many windows (same checkpoint).
        """
        import torch

        from reliquary.constants import LAYER_INDEX, PROOF_TOPK
        from reliquary.shared.forward import forward_single_layer

        proof_input = torch.tensor([tokens], device=f"cuda:{self.proof_gpu}")
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )
        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]

        # --- randomness-free half of create_commitments_batch (cache on CPU) ---
        # Buckets are in [-7, 7]; int8 is lossless and the int64 matmul at fire
        # time upcasts identically.
        buckets_cpu = buckets_from_hidden(hidden_states, PROOF_TOPK).cpu()

        # --- fp32 token log-probs, exactly as the reference miner computes ---
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs: list[float] = [
            log_probs[i - 1, tokens[i]].item()
            for i in range(prompt_length, len(tokens))
        ]
        del logits, log_probs, hidden_states
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return ProofPayload(
            tokens=list(tokens),
            prompt_length=prompt_length,
            completion_length=len(tokens) - prompt_length,
            token_logprobs=token_logprobs,
            buckets=buckets_cpu,
            model_name=getattr(self.hf_model, "name_or_path", "unknown"),
            finished_with_eos=finished_with_eos,
        )

    # ------------------------------------------------------------------
    # Cheap, window-dependent half — run on the hot path when a window opens.
    # ------------------------------------------------------------------
    def finalize_commit(self, payload: ProofPayload, randomness: str, wallet) -> dict:
        """Apply the window ``randomness`` and sign — the only fire-time work.

        Returns a ``commit`` dict byte-identical in shape to
        ``engine._build_grail_commit``. The ``buckets @ r_vec`` matvec is a few
        microseconds; signing is the dominant cost and is still sub-millisecond.
        """
        from reliquary.constants import GRAIL_PROOF_VERSION, LAYER_INDEX
        from reliquary.protocol.signatures import sign_commit_binding

        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = commitments_from_buckets(payload.buckets, r_vec)

        signature = sign_commit_binding(
            payload.tokens, randomness, payload.model_name, LAYER_INDEX,
            commitments, wallet,
        )

        return {
            "tokens": payload.tokens,
            "commitments": commitments,
            "proof_version": GRAIL_PROOF_VERSION,
            "model": {"name": payload.model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": {
                "prompt_length": payload.prompt_length,
                "completion_length": payload.completion_length,
                "success": True,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": payload.token_logprobs,
            },
        }
