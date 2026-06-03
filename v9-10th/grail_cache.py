"""Precompute-and-replay split of the GRAIL sketch commitment.

The window-open critical path must be as short as possible (fire in the first
drand round to win the BATCH_FILLED race), and pregeneration must be able to
prepare everything heavy *before* the per-window randomness is revealed.

The key observation, read straight out of
``GRAILVerifier.create_commitments_batch``:

    sketch[pos] = ( buckets[pos] · r_vec ) mod PRIME_Q

where

  * ``buckets[pos]`` — the signed top-K log-magnitude bucket vector of the
    model's last hidden state at token ``pos``. A pure function of
    ``(model weights, tokens)``. **Independent of the window randomness.**
  * ``r_vec`` — ``GRAILVerifier.generate_r_vec(state.randomness)``. The *only*
    randomness-dependent term, and it enters as a trivial int dot product.

So a miner can run the (expensive) HF forward once per checkpoint, cache the
``buckets`` tensor per rollout, and at window-open recompute the commitments
with the freshly-revealed ``r_vec`` in microseconds.

``compute_buckets`` + ``project_buckets`` reproduce
``create_commitments_batch`` **bit-for-bit** — they are literally its two
halves split at the bucketing line. ``assert_bit_identical`` proves it against
the canonical implementation; see ``tests/unit/test_grail_cache.py``.

This module imports the bucketing kernel from ``grail_verifier`` rather than
re-implementing it, so it can never drift from the protocol definition.
"""

from __future__ import annotations

import torch

from reliquary.constants import PRIME_Q, PROOF_NUM_BUCKETS, PROOF_TOPK
from reliquary.protocol.grail_verifier import log_magnitude_bucket_vectorized


def compute_buckets(
    h_layer: torch.Tensor,
    topk: int = PROOF_TOPK,
    num_buckets: int = PROOF_NUM_BUCKETS,
) -> torch.Tensor:
    """Return the per-position signed log-magnitude bucket matrix.

    This is the randomness-independent prefix of
    ``GRAILVerifier.create_commitments_batch``. The result is cached during
    pregeneration; ``project_buckets`` consumes it at window-open.

    Args:
        h_layer: ``[seq_len, hidden_dim]`` last-hidden-state tensor, exactly as
            handed to ``create_commitments_batch`` (bf16 on the proof GPU in the
            production path — pass it unchanged for bit-identity).
        topk / num_buckets: protocol constants; override only in tests.

    Returns:
        ``[seq_len, topk]`` ``int8`` tensor on ``h_layer.device``. Bucket values
        live in ``[-(num_buckets - 1), num_buckets - 1]`` (i.e. [-7, 7]), so
        int8 storage is lossless and ~8x smaller than the int64 the canonical
        path materialises — important when caching thousands of rollouts.
    """
    abs_h = h_layer.abs()
    _, topk_indices = torch.topk(abs_h, k=topk, dim=1)
    del abs_h
    # Sort the selected indices so the bucket ordering is canonical and
    # independent of topk's internal ordering — matches create_commitments_batch.
    topk_indices, _ = torch.sort(topk_indices, dim=1)
    signed_values = torch.gather(h_layer, dim=1, index=topk_indices)
    buckets = log_magnitude_bucket_vectorized(signed_values, num_buckets)  # int64
    return buckets.to(torch.int8)


def project_buckets(buckets: torch.Tensor, r_vec: torch.Tensor) -> list[dict]:
    """Recompute the GRAIL commitments from cached buckets + the window r_vec.

    Bit-identical to ``create_commitments_batch``'s suffix. Because every
    operand is a small integer (``|bucket| <= 7``, ``|r_vec| <= 127``, ``topk``
    terms → ``|sketch| <= 7 * 127 * topk`` ≪ 2**24), the float32 matmul is
    exact-integer on both CPU and GPU, so the result is device-independent —
    the projection can run wherever the cached buckets live (CPU is fine and
    frees proof-GPU memory for the pregeneration forward passes).

    Args:
        buckets: ``[seq_len, topk]`` tensor from ``compute_buckets`` (any int
            dtype; int8 as cached).
        r_vec: ``[topk]`` int tensor from
            ``GRAILVerifier.generate_r_vec(state.randomness)``.

    Returns:
        ``[{"sketch": int}, ...]`` of length ``seq_len`` — the ``commitments``
        list shipped inside the rollout commit dict.
    """
    buckets_f = buckets.to(torch.float32)
    r_vec_f = r_vec.to(torch.float32).to(buckets_f.device)
    sketches = (buckets_f @ r_vec_f).to(torch.int64)
    sketch_vals = [s % PRIME_Q for s in sketches.tolist()]
    return [{"sketch": v} for v in sketch_vals]


def assert_bit_identical(verifier, h_layer: torch.Tensor, r_vec: torch.Tensor) -> None:
    """Raise ``AssertionError`` unless cache→project equals the canonical path.

    A cheap runtime guard the pregeneration daemon can call once at startup (or
    a test can call on random tensors) to prove the cached-bucket replay matches
    ``GRAILVerifier.create_commitments_batch`` exactly for the live model's
    hidden-state dtype/device.
    """
    canonical = verifier.create_commitments_batch(h_layer, r_vec)
    replayed = project_buckets(
        compute_buckets(h_layer, verifier.topk, verifier.num_buckets), r_vec
    )
    if canonical != replayed:
        # Find the first divergent position for a useful message.
        for i, (a, b) in enumerate(zip(canonical, replayed)):
            if a != b:
                raise AssertionError(
                    f"grail_cache replay diverged at position {i}: "
                    f"canonical={a} replayed={b}"
                )
        raise AssertionError(
            f"grail_cache replay length mismatch: "
            f"canonical={len(canonical)} replayed={len(replayed)}"
        )
