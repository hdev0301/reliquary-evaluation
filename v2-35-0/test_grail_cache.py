"""The cached-bucket GRAIL replay must equal the canonical sketch bit-for-bit.

Pregeneration caches ``compute_buckets(h_layer)`` (randomness-free) and replays
``project_buckets(buckets, r_vec)`` at window-open. If that ever diverged from
``GRAILVerifier.create_commitments_batch`` the validator would reject every
submission with GRAIL_FAIL, so this is the load-bearing correctness test.
"""

from __future__ import annotations

import torch

from reliquary.constants import PROOF_NUM_BUCKETS, PROOF_TOPK
from reliquary.miner.grail_cache import (
    assert_bit_identical,
    compute_buckets,
    project_buckets,
)
from reliquary.protocol.grail_verifier import GRAILVerifier

_RANDOMNESS = "a3f1" * 16  # 64 hex chars, like a drand quicknet randomness


def _verifier(hidden_dim: int = 128) -> GRAILVerifier:
    return GRAILVerifier(hidden_dim=hidden_dim)


def test_replay_matches_canonical_float32():
    v = _verifier()
    torch.manual_seed(0)
    h = torch.randn(40, v.hidden_dim, dtype=torch.float32)
    r_vec = v.generate_r_vec(_RANDOMNESS)

    canonical = v.create_commitments_batch(h, r_vec)
    replayed = project_buckets(compute_buckets(h), r_vec)

    assert canonical == replayed


def test_replay_matches_canonical_bfloat16():
    # Production hidden states are bf16 on GPU; the cache must be exact for the
    # exact dtype the forward pass produces.
    v = _verifier()
    torch.manual_seed(1)
    h = torch.randn(64, v.hidden_dim).to(torch.bfloat16)
    r_vec = v.generate_r_vec(_RANDOMNESS)

    assert v.create_commitments_batch(h, r_vec) == project_buckets(compute_buckets(h), r_vec)


def test_buckets_are_int8_and_bounded():
    v = _verifier()
    torch.manual_seed(2)
    h = torch.randn(16, v.hidden_dim)
    buckets = compute_buckets(h)
    assert buckets.dtype == torch.int8
    assert buckets.shape == (16, PROOF_TOPK)
    assert int(buckets.abs().max()) <= PROOF_NUM_BUCKETS - 1


def test_projection_is_device_independent_for_cached_buckets():
    # int8 cache projected on CPU must equal int64 buckets projected in place.
    v = _verifier()
    torch.manual_seed(3)
    h = torch.randn(24, v.hidden_dim)
    r_vec = v.generate_r_vec(_RANDOMNESS)

    int8_buckets = compute_buckets(h)               # int8
    int64_buckets = int8_buckets.to(torch.int64)    # widen
    assert project_buckets(int8_buckets, r_vec) == project_buckets(int64_buckets, r_vec)


def test_assert_bit_identical_passes_on_random_input():
    v = _verifier(hidden_dim=96)
    torch.manual_seed(4)
    h = torch.randn(50, v.hidden_dim).to(torch.bfloat16)
    r_vec = v.generate_r_vec(_RANDOMNESS)
    assert_bit_identical(v, h, r_vec)  # raises on mismatch


def test_different_randomness_changes_sketch_but_not_buckets():
    v = _verifier()
    torch.manual_seed(5)
    h = torch.randn(32, v.hidden_dim)
    buckets = compute_buckets(h)

    r1 = v.generate_r_vec(_RANDOMNESS)
    r2 = v.generate_r_vec("beef" * 16)
    # buckets cached once, two different windows reuse them:
    c1 = project_buckets(buckets, r1)
    c2 = project_buckets(buckets, r2)
    assert c1 != c2  # window randomness genuinely changes the commitment
