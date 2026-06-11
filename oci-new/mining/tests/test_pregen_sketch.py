"""The pregen split must be BIT-IDENTICAL to the validator's on-the-spot path.

If ``commitments_from_buckets(buckets_from_hidden(h), r_vec)`` ever diverged
from ``GRAILVerifier.create_commitments_batch(h, r_vec)``, a pregenerated proof
would fail GRAIL even on the correct model. This test pins the equivalence.

Dependency-light: needs only torch + numpy + the reliquary package (no GPU, no
model download). Run with:  python -m mining.tests.test_pregen_sketch
"""

from __future__ import annotations


def test_pregen_split_matches_verifier() -> None:
    import torch

    from reliquary.constants import PROOF_TOPK
    from reliquary.protocol.grail_verifier import GRAILVerifier

    from mining.common.grail_proof import buckets_from_hidden, commitments_from_buckets

    hidden_dim = 128
    verifier = GRAILVerifier(hidden_dim=hidden_dim)

    for seed in range(5):
        torch.manual_seed(seed)
        h = torch.randn(37, hidden_dim)  # [seq_len, hidden_dim]
        randomness = ("%064x" % (0xABCDEF0123456789 * (seed + 1)))[:64]
        r_vec = verifier.generate_r_vec(randomness)

        reference = verifier.create_commitments_batch(h, r_vec)

        buckets = buckets_from_hidden(h, PROOF_TOPK)          # cached, randomness-free
        assert buckets.dtype == torch.int8
        # int8 round-trip must be lossless for the bucket range.
        assert torch.equal(buckets.to(torch.int64), buckets_from_hidden(h, PROOF_TOPK).to(torch.int64))
        mine = commitments_from_buckets(buckets, r_vec)       # late projection

        assert mine == reference, f"sketch divergence at seed={seed}"

    print("OK: pregen split is bit-identical to create_commitments_batch (5 seeds)")


if __name__ == "__main__":
    test_pregen_split_matches_verifier()
