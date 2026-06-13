"""Reliquary optimized mining toolkit.

A separate, self-contained mining subsystem that layers three performance
wins on top of the reference ``reliquary.miner`` engine:

  1. **vLLM generation** — batched, paged-attention sampling instead of
     ``transformers.generate`` (``mining.common.vllm_generator``).
  2. **Pregeneration** — the expensive, window-independent work (sampling +
     the HF GRAIL forward pass + activation bucketing) is computed ahead of
     time, keyed to the current checkpoint, so a window can be answered in
     milliseconds. Only the per-window randomness projection + signing +
     drand-round happen on the hot path (``mining.common.pregen`` and
     ``mining.common.grail_proof``).
  3. **Frontier prediction** — for a *well-trained* policy most prompts are
     8/8 or 0/8 (σ ≈ 0 → ``OUT_OF_ZONE``). Per environment we predict which
     prompts sit in the trainable band (σ ≥ ``SIGMA_MIN``) *before* spending a
     slot. For OpenCode this is a local shadow grader built from the public
     ``nvidia/OpenCodeInstruct`` unit tests (``mining.opencode``).

Layout (one directory per environment so new envs slot in cleanly):

    mining/
      common/     env-agnostic engine: vLLM, GRAIL pregen, pool, fire, state
      opencode/   OpenCodeInstruct miner (this directory)
      openmath/   OpenMathInstruct miner (added later — sibling of opencode/)
      scripts/    offline prep (build the local frontier oracle)
      tests/      fast, dependency-light correctness tests

Everything reuses ``reliquary.protocol`` / ``reliquary.constants`` /
``reliquary.miner.submitter`` so the wire format stays byte-for-byte
compatible with the live validator. Heavy deps (torch, vllm, transformers,
datasets) are imported lazily so this package is import-safe without a GPU.
"""

__all__ = ["common", "opencode"]
