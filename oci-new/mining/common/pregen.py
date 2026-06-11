"""Pregeneration pool: keep a depth of ready, in-zone, fully-proven groups.

The pool runs the heavy work (vLLM sampling → local σ screen → HF GRAIL forward
pass → activation bucketing) *ahead of* the windows that will carry it, keyed
to the live ``checkpoint_n``. When a window opens, the miner pops the best
ready groups and fires them in milliseconds (``mining.common.fire``).

The pool is environment-agnostic. The env supplies an ``EnvProducer`` with:

    next_candidates(n, exclude) -> list[Candidate]      # which prompts to try
    screen(candidate, rollouts) -> ScreenResult         # predict σ locally

so OpenCode (local unit-test shadow grader) and a future OpenMath producer
(public-label reward) plug into the same machinery.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    prompt_idx: int
    prompt_tokens: list[int]
    context: object = None       # opaque, passed back to screen() (e.g. cases)


@dataclass
class ScreenResult:
    in_zone: bool
    sigma: float
    k_correct: float             # expected #correct out of M (proxy)
    reward_vec: list[float] = field(default_factory=list)
    score: float = 0.0           # higher = fire sooner (distance from σ floor, low crowding)
    reject_reason: str = ""      # why an in-σ-band group was rejected (truncated/eos/shape)
    selected_indices: list[int] = field(default_factory=list)  # the 8 rollouts to submit
    pass_idx: list[int] = field(default_factory=list)   # terminated passes, best-first
    fail_idx: list[int] = field(default_factory=list)   # terminated fails, worst-first
    all_rewards: list[float] = field(default_factory=list)


@dataclass(order=True)
class ReadyGroup:
    # Ordered by negative score so heap/sort puts the best first.
    sort_key: float
    prompt_idx: int = field(compare=False)
    env_name: str = field(compare=False)
    payloads: list = field(compare=False, default_factory=list)   # list[ProofPayload]
    checkpoint_n: int = field(compare=False, default=0)
    sigma: float = field(compare=False, default=0.0)
    score: float = field(compare=False, default=0.0)


class PregenPool:
    """Thread-safe store of ready groups, refilled by a background producer."""

    def __init__(self, generator, proof_builder, config) -> None:
        self.generator = generator
        self.proof_builder = proof_builder
        self.config = config
        self._lock = threading.Lock()
        self._ready: dict[str, list[ReadyGroup]] = {}
        self._checkpoint_n: int = 0
        # Prompts already queued/fired this checkpoint, to avoid re-screening
        # the same idx (it would only get cooled after it wins anyway).
        self._claimed: dict[str, set[int]] = {}

    # ------------------------------------------------------------------
    # Checkpoint / cooldown lifecycle
    # ------------------------------------------------------------------
    def set_checkpoint(self, checkpoint_n: int) -> bool:
        """Update the pool's checkpoint. On change, flush everything (stale model).

        Returns True if the checkpoint changed (caller should reload the vLLM
        and HF models before producing again).
        """
        with self._lock:
            if checkpoint_n == self._checkpoint_n:
                return False
            logger.info(
                "checkpoint %d -> %d: flushing pregen pool (%d groups dropped)",
                self._checkpoint_n, checkpoint_n,
                sum(len(v) for v in self._ready.values()),
            )
            self._checkpoint_n = checkpoint_n
            self._ready.clear()
            self._claimed.clear()
            return True

    def drop_cooled(self, env_name: str, cooldown: set[int]) -> None:
        """Remove ready groups whose prompt entered cooldown (would reject)."""
        with self._lock:
            groups = self._ready.get(env_name)
            if not groups:
                return
            self._ready[env_name] = [g for g in groups if g.prompt_idx not in cooldown]

    # ------------------------------------------------------------------
    # Read side (fire path)
    # ------------------------------------------------------------------
    def depth(self, env_name: str) -> int:
        with self._lock:
            return len(self._ready.get(env_name, ()))

    def pop_best(self, env_name: str, *, checkpoint_n: int, exclude: set[int]) -> "ReadyGroup | None":
        """Pop the highest-scoring ready group not in ``exclude`` for this ckpt."""
        with self._lock:
            groups = self._ready.get(env_name)
            if not groups:
                return None
            groups.sort()  # best (lowest sort_key = highest score) first
            for i, g in enumerate(groups):
                if g.checkpoint_n == checkpoint_n and g.prompt_idx not in exclude:
                    return groups.pop(i)
            return None

    # ------------------------------------------------------------------
    # Write side (producer thread)
    # ------------------------------------------------------------------
    def _claimed_set(self, env_name: str) -> set[int]:
        return self._claimed.setdefault(env_name, set())

    def produce_once(self, env_name: str, producer) -> int:
        """One screen→prove cycle. BLOCKING (GPU) — call via asyncio.to_thread.

        Returns the number of in-zone groups added. Runs vLLM on a batch of
        candidate prompts, screens each for σ locally, and for the survivors
        builds the 8 GRAIL proof payloads and inserts them. Candidates are
        chosen by the env producer; already-claimed idxs this checkpoint are
        excluded so we keep finding *new* frontier prompts.
        """
        with self._lock:
            checkpoint_n = self._checkpoint_n
            if len(self._ready.get(env_name, ())) >= self.config.pool_max_depth:
                return 0
            exclude = set(self._claimed_set(env_name))

        candidates = producer.next_candidates(
            self.config.screen_batch_prompts, exclude=exclude
        )
        print(f"@@PRODUCE candidates={len(candidates)} ckpt={checkpoint_n}", flush=True)
        if not candidates:
            return 0

        import time as _t
        _g0 = _t.time()
        n_gen = max(self._m_rollouts(), getattr(self.config, "overgen_rollouts", 0))
        groups = self.generator.generate_groups(
            [c.prompt_tokens for c in candidates], n=n_gen,
            max_tokens=getattr(self.config, "gen_max_tokens", 0),
        )
        print(f"@@PRODUCE generated {len(groups)}x{n_gen} rollouts in {_t.time()-_g0:.1f}s", flush=True)

        added = 0
        with self._lock:
            for c in candidates:
                self._claimed_set(env_name).add(c.prompt_idx)

        screened = 0
        k_hist: dict[int, int] = {}
        reject_hist: dict[str, int] = {}
        sigma_max = 0.0
        in_zone_n = 0
        for cand, rollouts in zip(candidates, groups):
            try:
                screen = producer.screen(cand, rollouts)
            except Exception:
                logger.exception("screen failed for prompt %d", cand.prompt_idx)
                continue
            screened += 1
            kk = int(round(screen.k_correct))
            k_hist[kk] = k_hist.get(kk, 0) + 1
            sigma_max = max(sigma_max, screen.sigma)
            if screen.reject_reason:
                reject_hist[screen.reject_reason] = reject_hist.get(screen.reject_reason, 0) + 1
            if not screen.in_zone:
                continue
            in_zone_n += 1
            # Build the 8 proof payloads. If the producer offers a p_stop-aware
            # finalizer (OpenCode: pick rollouts that clear the BAD_TERMINATION
            # floor), use it; else precompute the σ-selected 8 directly.
            try:
                if hasattr(producer, "finalize_payloads"):
                    payloads = producer.finalize_payloads(rollouts, screen, self.proof_builder)
                    if not payloads:
                        print(f"@@PRODUCE dropped prompt {cand.prompt_idx}: no clean (p_stop) in-zone group", flush=True)
                        continue
                else:
                    sel = screen.selected_indices or list(range(len(rollouts)))
                    payloads = [
                        self.proof_builder.precompute(
                            rollouts[i].tokens, rollouts[i].prompt_length,
                            finished_with_eos=rollouts[i].finished_with_eos,
                        )
                        for i in sel
                    ]
            except Exception:
                logger.exception("proof finalize failed for prompt %d", cand.prompt_idx)
                continue
            # Optional blunt EOS gate (default off; finalize_payloads handles p_stop).
            min_eos = getattr(self.config, "min_eos_prob", 0.0)
            if min_eos > 0.0 and self._eos_prob_risk(payloads, min_eos):
                continue
            group = ReadyGroup(
                sort_key=-screen.score,
                prompt_idx=cand.prompt_idx,
                env_name=env_name,
                payloads=payloads,
                checkpoint_n=checkpoint_n,
                sigma=screen.sigma,
                score=screen.score,
            )
            with self._lock:
                # Guard against a checkpoint flip mid-cycle.
                if checkpoint_n != self._checkpoint_n:
                    break
                self._ready.setdefault(env_name, []).append(group)
                added += 1
        if screened:
            k_str = " ".join(f"k{k}={k_hist[k]}" for k in sorted(k_hist))
            r_str = " ".join(f"{r}={reject_hist[r]}" for r in sorted(reject_hist))
            print(
                f"@@PRODUCE screened={screened} in_zone={in_zone_n} added={added} "
                f"sigma_max={sigma_max:.2f} | {k_str} | band_rejects: {r_str or 'none'}",
                flush=True,
            )
            # Always log a cycle summary so the in-zone hit rate and the
            # k-distribution (why a well-trained model is mostly out-of-zone)
            # are visible for tuning.
            logger.info(
                "pregen cycle %s: screened=%d in_zone=%d added=%d depth=%d sigma_max=%.2f | %s",
                env_name, screened, in_zone_n, added, self.depth(env_name), sigma_max, k_str,
            )
        return added

    @staticmethod
    def _eos_prob_risk(payloads, min_eos: float) -> bool:
        """True if any EOS-terminated rollout's T=1.0 EOS prob is below margin."""
        for p in payloads:
            if getattr(p, "finished_with_eos", False) and p.token_logprobs:
                if math.exp(p.token_logprobs[-1]) < min_eos:
                    return True
        return False

    @staticmethod
    def _m_rollouts() -> int:
        from reliquary.constants import M_ROLLOUTS

        return M_ROLLOUTS
