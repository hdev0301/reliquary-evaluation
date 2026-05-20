"""Per-prompt σ history for GRPO miner prompt selection.

OpenMathInstruct prompts have very uneven σ outcomes against Qwen3-4B:
most augmented_math prompts produce 8/8 correct (σ → 0, OUT_OF_ZONE)
but a small minority land in the σ ≥ SIGMA_MIN zone the validator accepts.
Uniform-random sampling over ``promising_indices()`` wastes most attempts
on the easy majority.

This cache tracks per-prompt-idx σ outcomes across all generated groups
(including pre-gen ones that get σ-filtered locally) so the next sampling
biases toward prompts that previously landed in zone. Persisted as
append-only JSONL — robust to torn writes, recoverable across miner
restarts.

Schema: each line is ``{"idx": int, "sigma": float, "in_zone": 0|1, "ts": float}``.
"""

from __future__ import annotations

import json
import logging
import os
import time

from reliquary.constants import SIGMA_MIN

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = os.path.expanduser("~/.cache/reliquary/prompt_sigma.jsonl")


class PromptSigmaCache:
    """Per-prompt-idx σ history with biased-sampling support.

    All state is kept in-memory (one dict entry per known idx). Writes
    are append-only JSONL — torn writes corrupt only one line and load
    skips malformed records, so the cache survives miner crashes.

    Sampling: with probability ``hot_bias`` draw from prompts that have
    landed in σ-zone at least once before. Otherwise sample uniformly
    over the pool, rejecting indices that have been seen ``ban_after_n_zero``
    times with zero in-zone hits.
    """

    # σ for a binary-reward group on 8 samples can only take discrete
    # values: 0 (sum 0/8 or 8/8), 0.331 (1/8 or 7/8), 0.433 (2/8 or 6/8),
    # 0.484 (3/8 or 5/8), 0.5 (4/8). SIGMA_MIN = 0.43, so 1/8 and 7/8
    # prompts are one stochastic rollout-flip away from being in-zone.
    # We track them as "near" and weight sampling toward them after hot.
    _NEAR_ZONE_MIN: ClassVar = 0.20  # rounded slightly below the 0.331 σ
                                      # for 1/8 to absorb floating-point
                                      # comparison noise.

    def __init__(
        self,
        path: str = DEFAULT_CACHE_PATH,
        *,
        hot_bias: float = 0.4,
        near_bias: float = 0.3,
        ban_after_n_zero: int = 2,
    ) -> None:
        self.path = path
        self.hot_bias = hot_bias
        self.near_bias = near_bias
        self.ban_after_n_zero = ban_after_n_zero
        # idx -> {"in_zone": int, "total": int, "extreme": int, "near": int}
        # "extreme" counts records where rewards_sum was 0 or 8 — the
        # σ=0 modes that correspond to "too hard" / "too easy" prompts
        # and should be banned immediately. Plain σ=0 records WITHOUT
        # a rewards_sum tag (e.g. length-drops) still use ban_after_n_zero.
        # "near" counts records where σ ∈ [NEAR_ZONE_MIN, SIGMA_MIN) —
        # one stochastic flip from σ-zone. We bias sampling toward these.
        self._history: dict[int, dict[str, int]] = {}
        # Hot (σ ≥ SIGMA_MIN) and Near (σ near-but-under SIGMA_MIN). Both
        # are tracked as set + list — set for fast updates, list for
        # O(1) random.choice when sampling.
        self._hot: set[int] = set()
        self._hot_list: list[int] = []
        self._hot_list_stale = True
        self._near: set[int] = set()
        self._near_list: list[int] = []
        self._near_list_stale = True
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            logger.info("prompt cache: no file at %s — starting fresh", self.path)
            return
        n_records = 0
        try:
            with open(self.path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    idx = rec.get("idx")
                    if not isinstance(idx, int):
                        continue
                    h = self._history.setdefault(
                        idx, {"in_zone": 0, "total": 0, "extreme": 0, "near": 0},
                    )
                    h["in_zone"] += int(rec.get("in_zone", 0))
                    h["total"] += 1
                    rs = rec.get("rewards_sum")
                    if rs is not None and (rs == 0 or rs == 8):
                        h["extreme"] += 1
                    sigma = rec.get("sigma", 0.0)
                    if self._NEAR_ZONE_MIN <= sigma < SIGMA_MIN:
                        h["near"] += 1
                    n_records += 1
        except OSError as e:
            logger.warning("prompt cache load failed: %s", e)
            return
        for idx, h in self._history.items():
            if h["in_zone"] >= 1:
                self._hot.add(idx)
            elif h.get("near", 0) >= 1 and not self.is_banned(idx):
                self._near.add(idx)
        self._hot_list_stale = True
        self._near_list_stale = True
        logger.info(
            "prompt cache loaded: %d records, %d distinct prompts, "
            "%d hot, %d near, %d banned",
            n_records, len(self._history), len(self._hot), len(self._near),
            sum(1 for i in self._history if self.is_banned(i)),
        )

    def record(self, idx: int, sigma: float, rewards_sum: int | None = None) -> None:
        """Append one σ outcome for ``idx`` and update in-memory state.

        When ``rewards_sum`` is supplied and equals 0 or 8 (all-wrong /
        all-correct), the entry is also counted as ``extreme`` — and a
        single extreme hit is enough to ban the prompt from future
        sampling. Pass ``None`` for length-drops where σ=0 doesn't carry
        the same "obviously out of zone" signal.
        """
        in_zone = 1 if sigma >= SIGMA_MIN else 0
        is_near = self._NEAR_ZONE_MIN <= sigma < SIGMA_MIN
        rec: dict = {
            "idx": int(idx),
            "sigma": round(float(sigma), 4),
            "in_zone": in_zone,
            "ts": round(time.time(), 1),
        }
        is_extreme = rewards_sum is not None and (rewards_sum == 0 or rewards_sum == 8)
        if rewards_sum is not None:
            rec["rewards_sum"] = int(rewards_sum)
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            logger.debug("prompt cache write failed for idx=%d: %s", idx, e)

        h = self._history.setdefault(
            idx, {"in_zone": 0, "total": 0, "extreme": 0, "near": 0},
        )
        h["in_zone"] += in_zone
        h["total"] += 1
        if is_extreme:
            h["extreme"] += 1
        if is_near:
            h["near"] += 1
        if in_zone and idx not in self._hot:
            self._hot.add(idx)
            self._hot_list_stale = True
            # Promote from near to hot if previously tracked there.
            if idx in self._near:
                self._near.discard(idx)
                self._near_list_stale = True
        elif is_near and idx not in self._hot and idx not in self._near:
            if not self.is_banned(idx):
                self._near.add(idx)
                self._near_list_stale = True
        # If the new record bans the idx, evict from near/hot.
        if self.is_banned(idx):
            if idx in self._near:
                self._near.discard(idx)
                self._near_list_stale = True
            if idx in self._hot:
                self._hot.discard(idx)
                self._hot_list_stale = True

    def is_banned(self, idx: int) -> bool:
        """True if ``idx`` should be excluded from future sampling.

        Three bans:
        - one ``extreme`` hit (rewards_sum 0 or 8) — too hard / too easy
        - ``ban_after_n_zero`` non-extreme σ=0 hits with no near or hot
          signal — repeated drops with no σ-zone potential observed
        - hot prompts are never banned (the in_zone hit dominates)
        """
        h = self._history.get(idx)
        if h is None:
            return False
        if h.get("extreme", 0) >= 1:
            return True
        # Spare prompts that have shown ANY signal of σ-zone potential —
        # the in_zone history and the near-zone history (σ ∈ [0.20, 0.43))
        # are both reasons to keep re-sampling them. Stochastic rollouts
        # mean a "near" prompt may flip into the σ-zone next attempt.
        if h["in_zone"] >= 1 or h.get("near", 0) >= 1:
            return False
        return h["total"] >= self.ban_after_n_zero

    def pick(
        self,
        pool: list[int],
        cooldown: set[int],
        rng,
        *,
        max_attempts: int = 1000,
    ) -> int:
        """Pick a prompt idx from ``pool``, biased toward known σ-hitters.

        Three tiers:
        - hot (σ ≥ SIGMA_MIN): previously landed in zone. Highest priority.
        - near (σ ∈ [0.20, SIGMA_MIN)): one stochastic flip from zone.
        - cold (unseen, not banned): random pool fallback.

        Hot/near picks may collide with cooldown — we retry a few times
        before falling through to the next tier.

        Raises ``RuntimeError`` if no eligible idx can be found.
        """
        if self._hot_list_stale:
            self._hot_list = list(self._hot)
            self._hot_list_stale = False
        if self._near_list_stale:
            self._near_list = list(self._near)
            self._near_list_stale = False

        u = rng.random()

        # Hot tier
        if u < self.hot_bias and self._hot_list:
            for _ in range(20):
                idx = self._hot_list[rng.randrange(len(self._hot_list))]
                if idx not in cooldown:
                    return idx

        # Near tier
        if u < self.hot_bias + self.near_bias and self._near_list:
            for _ in range(20):
                idx = self._near_list[rng.randrange(len(self._near_list))]
                if idx not in cooldown and not self.is_banned(idx):
                    return idx

        # Cold path: uniform random over pool with rejection of banned +
        # cooldown indices. With <50% rejection rate this is near-O(1).
        n = len(pool)
        for _ in range(max_attempts):
            idx = pool[rng.randrange(n)]
            if idx in cooldown:
                continue
            if self.is_banned(idx):
                continue
            return idx

        # Rare full scan when pool is mostly banned/cooldown.
        eligible = [i for i in pool if i not in cooldown and not self.is_banned(i)]
        if not eligible:
            raise RuntimeError("no eligible prompt — pool exhausted")
        return rng.choice(eligible)

    def stats(self) -> dict:
        return {
            "distinct_prompts": len(self._history),
            "hot": len(self._hot),
            "near": len(self._near),
            "banned": sum(1 for idx in self._history if self.is_banned(idx)),
        }
