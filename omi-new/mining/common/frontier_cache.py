"""Persistent frontier memory — the "pregen" of prompt SELECTION.

A well-trained policy sits at its learning frontier on only a small, slowly-
drifting set of prompts. The validator republishes the checkpoint every ~10
windows, and every reload flushes the pregen pool AND the in-memory σ-bucket
learner (``OpenCodeFrontier.reset_for_checkpoint``), forcing a cold-start
re-discovery of that rare frontier each time — the dominant cause of sustained
``in_zone=0`` right after a reload.

``FrontierCache`` removes that cold start. It persists, across reloads and
process restarts:

  * per-difficulty-bucket σ aggregates ``(sum_sigma, n)`` → used to WARM-START
    the bucket learner so high-σ bands keep positive sampling weight from cycle
    0 (decayed across a checkpoint change, since the model drifts);
  * a bounded set of concrete ``prompt_idx`` whose screened σ cleared the zone
    floor ("hot prompts") → re-tried FIRST after a reload.

It stores ONLY prompt indices and scalar σ aggregates — never model weights,
tokens, or rollouts — so it is entirely consensus/GRAIL-inert: every cached
prompt is still freshly generated, screened, and GRAIL-proven against the LIVE
checkpoint before firing. A prompt that has gone stale (now solved 8/8) simply
screens out and is never fired. Cooldown is always honored by the caller.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)


class FrontierCache:
    def __init__(self, path: str, *, hot_cap: int = 6000, warm_decay: float = 0.5) -> None:
        self.path = path
        self.hot_cap = max(0, int(hot_cap))
        self.warm_decay = float(warm_decay)
        self._lock = threading.Lock()
        self.checkpoint_n: int = -1
        self.buckets: dict[int, list] = {}   # bucket -> [sum_sigma, n]
        self.hot: dict[int, float] = {}       # prompt_idx -> best σ seen
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            if not os.path.exists(self.path):
                return
            opener = gzip.open if self.path.endswith(".gz") else open
            with opener(self.path, "rt") as f:
                d = json.load(f)
            self.checkpoint_n = int(d.get("checkpoint_n", -1))
            self.buckets = {int(k): [float(v[0]), int(v[1])] for k, v in d.get("buckets", {}).items()}
            self.hot = {int(k): float(v) for k, v in d.get("hot", {}).items()}
            logger.info(
                "frontier cache loaded: ckpt=%d buckets=%d hot=%d from %s",
                self.checkpoint_n, len(self.buckets), len(self.hot), self.path,
            )
        except Exception:
            logger.exception("frontier cache load failed; starting empty")
            self.checkpoint_n, self.buckets, self.hot = -1, {}, {}

    # ------------------------------------------------------------------
    # Warm-start the bucket learner (called from reset_for_checkpoint).
    # ------------------------------------------------------------------
    def seed_bucket_stats(self, stats, checkpoint_n: int) -> None:
        """Seed a ``_BucketStats`` from cache, decaying on a checkpoint change.

        Same checkpoint (restart) → restore exactly. Newer checkpoint → keep the
        shape but down-weight (``warm_decay``) so a drifted model can override
        stale bands quickly while still skipping the cold start.
        """
        with self._lock:
            if not self.buckets:
                return
            decay = 1.0 if checkpoint_n == self.checkpoint_n else self.warm_decay
            for b, (s, n) in self.buckets.items():
                nn = max(1, int(round(n * decay)))
                ss = s * decay
                stats.sum_sigma[b] = ss
                stats.n[b] = nn
            logger.info(
                "frontier cache: warm-started %d buckets (decay=%.2f, cache_ckpt=%d, cur_ckpt=%d)",
                len(self.buckets), decay, self.checkpoint_n, checkpoint_n,
            )

    # ------------------------------------------------------------------
    def hot_set(self) -> set:
        with self._lock:
            return set(self.hot)

    def record_hot(self, prompt_idx: int, sigma: float) -> None:
        with self._lock:
            prev = self.hot.get(prompt_idx)
            if prev is None or sigma > prev:
                self.hot[prompt_idx] = float(sigma)
                self._dirty = True
            if self.hot_cap and len(self.hot) > self.hot_cap:
                # Evict the lowest-σ entries (least promising).
                keep = sorted(self.hot.items(), key=lambda kv: -kv[1])[: self.hot_cap]
                self.hot = dict(keep)

    def drop(self, prompt_idxs) -> None:
        """Forget prompts that are now in cooldown (won → never re-fireable)."""
        if not prompt_idxs:
            return
        with self._lock:
            changed = False
            for i in prompt_idxs:
                if i in self.hot:
                    del self.hot[i]
                    changed = True
            self._dirty = self._dirty or changed

    # ------------------------------------------------------------------
    def snapshot_buckets(self, stats, checkpoint_n: int) -> None:
        with self._lock:
            self.buckets = {
                int(b): [float(stats.sum_sigma.get(b, 0.0)), int(stats.n.get(b, 0))]
                for b in stats.n
            }
            self.checkpoint_n = int(checkpoint_n)
            self._dirty = True

    def flush(self, force: bool = False) -> None:
        with self._lock:
            if not (self._dirty or force):
                return
            payload = {
                "checkpoint_n": self.checkpoint_n,
                "buckets": {str(k): v for k, v in self.buckets.items()},
                "hot": {str(k): v for k, v in self.hot.items()},
            }
            self._dirty = False
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            opener = gzip.open if self.path.endswith(".gz") else open
            with opener(tmp, "wt") as f:
                json.dump(payload, f)
            os.replace(tmp, self.path)
        except Exception:
            logger.exception("frontier cache flush failed")
