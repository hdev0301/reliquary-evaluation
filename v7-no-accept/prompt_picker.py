"""Supabase-backed prepared-prompt picker.

Augments the default uniform-random prompt selection with a persistent
known-bad set keyed on (checkpoint_hash, prompt_idx). When the local
zone pre-check reports σ < SIGMA_MIN for a (checkpoint_hash, prompt_idx)
pair, the picker remembers it; subsequent windows (and subsequent miner
restarts, and sibling miner boxes sharing the same Supabase project)
skip those prompts in addition to the validator's own cooldown_prompts.

Schema (sql/migrate_supabase_schema.sql):
    prompt_outcomes(
      prompt_idx, checkpoint_hash, k, sigma, status,
      avg_completion_len, truncated_count, miner_hotkey, last_seen
    )
    PRIMARY KEY (prompt_idx, checkpoint_hash)

The PK is per-(prompt, ckpt) so each upsert REPLACES the previous row —
we keep the latest outcome only, not full history. That's the right
shape for the picker which only needs current state.

Out-of-zone submissions burn a /submit slot — at most 8 per
hotkey-window (MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW), so a single
wasted attempt is ~12% of the window's accept budget. Avoiding them is
the cheapest lever for raising the accept rate.

The cache scopes per checkpoint_hash because every validator checkpoint
advance retrains the model; a prompt that was hopeless under ckpt N
may be in-zone under ckpt N+1.
"""

from __future__ import annotations

import logging
import random as _random
from typing import Any

logger = logging.getLogger(__name__)


class SupabasePromptPicker:
    def __init__(
        self,
        url: str,
        key: str,
        *,
        hotkey: str,
        sigma_min: float,
        table: str = "prompt_outcomes",
    ) -> None:
        from supabase import create_client

        self._client = create_client(url, key)
        self._hotkey = hotkey
        self._sigma_min = float(sigma_min)
        self._table = table
        # Keyed on checkpoint_hash (HF revision string) — not ckpt_n —
        # because the validator's checkpoint_hash is the canonical
        # identity the protocol binds against.
        self._bad_by_ckpt: dict[str, set[int]] = {}
        self._good_by_ckpt: dict[str, set[int]] = {}
        # Per-(ckpt, prompt) σ from prompt_outcomes. Populated by
        # hydrate() and record(); read by pick() to weight selection
        # toward the σ ≈ 0.5 sweet spot (k=4/8) which is by far the
        # most common in-zone outcome at the validator (~52% of
        # accepted submissions land here per cross-miner sampling).
        # Prompts merged in from the R2 intel refresh that don't have
        # σ data fall back to a uniform weight via get(idx, 0.43).
        self._sigma_by_ckpt: dict[str, dict[int, float]] = {}
        # Per-process, per-ckpt tracking of attempted prompts. A prompt
        # we've already submitted (or skipped) in this session shouldn't
        # be re-picked: the validator's MAX_SUBMISSIONS_PER_PROMPT and
        # HASH_DEDUP_RETENTION_WINDOWS both eventually block re-submits,
        # and re-picking known_good entries that we've already used burns
        # iterations on guaranteed rejections (worst case loops the
        # picker on a single prompt forever — observed in prod when a
        # provisionally-accepted but GRAIL-rejected prompt stays in
        # known_good across the session).
        self._attempted_by_ckpt: dict[str, set[int]] = {}

    @property
    def sigma_min(self) -> float:
        return self._sigma_min

    def hydrate(self, checkpoint_hash: str) -> tuple[int, int]:
        """Load known-good/bad sets for *checkpoint_hash* into local cache.

        Returns ``(n_good, n_bad)``. Re-callable: idempotent and safe to
        invoke on every checkpoint advance. Empty *checkpoint_hash*
        (validator hasn't published yet) yields ``(0, 0)`` without
        querying Supabase.
        """
        if not checkpoint_hash:
            self._bad_by_ckpt[checkpoint_hash] = set()
            self._good_by_ckpt[checkpoint_hash] = set()
            return 0, 0
        good: set[int] = set()
        bad: set[int] = set()
        page_size = 1000
        offset = 0
        while True:
            res = (
                self._client.table(self._table)
                .select("prompt_idx, sigma")
                .eq("checkpoint_hash", checkpoint_hash)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            rows: list[dict[str, Any]] = res.data or []
            if not rows:
                break
            sigma_map = self._sigma_by_ckpt.setdefault(checkpoint_hash, {})
            for row in rows:
                idx = int(row["prompt_idx"])
                sigma = float(row["sigma"])
                if sigma < self._sigma_min:
                    bad.add(idx)
                else:
                    good.add(idx)
                    sigma_map[idx] = sigma
            if len(rows) < page_size:
                break
            offset += page_size
        # Bad wins over good within a single ckpt: once we've observed
        # σ < floor on a prompt at this ckpt, the model is unlikely to
        # produce a meaningfully different σ on a re-roll — sampling
        # variance is bounded and the ground-truth difficulty hasn't
        # changed. Re-picking the same prompt just burns a window for
        # a guaranteed zone-skip. (The good/bad partition resets on
        # every ckpt advance, which is when the model actually changes.)
        good -= bad
        self._bad_by_ckpt[checkpoint_hash] = bad
        self._good_by_ckpt[checkpoint_hash] = good
        return len(good), len(bad)

    def known_bad(self, checkpoint_hash: str) -> set[int]:
        return self._bad_by_ckpt.get(checkpoint_hash, set())

    def known_good(self, checkpoint_hash: str) -> set[int]:
        return self._good_by_ckpt.get(checkpoint_hash, set())

    def pick(
        self,
        env,
        cooldown_prompts: set[int],
        checkpoint_hash: str,
        *,
        rng: _random.Random | None = None,
        max_attempts: int = 1000,
    ) -> int:
        """Pick a prompt — prefer known-good for current ckpt over random.

        Lookup order:
          1. known_good \\ cooldown_prompts \\ session_attempted  — prep-vetted
             in-zone prompts not already tried this session.
          2. Random index not in (cooldown_prompts ∪ known_bad ∪
             session_attempted) — fallback when known-good is empty or
             exhausted.

        Raises ``RuntimeError`` if no eligible prompt exists.
        """
        rng = rng or _random
        n = len(env)
        good = self._good_by_ckpt.get(checkpoint_hash, set())
        bad = self._bad_by_ckpt.get(checkpoint_hash, set())
        attempted = self._attempted_by_ckpt.get(checkpoint_hash, set())

        good_eligible = good - cooldown_prompts - bad - attempted
        if good_eligible:
            # σ-weighted choice: prefer prompts with σ near 0.5 (k=4/8),
            # the maximum-σ outcome and ~52% of all in-zone submissions
            # per cross-miner sampling. Weight = max(0.01, σ − 0.40), so
            # σ=0.500 → 0.10, σ=0.484 → 0.084, σ=0.433 → 0.033.
            # Monotone in σ above 0.40; falls back to uniform if no σ
            # data (e.g. prompts merged in from R2 intel refresh that
            # don't have a recorded σ yet).
            sigma_map = self._sigma_by_ckpt.get(checkpoint_hash) or {}
            if sigma_map:
                candidates = list(good_eligible)
                weights = [
                    max(0.01, sigma_map.get(p, self._sigma_min) - 0.40)
                    for p in candidates
                ]
                return rng.choices(candidates, weights=weights, k=1)[0]
            return rng.choice(list(good_eligible))

        excluded = cooldown_prompts | bad | attempted
        if len(excluded) < n / 2:
            for _ in range(max_attempts):
                idx = rng.randrange(n)
                if idx not in excluded:
                    return idx
            raise RuntimeError(
                f"no eligible prompt after {max_attempts} attempts "
                f"(n={n}, excluded={len(excluded)})"
            )
        eligible = [i for i in range(n) if i not in excluded]
        if not eligible:
            raise RuntimeError("no eligible prompt — env fully excluded")
        return rng.choice(eligible)

    def record(
        self,
        *,
        prompt_idx: int,
        checkpoint_hash: str,
        k: int,
        sigma: float,
        status: str,
        avg_completion_len: int | None = None,
        truncated_count: int | None = None,
    ) -> None:
        """Persist outcome to Supabase and update local cache.

        Best-effort: a write failure logs and returns; the mining loop
        must keep running. Synchronous — wrap in ``asyncio.to_thread``
        from the engine's asyncio loop.
        """
        if not checkpoint_hash:
            # Pre-first-publish: validator's checkpoint_hash is empty.
            # Skip Supabase write — the schema's PK requires a non-empty
            # value, and there's no useful identity to scope the row to.
            self._attempted_by_ckpt.setdefault(checkpoint_hash, set()).add(prompt_idx)
            return
        row: dict[str, Any] = {
            "prompt_idx": int(prompt_idx),
            "checkpoint_hash": str(checkpoint_hash),
            "k": int(k),
            "sigma": float(sigma),
            "status": str(status),
            "miner_hotkey": self._hotkey,
        }
        if avg_completion_len is not None:
            row["avg_completion_len"] = int(avg_completion_len)
        if truncated_count is not None:
            row["truncated_count"] = int(truncated_count)
        try:
            self._client.table(self._table).upsert(
                row, on_conflict="prompt_idx,checkpoint_hash"
            ).execute()
        except Exception:
            logger.exception(
                "supabase upsert failed (prompt=%d ckpt=%s...); cache-only update",
                prompt_idx, checkpoint_hash[:12],
            )
        if sigma < self._sigma_min:
            self._bad_by_ckpt.setdefault(checkpoint_hash, set()).add(prompt_idx)
            self._good_by_ckpt.get(checkpoint_hash, set()).discard(prompt_idx)
            # Drop σ data for now-bad prompt; it won't be picked.
            self._sigma_by_ckpt.get(checkpoint_hash, {}).pop(prompt_idx, None)
        else:
            self._good_by_ckpt.setdefault(checkpoint_hash, set()).add(prompt_idx)
            self._bad_by_ckpt.get(checkpoint_hash, set()).discard(prompt_idx)
            self._sigma_by_ckpt.setdefault(checkpoint_hash, {})[prompt_idx] = (
                float(sigma)
            )
        # Always mark as attempted this session — even successful
        # provisional accepts shouldn't be re-picked: the validator may
        # later reject at GRAIL (distribution_suspicious, etc.) and
        # re-submitting wastes the next window.
        self._attempted_by_ckpt.setdefault(checkpoint_hash, set()).add(prompt_idx)


def build_picker_from_env(hotkey: str) -> SupabasePromptPicker | None:
    """Return a configured picker, or None if env vars aren't set.

    Reads:
      * RELIQUARY_SUPABASE_URL
      * RELIQUARY_SUPABASE_KEY
      * RELIQUARY_SIGMA_MIN (default 0.43, matches SIGMA_MIN constant)
    """
    import os

    url = os.environ.get("RELIQUARY_SUPABASE_URL", "").strip()
    key = os.environ.get("RELIQUARY_SUPABASE_KEY", "").strip()
    if not url or not key:
        return None
    try:
        sigma_min = float(os.environ.get("RELIQUARY_SIGMA_MIN", "0.43"))
    except ValueError:
        sigma_min = 0.43
    try:
        return SupabasePromptPicker(url, key, hotkey=hotkey, sigma_min=sigma_min)
    except Exception:
        logger.exception("failed to construct SupabasePromptPicker; disabling")
        return None
