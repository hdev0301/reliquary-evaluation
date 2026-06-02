"""Supabase-backed persistence for miner pregen + prompt outcomes.

Two tables back the cache:

* ``prompt_outcomes``  — one row per (prompt_idx, checkpoint_hash); records
  whether the model scored the prompt in-zone, dud, or oof at this
  checkpoint. Used by the picker to skip prompts we've already classified.

* ``pregen_batches``   — full M=8 token batches for in-zone prompts at a
  given checkpoint. Tokens are checkpoint-bound but window-agnostic; the
  envelope+sketch are re-built at submit time, so a batch survives
  miner restarts without losing the ~160 s of GPU work it took to make.

All writes are best-effort and never block the critical path: failures
are logged and ignored. Reads happen once at startup (hydrate) and
on-demand from the picker. No per-submit network calls.

The schema is created by ``ensure_schema()`` (idempotent CREATE TABLE
IF NOT EXISTS) at startup, so a fresh Supabase project bootstraps
itself on first miner launch.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# DDL applied by ensure_schema(). Idempotent so re-running on an existing
# project is a no-op. RLS intentionally disabled — service_role bypasses
# it anyway, and we don't expose these tables to clients.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prompt_outcomes (
    prompt_idx       BIGINT NOT NULL,
    checkpoint_hash  TEXT NOT NULL,
    k                INTEGER NOT NULL,
    sigma            DOUBLE PRECISION NOT NULL,
    status           TEXT NOT NULL,
    avg_completion_len INTEGER,
    truncated_count  INTEGER,
    miner_hotkey     TEXT,
    last_seen        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (prompt_idx, checkpoint_hash)
);
CREATE INDEX IF NOT EXISTS idx_outcomes_ckpt_status
    ON prompt_outcomes (checkpoint_hash, status);

CREATE TABLE IF NOT EXISTS pregen_batches (
    id               BIGSERIAL PRIMARY KEY,
    prompt_idx       BIGINT NOT NULL,
    checkpoint_hash  TEXT NOT NULL,
    local_n          INTEGER NOT NULL,
    sigma            DOUBLE PRECISION NOT NULL,
    k                INTEGER NOT NULL,
    rollouts         JSONB NOT NULL,
    miner_hotkey     TEXT,
    tier             TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consumed_at      TIMESTAMPTZ,
    UNIQUE (prompt_idx, checkpoint_hash)
);
CREATE INDEX IF NOT EXISTS idx_pregen_unconsumed_ckpt
    ON pregen_batches (checkpoint_hash)
    WHERE consumed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_pregen_tier
    ON pregen_batches (checkpoint_hash, tier)
    WHERE consumed_at IS NULL;

-- Validator-accepted rollout hashes scraped from R2 archive windows.
-- Used as a pre-flight bloom filter: before submitting a freshly-generated
-- batch, hash each rollout and drop any that already appear here — those
-- would reject as HASH_DUPLICATE at the validator. ``rollout_hash`` is
-- the hex string of the validator's per-rollout hash (matches the
-- ``hash`` field in R2 window batch entries).
CREATE TABLE IF NOT EXISTS accepted_rollout_hashes (
    rollout_hash     TEXT NOT NULL,
    prompt_idx       BIGINT NOT NULL,
    checkpoint_hash  TEXT NOT NULL,
    window_n         BIGINT,
    miner_hotkey     TEXT,
    first_seen       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (rollout_hash)
);
CREATE INDEX IF NOT EXISTS idx_arh_prompt_ckpt
    ON accepted_rollout_hashes (prompt_idx, checkpoint_hash);
"""


@dataclass
class PromptOutcome:
    prompt_idx: int
    checkpoint_hash: str
    k: int
    sigma: float
    status: str  # 'good' | 'dud' | 'oof'
    avg_completion_len: Optional[int] = None
    truncated_count: Optional[int] = None
    miner_hotkey: Optional[str] = None


@dataclass
class PersistedBatch:
    prompt_idx: int
    checkpoint_hash: str
    local_n: int
    sigma: float
    k: int
    rollouts: list[dict]  # list of {tokens: [int], prompt_length: int, reward: float}
    miner_hotkey: Optional[str] = None
    # Difficulty tier — set by prep when the batch is saved. Lets the
    # consumer (engine) draw a diverse mix per submission window:
    #   "stable"      — high cross-ckpt confidence (score ≥ skip-prescreen)
    #   "proven"      — landed good ≥1 time historically
    #   "exploratory" — no prior history (random pick or first observation)
    tier: Optional[str] = None


class SupabaseCache:
    """Async-safe cache facade. All public methods are sync; callers run
    them via ``asyncio.to_thread`` if they want to avoid blocking the
    event loop (Supabase client is synchronous httpx under the hood).

    Init with ``url`` and ``key`` from env. If either is empty the cache
    enters disabled mode: every method is a no-op returning empty/None.
    Disabled mode lets us wire the cache into the engine unconditionally
    without adding ``if cache is not None`` checks at every call site.
    """

    def __init__(self, url: str, key: str, *, miner_hotkey: str | None = None):
        self.miner_hotkey = miner_hotkey
        self._client = None
        self.enabled = bool(url and key)
        if not self.enabled:
            logger.info("SupabaseCache disabled (no url/key)")
            return
        try:
            from supabase import create_client
            self._client = create_client(url, key)
            logger.info("SupabaseCache enabled url=%s", url)
        except Exception:
            logger.exception("SupabaseCache init failed; running disabled")
            self.enabled = False

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------
    def ensure_schema(self) -> bool:
        """Apply DDL. Requires a postgres function ``exec_sql(sql text)``
        in the target project; supabase-py has no direct DDL endpoint.
        If the function is missing, returns False and logs the SQL the
        operator should paste into the SQL editor by hand.
        """
        if not self.enabled:
            return False
        try:
            self._client.rpc("exec_sql", {"sql": SCHEMA_SQL}).execute()
            logger.info("supabase schema applied")
            return True
        except Exception as e:
            logger.warning(
                "supabase ensure_schema() failed (%s) — paste SCHEMA_SQL "
                "manually into the SQL editor", e,
            )
            return False

    # ------------------------------------------------------------------
    # prompt_outcomes
    # ------------------------------------------------------------------
    def upsert_outcome(self, o: PromptOutcome) -> None:
        if not self.enabled:
            return
        try:
            row = {
                "prompt_idx": int(o.prompt_idx),
                "checkpoint_hash": o.checkpoint_hash,
                "k": int(o.k),
                "sigma": float(o.sigma),
                "status": o.status,
                "avg_completion_len": (
                    int(o.avg_completion_len) if o.avg_completion_len is not None else None
                ),
                "truncated_count": (
                    int(o.truncated_count) if o.truncated_count is not None else None
                ),
                "miner_hotkey": o.miner_hotkey or self.miner_hotkey,
                "last_seen": "now()",
            }
            self._client.table("prompt_outcomes").upsert(row).execute()
        except Exception:
            logger.exception("upsert_outcome failed prompt=%d", o.prompt_idx)

    def prompt_stats_across_ckpts(
        self,
        since_iso: str | None = None,
        exclude_ckpts: set[str] | None = None,
    ) -> dict[int, dict]:
        """Return per-prompt aggregate stats across all ckpts.

        ``{prompt_idx: {'good': int, 'dud': int, 'oof': int,
                         'mean_k': float, 'observations': int,
                         'good_ts': list[str], 'bad_ts': list[str]}}``

        ``good_ts`` / ``bad_ts`` are ISO timestamps (the row's
        ``last_seen``) used by recency-weighted scoring downstream — newer
        observations count more than old ones.

        Args:
            since_iso: drop rows with ``last_seen < since_iso``. After a
                ckpt reset, set this to the reset timestamp so poisoned
                pre-reset rows don't pollute the priority queue.
            exclude_ckpts: drop rows whose ``checkpoint_hash`` is in this
                set. Use for surgical removal of specific bad ckpts when a
                time-based cutoff would over-exclude.
        """
        if not self.enabled:
            return {}
        try:
            stats: dict[int, dict] = {}
            page_size = 1000
            start = 0
            excl_list = list(exclude_ckpts) if exclude_ckpts else None
            while True:
                q = (
                    self._client.table("prompt_outcomes")
                    .select("prompt_idx,status,k,last_seen,checkpoint_hash")
                )
                if since_iso:
                    q = q.gte("last_seen", since_iso)
                if excl_list:
                    q = q.not_.in_("checkpoint_hash", excl_list)
                res = q.range(start, start + page_size - 1).execute()
                rows = res.data or []
                if not rows:
                    break
                for r in rows:
                    pid = int(r["prompt_idx"])
                    s = stats.setdefault(pid, {
                        "good": 0, "dud": 0, "oof": 0,
                        "_k_sum": 0, "observations": 0,
                        "good_ts": [], "bad_ts": [],
                    })
                    status = r.get("status") or ""
                    ts = r.get("last_seen")
                    if status == "good":
                        s["good"] += 1
                        if ts:
                            s["good_ts"].append(ts)
                    elif status in ("dud", "oof"):
                        s[status] += 1
                        if ts:
                            s["bad_ts"].append(ts)
                    s["_k_sum"] += int(r.get("k") or 0)
                    s["observations"] += 1
                if len(rows) < page_size:
                    break
                start += page_size
            for pid, s in stats.items():
                s["mean_k"] = s["_k_sum"] / max(1, s["observations"])
                del s["_k_sum"]
            return stats
        except Exception:
            logger.exception("prompt_stats_across_ckpts failed")
            return {}

    def fresh_good_pids(
        self, checkpoint_hash: str, since_iso: str | None = None,
    ) -> list[tuple[int, str]]:
        """Return ``[(prompt_idx, last_seen)]`` for ``status='good'`` rows
        under the current ckpt newer than ``since_iso``. Used by the prep
        picker to mid-run-refresh its priority queue with intel that other
        miners just produced (via scrape_intel) under the live ckpt.
        """
        if not self.enabled:
            return []
        try:
            q = (
                self._client.table("prompt_outcomes")
                .select("prompt_idx,last_seen")
                .eq("checkpoint_hash", checkpoint_hash)
                .eq("status", "good")
            )
            if since_iso:
                q = q.gt("last_seen", since_iso)
            res = q.execute()
            return [
                (int(r["prompt_idx"]), r["last_seen"])
                for r in (res.data or [])
            ]
        except Exception:
            logger.exception(
                "fresh_good_pids failed ckpt=%s since=%s",
                checkpoint_hash, since_iso,
            )
            return []

    def good_counts_across_ckpts(self) -> dict[int, int]:
        """Return ``{prompt_idx: count_of_distinct_ckpts_where_good}``.

        Prompts that landed `status='good'` across many ckpts are
        structurally mid-difficulty for this base model+env, regardless of
        which ckpt is live now. Used by the prep picker to prime the
        priority queue ahead of random sampling. Paginates the Supabase
        select since the REST endpoint caps at 1000 rows per call.
        """
        if not self.enabled:
            return {}
        try:
            seen: dict[int, set[str]] = {}
            page_size = 1000
            start = 0
            while True:
                res = (
                    self._client.table("prompt_outcomes")
                    .select("prompt_idx,checkpoint_hash")
                    .eq("status", "good")
                    .range(start, start + page_size - 1)
                    .execute()
                )
                rows = res.data or []
                if not rows:
                    break
                for r in rows:
                    seen.setdefault(int(r["prompt_idx"]), set()).add(
                        r["checkpoint_hash"]
                    )
                if len(rows) < page_size:
                    break
                start += page_size
            return {pid: len(ckpts) for pid, ckpts in seen.items()}
        except Exception:
            logger.exception("good_counts_across_ckpts failed")
            return {}

    def load_outcomes(self, checkpoint_hash: str) -> list[PromptOutcome]:
        if not self.enabled:
            return []
        try:
            res = (
                self._client.table("prompt_outcomes")
                .select("*")
                .eq("checkpoint_hash", checkpoint_hash)
                .execute()
            )
            return [
                PromptOutcome(
                    prompt_idx=int(r["prompt_idx"]),
                    checkpoint_hash=r["checkpoint_hash"],
                    k=int(r["k"]),
                    sigma=float(r["sigma"]),
                    status=r["status"],
                    avg_completion_len=r.get("avg_completion_len"),
                    truncated_count=r.get("truncated_count"),
                    miner_hotkey=r.get("miner_hotkey"),
                )
                for r in (res.data or [])
            ]
        except Exception:
            logger.exception("load_outcomes failed ckpt=%s", checkpoint_hash)
            return []

    # ------------------------------------------------------------------
    # accepted_rollout_hashes
    # ------------------------------------------------------------------
    def upsert_accepted_hashes(self, rows: list[dict]) -> int:
        """Bulk-insert validator-accepted rollout hashes from R2 windows.

        ``rows`` is a list of dicts with keys:
            rollout_hash (str, hex), prompt_idx (int),
            checkpoint_hash (str), window_n (int|None),
            miner_hotkey (str|None)

        Returns the number of rows submitted (upsert ignores collisions
        on ``rollout_hash`` so re-runs are idempotent).
        """
        if not self.enabled or not rows:
            return 0
        try:
            cleaned = []
            for r in rows:
                h = r.get("rollout_hash")
                if not h:
                    continue
                cleaned.append({
                    "rollout_hash": str(h),
                    "prompt_idx": int(r["prompt_idx"]),
                    "checkpoint_hash": str(r["checkpoint_hash"]),
                    "window_n": int(r["window_n"]) if r.get("window_n") is not None else None,
                    "miner_hotkey": r.get("miner_hotkey"),
                })
            if not cleaned:
                return 0
            (
                self._client.table("accepted_rollout_hashes")
                .upsert(cleaned, on_conflict="rollout_hash")
                .execute()
            )
            return len(cleaned)
        except Exception:
            logger.exception("upsert_accepted_hashes failed n=%d", len(rows))
            return 0

    def accepted_hashes_for_prompt(
        self, prompt_idx: int, checkpoint_hash: str | None = None,
    ) -> set[str]:
        """Return the set of hex hashes already accepted for a prompt.

        Pre-flight check: a freshly-generated rollout whose hash is in
        this set will reject as HASH_DUPLICATE at the validator — drop
        and re-sample with a different seed before submitting.
        Scoping to ``checkpoint_hash`` is recommended (rollout hashes
        only collide within the active hash_set window, which the
        validator keys per-ckpt). Pass ``None`` to check against all
        ckpts (paranoid mode).
        """
        if not self.enabled:
            return set()
        try:
            q = (
                self._client.table("accepted_rollout_hashes")
                .select("rollout_hash")
                .eq("prompt_idx", int(prompt_idx))
            )
            if checkpoint_hash is not None:
                q = q.eq("checkpoint_hash", checkpoint_hash)
            res = q.execute()
            return {r["rollout_hash"] for r in (res.data or [])}
        except Exception:
            logger.exception(
                "accepted_hashes_for_prompt failed prompt=%d ckpt=%s",
                prompt_idx, checkpoint_hash,
            )
            return set()

    # ------------------------------------------------------------------
    # pregen_batches
    # ------------------------------------------------------------------
    def save_batch(self, b: PersistedBatch) -> None:
        if not self.enabled:
            return
        try:
            row = {
                "prompt_idx": int(b.prompt_idx),
                "checkpoint_hash": b.checkpoint_hash,
                "local_n": int(b.local_n),
                "sigma": float(b.sigma),
                "k": int(b.k),
                "rollouts": b.rollouts,
                "miner_hotkey": b.miner_hotkey or self.miner_hotkey,
                "tier": b.tier,
                "consumed_at": None,
            }
            self._client.table("pregen_batches").upsert(
                row, on_conflict="prompt_idx,checkpoint_hash"
            ).execute()
        except Exception:
            logger.exception("save_batch failed prompt=%d", b.prompt_idx)

    def load_unconsumed_batches(self, checkpoint_hash: str) -> list[PersistedBatch]:
        if not self.enabled:
            return []
        try:
            res = (
                self._client.table("pregen_batches")
                .select("*")
                .eq("checkpoint_hash", checkpoint_hash)
                .is_("consumed_at", "null")
                .execute()
            )
            return [
                PersistedBatch(
                    prompt_idx=int(r["prompt_idx"]),
                    checkpoint_hash=r["checkpoint_hash"],
                    local_n=int(r["local_n"]),
                    sigma=float(r["sigma"]),
                    k=int(r["k"]),
                    rollouts=r["rollouts"],
                    miner_hotkey=r.get("miner_hotkey"),
                    tier=r.get("tier"),
                )
                for r in (res.data or [])
            ]
        except Exception:
            logger.exception(
                "load_unconsumed_batches failed ckpt=%s", checkpoint_hash,
            )
            return []

    def load_unconsumed_batches_by_tier(
        self, checkpoint_hash: str, tier: str
    ) -> list[PersistedBatch]:
        """Load unconsumed pregen batches filtered to a single tier.

        Used by the engine to draw a diverse mix per submission window
        (e.g. one ``stable`` + one ``proven`` + one ``exploratory``).
        """
        if not self.enabled:
            return []
        try:
            res = (
                self._client.table("pregen_batches")
                .select("*")
                .eq("checkpoint_hash", checkpoint_hash)
                .eq("tier", tier)
                .is_("consumed_at", "null")
                .execute()
            )
            return [
                PersistedBatch(
                    prompt_idx=int(r["prompt_idx"]),
                    checkpoint_hash=r["checkpoint_hash"],
                    local_n=int(r["local_n"]),
                    sigma=float(r["sigma"]),
                    k=int(r["k"]),
                    rollouts=r["rollouts"],
                    miner_hotkey=r.get("miner_hotkey"),
                    tier=r.get("tier"),
                )
                for r in (res.data or [])
            ]
        except Exception:
            logger.exception(
                "load_unconsumed_batches_by_tier failed ckpt=%s tier=%s",
                checkpoint_hash, tier,
            )
            return []

    def tier_counts(self, checkpoint_hash: str) -> dict[str, int]:
        """Return ``{tier: unconsumed_count}`` for the given ckpt.

        Useful for picker logic that wants to know what tiers are
        available before issuing a per-tier load.
        """
        if not self.enabled:
            return {}
        try:
            res = (
                self._client.table("pregen_batches")
                .select("tier")
                .eq("checkpoint_hash", checkpoint_hash)
                .is_("consumed_at", "null")
                .execute()
            )
            counts: dict[str, int] = {}
            for r in (res.data or []):
                t = r.get("tier") or "untagged"
                counts[t] = counts.get(t, 0) + 1
            return counts
        except Exception:
            logger.exception("tier_counts failed ckpt=%s", checkpoint_hash)
            return {}

    def mark_consumed(self, prompt_idx: int, checkpoint_hash: str) -> None:
        if not self.enabled:
            return
        try:
            self._client.table("pregen_batches").update(
                {"consumed_at": "now()"}
            ).eq("prompt_idx", prompt_idx).eq(
                "checkpoint_hash", checkpoint_hash
            ).execute()
        except Exception:
            logger.exception(
                "mark_consumed failed prompt=%d ckpt=%s",
                prompt_idx, checkpoint_hash,
            )

    def purge_other_checkpoints(self, current_ckpt_hash: str) -> None:
        """Unconditional purge — keeps only ``current_ckpt_hash``.

        DEPRECATED: prefer ``purge_old_checkpoints`` which keeps a
        sliding window of the most-recent N ckpts. The unconditional
        variant wipes the table flat on every advance, which creates
        a post-advance gap before the producer refills.
        """
        if not self.enabled:
            return
        try:
            self._client.table("pregen_batches").delete().neq(
                "checkpoint_hash", current_ckpt_hash
            ).execute()
        except Exception:
            logger.exception("purge_other_checkpoints failed")

    def purge_old_checkpoints(
        self, current_ckpt_hash: str, *, keep_last_n: int = 5,
    ) -> int:
        """Keep pregen rows for the ``keep_last_n`` most-recent ckpts plus
        the current one; delete everything else.

        Recency is decided by ``max(created_at)`` per ``checkpoint_hash``.
        ``current_ckpt_hash`` is always kept (added to the keep set even
        if it has no rows yet), so calling this at worker startup is safe.

        Returns the number of distinct checkpoint hashes purged.
        """
        if not self.enabled:
            return 0
        try:
            latest_by_ckpt: dict[str, str] = {}
            page = 1000
            offset = 0
            while True:
                res = (
                    self._client.table("pregen_batches")
                    .select("checkpoint_hash,created_at")
                    .range(offset, offset + page - 1)
                    .execute()
                )
                rows = res.data or []
                for row in rows:
                    h = row.get("checkpoint_hash")
                    ts = row.get("created_at")
                    if not h or not ts:
                        continue
                    prev = latest_by_ckpt.get(h)
                    if prev is None or ts > prev:
                        latest_by_ckpt[h] = ts
                if len(rows) < page:
                    break
                offset += page

            ranked = sorted(
                latest_by_ckpt.items(), key=lambda kv: kv[1], reverse=True,
            )
            keep = {current_ckpt_hash} | {h for h, _ in ranked[:keep_last_n]}
            purge_hashes = [h for h in latest_by_ckpt if h not in keep]
            if not purge_hashes:
                return 0
            (
                self._client.table("pregen_batches")
                .delete()
                .in_("checkpoint_hash", purge_hashes)
                .execute()
            )
            return len(purge_hashes)
        except Exception:
            logger.exception("purge_old_checkpoints failed")
            return 0


def cache_from_env(miner_hotkey: str | None = None) -> SupabaseCache:
    """Build a cache from ``RELIQUARY_SUPABASE_URL`` / ``RELIQUARY_SUPABASE_KEY``.
    Returns a disabled cache if either is empty.
    """
    url = os.environ.get("RELIQUARY_SUPABASE_URL", "").strip()
    key = os.environ.get("RELIQUARY_SUPABASE_KEY", "").strip()
    return SupabaseCache(url, key, miner_hotkey=miner_hotkey)


def resolve_hotkey(name_or_ss58: str | None) -> str | None:
    """Return ``name_or_ss58`` as-is when it already looks like an ss58
    address. Otherwise treat it as a bittensor hotkey *name* under the wallet
    ``$BT_WALLET_NAME`` and resolve to its ss58Address by reading
    ``~/.bittensor/wallets/<wallet>/hotkeys/<name>``. Falls back to the input
    string when resolution fails so the call is always safe.
    """
    import json
    if not name_or_ss58:
        return name_or_ss58
    if 46 <= len(name_or_ss58) <= 50:
        return name_or_ss58
    wallet = os.environ.get("BT_WALLET_NAME", "").strip()
    if not wallet:
        return name_or_ss58
    path = os.path.expanduser(
        f"~/.bittensor/wallets/{wallet}/hotkeys/{name_or_ss58}"
    )
    try:
        with open(path) as f:
            return json.load(f).get("ss58Address") or name_or_ss58
    except Exception as e:
        logger.warning("resolve_hotkey(%r) failed: %s", name_or_ss58, e)
        return name_or_ss58
