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
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consumed_at      TIMESTAMPTZ,
    UNIQUE (prompt_idx, checkpoint_hash)
);
CREATE INDEX IF NOT EXISTS idx_pregen_unconsumed_ckpt
    ON pregen_batches (checkpoint_hash)
    WHERE consumed_at IS NULL;
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
                )
                for r in (res.data or [])
            ]
        except Exception:
            logger.exception(
                "load_unconsumed_batches failed ckpt=%s", checkpoint_hash,
            )
            return []

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
        """On checkpoint advance, evict batches for old checkpoints.
        Their tokens are no longer valid against the current model.
        Prompt outcomes are KEPT — different ckpt = different row, and the
        history is useful for the picker's exploration heuristics.
        """
        if not self.enabled:
            return
        try:
            self._client.table("pregen_batches").delete().neq(
                "checkpoint_hash", current_ckpt_hash
            ).execute()
        except Exception:
            logger.exception("purge_other_checkpoints failed")


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
