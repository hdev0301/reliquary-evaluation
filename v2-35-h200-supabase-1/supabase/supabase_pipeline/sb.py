"""Supabase REST client + (de)serialization for the SN81 honest pregen pipeline.

Dependency-light (httpx + torch + numpy): importable on a GPU-FREE submit box.
A "group" row in `pregen` carries the full GRAIL artifacts (tokens +
per-token int8 `buckets` + token_logprobs) so the consumer can bind to the live
window randomness, sign, and POST without any model.

Env:
  RELIQUARY_SUPABASE_URL    https://<ref>.supabase.co
  RELIQUARY_SUPABASE_KEY    service_role (or a key with rights on the table)
  RELIQUARY_SUPABASE_TABLE  default: pregen
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import numpy as np
import torch


def _cfg() -> tuple[str, str, str]:
    url = os.environ["RELIQUARY_SUPABASE_URL"].rstrip("/")
    key = os.environ["RELIQUARY_SUPABASE_KEY"]
    table = os.environ.get("RELIQUARY_SUPABASE_TABLE", "pregen")
    return url, key, table


# --------------------------------------------------------------------------- #
# Duck-typed group/rollout objects the MiningEngine submit path consumes.
# (Same attribute surface as reliquary.miner.pregen.PreparedGroup/PreparedRollout,
#  but with no heavy imports — the engine only reads these attributes.)
# --------------------------------------------------------------------------- #
@dataclass
class SBRollout:
    all_tokens: list[int]
    prompt_length: int
    completion_length: int
    reward: float
    token_logprobs: list[float]
    buckets: torch.Tensor          # CPU int8 [seq_len, topk]
    p_stop: float = 0.0


@dataclass
class SBGroup:
    prompt_idx: int
    checkpoint_hash: str
    sigma: float
    rollouts: list[SBRollout] = field(default_factory=list)
    _row_id: int | None = None     # supabase row id (for claim/consume bookkeeping)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def _ser_buckets(buckets: torch.Tensor) -> dict:
    arr = buckets.detach().to("cpu").contiguous().numpy()
    return {
        "buckets_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        "buckets_shape": list(arr.shape),
        "buckets_dtype": str(arr.dtype),
    }


def _deser_buckets(d: dict) -> torch.Tensor:
    arr = np.frombuffer(base64.b64decode(d["buckets_b64"]), dtype=np.dtype(d["buckets_dtype"]))
    arr = arr.reshape(d["buckets_shape"]).copy()      # copy -> writable tensor
    return torch.from_numpy(arr)


def _ser_rollout(pr) -> dict:
    d = {
        "tokens": [int(t) for t in pr.all_tokens],
        "prompt_length": int(pr.prompt_length),
        "completion_length": int(pr.completion_length),
        "reward": float(pr.reward),
        "token_logprobs": [float(x) for x in pr.token_logprobs],
        "p_stop": float(getattr(pr, "p_stop", 0.0)),
    }
    d.update(_ser_buckets(pr.buckets))
    return d


def _deser_rollout(d: dict) -> SBRollout:
    return SBRollout(
        all_tokens=list(d["tokens"]),
        prompt_length=int(d["prompt_length"]),
        completion_length=int(d["completion_length"]),
        reward=float(d["reward"]),
        token_logprobs=list(d["token_logprobs"]),
        buckets=_deser_buckets(d),
        p_stop=float(d.get("p_stop", 0.0)),
    )


def serialize_group(group, *, model_name: str, hidden_dim: int, miner_hotkey: str,
                    tier: str = "honest_first8") -> dict:
    """group: any object with .prompt_idx/.checkpoint_hash/.sigma/.rollouts."""
    n_correct = sum(1 for r in group.rollouts if float(r.reward) >= 1.0)
    return {
        "prompt_idx": int(group.prompt_idx),
        "checkpoint_hash": str(group.checkpoint_hash),
        "model_name": str(model_name),
        "hidden_dim": int(hidden_dim),
        "sigma": float(group.sigma),
        "n_correct": int(n_correct),
        "rollouts": [_ser_rollout(r) for r in group.rollouts],
        "miner_hotkey": str(miner_hotkey),
        "tier": tier,
        "status": "ready",
    }


def deserialize_row(row: dict) -> SBGroup:
    return SBGroup(
        prompt_idx=int(row["prompt_idx"]),
        checkpoint_hash=str(row["checkpoint_hash"]),
        sigma=float(row["sigma"]),
        rollouts=[_deser_rollout(r) for r in row["rollouts"]],
        _row_id=row.get("id"),
    )


# --------------------------------------------------------------------------- #
# REST client (PostgREST)
# --------------------------------------------------------------------------- #
class SupabaseClient:
    def __init__(self, timeout: float = 15.0):
        self.url, self.key, self.table = _cfg()
        self.base = f"{self.url}/rest/v1/{self.table}"
        self._h = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=timeout)

    # ---- producer side ----
    def upsert_group(self, row: dict) -> bool:
        """Insert one group; on (ckpt,prompt,hotkey) conflict, merge (idempotent)."""
        r = self._client.post(
            self.base,
            params={"on_conflict": "checkpoint_hash,prompt_idx,miner_hotkey"},
            headers={**self._h, "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=[row],
        )
        if r.status_code not in (200, 201, 204):
            raise RuntimeError(f"supabase upsert {r.status_code}: {r.text[:200]}")
        return True

    def ready_count(self, checkpoint_hash: str, miner_hotkey: str) -> int:
        r = self._client.get(
            self.base,
            params={"select": "id", "status": "eq.ready",
                    "checkpoint_hash": f"eq.{checkpoint_hash}",
                    "miner_hotkey": f"eq.{miner_hotkey}", "limit": "1"},
            headers={**self._h, "Prefer": "count=exact"},
        )
        # PostgREST returns the count in Content-Range: 0-0/<count>
        cr = r.headers.get("content-range", "*/0")
        try:
            return int(cr.split("/")[-1])
        except Exception:
            return 0

    # ---- consumer side ----
    def fetch_ready(self, checkpoint_hash: str, miner_hotkey: str, limit: int) -> list[dict]:
        r = self._client.get(
            self.base,
            params={"select": "*", "status": "eq.ready",
                    "checkpoint_hash": f"eq.{checkpoint_hash}",
                    "miner_hotkey": f"eq.{miner_hotkey}",
                    "order": "created_at.asc", "limit": str(limit)},
            headers=self._h,
        )
        r.raise_for_status()
        return r.json()

    def claim(self, ids: list[int]) -> None:
        """Mark rows consumed so a single consumer never re-pulls them."""
        if not ids:
            return
        idcsv = ",".join(str(int(i)) for i in ids)
        now = datetime.now(timezone.utc).isoformat()
        r = self._client.patch(
            self.base,
            params={"id": f"in.({idcsv})"},
            headers={**self._h, "Prefer": "return=minimal"},
            json={"status": "consumed", "consumed_at": now},
        )
        if r.status_code not in (200, 204):
            raise RuntimeError(f"supabase claim {r.status_code}: {r.text[:200]}")

    def close(self):
        self._client.close()


# --------------------------------------------------------------------------- #
# Consumer-side store: mimics PregenStore.pop_groups for MiningEngine.
# --------------------------------------------------------------------------- #
class SupabaseConsumerStore:
    def __init__(self, client: SupabaseClient, miner_hotkey: str):
        self.client = client
        self.hotkey = miner_hotkey

    def pop_groups(self, checkpoint_hash: str, exclude_idxs=None, n: int = 1) -> list[SBGroup]:
        exclude = set(int(x) for x in (exclude_idxs or set()))
        # over-fetch so the in-Python exclude filter (cooldown can be huge) still
        # yields up to n; cooldown sets are too large for a not.in.() URL filter.
        rows = self.client.fetch_ready(checkpoint_hash, self.hotkey, limit=max(n * 5, 25))
        picked, claim_ids = [], []
        for row in rows:
            if int(row["prompt_idx"]) in exclude:
                continue
            picked.append(deserialize_row(row))
            claim_ids.append(row["id"])
            if len(picked) >= n:
                break
        self.client.claim(claim_ids)   # claim==consume (single consumer); batch_filled groups are replenished by the producer
        return picked

    # MiningEngine never calls these on the submit path, but keep them safe:
    def set_active_checkpoint(self, *_a, **_k): pass
    def ready_count(self, checkpoint_hash: str) -> int:
        return self.client.ready_count(checkpoint_hash, self.hotkey)
    def prepared_idxs(self, *_a, **_k): return set()


if __name__ == "__main__":
    # offline serialization round-trip self-test (no Supabase needed)
    g = SBGroup(prompt_idx=42, checkpoint_hash="abc", sigma=0.5, rollouts=[
        SBRollout(all_tokens=[1, 2, 3, 4], prompt_length=2, completion_length=2,
                  reward=1.0, token_logprobs=[-0.1, -0.2],
                  buckets=torch.randint(-7, 8, (4, 12), dtype=torch.int8), p_stop=0.3),
    ])
    row = serialize_group(g, model_name="m", hidden_dim=2560, miner_hotkey="hk")
    g2 = deserialize_row({**row, "id": 1})
    assert g2.rollouts[0].all_tokens == g.rollouts[0].all_tokens
    assert torch.equal(g2.rollouts[0].buckets, g.rollouts[0].buckets)
    assert g2.rollouts[0].buckets.dtype == torch.int8
    print("sb.py serialization round-trip OK (buckets int8 preserved)")
