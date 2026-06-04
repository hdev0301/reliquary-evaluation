"""Supabase -> submit consumer (GPU-FREE).

Reuses reliquary's MiningEngine.mine_window UNCHANGED — so all GRAIL binding,
signing, merkle, drand-round, and envelope logic stays canonical — but feeds it a
Supabase-backed pregen shim instead of a live GPU pregenerator. It pulls honest,
submit-ready groups for the validator's CURRENT checkpoint, binds them to the live
window randomness, signs with the stardev wallet, and POSTs. No model, no vLLM.

Run on a low-latency box near the validator to win the per-window slot race.

Env (source scripts/.env + supabase_pipeline/.env):
  RELIQUARY_VALIDATOR_URL, BT_WALLET_NAME, BT_HOTKEY, BT_NETWORK, NETUID,
  RELIQUARY_SUPABASE_URL/KEY/TABLE, RELIQUARY_USE_DRAND
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import sb  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sb.consumer")


class SupabasePregenShim:
    """Mimics the Pregenerator surface MiningEngine reads — no GPU, no generation."""

    def __init__(self, tokenizer, verifier, checkpoint_n, checkpoint_hash, model_name, store):
        self.tokenizer = tokenizer
        self.verifier = verifier
        self._n = int(checkpoint_n)
        self._hash = checkpoint_hash or ""
        self._model_name = model_name
        self.store = store

    def current(self):
        return (self._n, self._hash, self._model_name)

    def request_checkpoint(self, repo_id, revision, n):
        # Follow the validator's checkpoint advance; pop_groups then pulls rows
        # tagged with the new hash (the producer must be generating for it).
        self._n = int(n)
        self._hash = revision or ""
        if repo_id:
            self._model_name = repo_id
        log.info("following checkpoint advance -> n=%s hash=%s", n, (revision or "")[:12])

    # no-ops: the shim does not generate
    def set_cooldown_provider(self, _fn): pass
    def set_priority_provider(self, _fn): pass
    def start(self): pass
    def stop(self): pass


class _EnvStub:
    """MiningEngine._build_and_submit only reads env.name."""
    def __init__(self, name): self.name = name


def _resolve_hidden_dim(repo_id: str, revision: str) -> int:
    # generate_r_vec ignores hidden_dim (uses topk only), but GRAILVerifier wants an int.
    try:
        import json
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id, "config.json", revision=revision or None)
        cfg = json.load(open(p))
        return int(cfg.get("text_config", {}).get("hidden_size") or cfg.get("hidden_size") or 2560)
    except Exception as e:
        log.warning("hidden_dim from config failed (%s); using 2560 (does not affect r_vec)", e)
        return 2560


async def _run():
    import bittensor as bt
    import httpx
    from transformers import AutoTokenizer

    from reliquary.miner.engine import MiningEngine
    from reliquary.miner.submitter import get_window_state_v2
    from reliquary.protocol.grail_verifier import GRAILVerifier

    wallet_name = os.environ.get("BT_WALLET_NAME", "ronnywebdev")
    hotkey_name = os.environ.get("BT_HOTKEY", "stardev")
    url = os.environ["RELIQUARY_VALIDATOR_URL"]   # consumer requires an explicit validator URL
    use_drand = os.environ.get("RELIQUARY_USE_DRAND", "1") == "1"
    environment = os.environ.get("RELIQUARY_ENVIRONMENT_NAME", "openmathinstruct")

    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    hk_ss58 = wallet.hotkey.ss58_address
    log.info("consumer hotkey=%s ss58=%s validator=%s", hotkey_name, hk_ss58, url)

    # Resolve the validator's CURRENT published checkpoint (we only submit matching rows).
    async with httpx.AsyncClient(timeout=30) as client:
        state = await get_window_state_v2(url, client=client)
    ckpt_hash = state.checkpoint_revision or ""
    ckpt_n = state.checkpoint_n or 0
    repo_id = state.checkpoint_repo_id or os.environ.get("RELIQUARY_CHECKPOINT", "")
    if not ckpt_hash:
        log.warning("validator has no published checkpoint revision yet; consumer will idle until one appears")
    log.info("current checkpoint n=%s hash=%s repo=%s", ckpt_n, ckpt_hash[:12], repo_id)

    hidden_dim = _resolve_hidden_dim(repo_id, ckpt_hash)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=ckpt_hash or None)

    client = sb.SupabaseClient()
    store = sb.SupabaseConsumerStore(client, miner_hotkey=hk_ss58)
    log.info("supabase table=%s ready-for-ckpt=%d", client.table, store.ready_count(ckpt_hash))

    shim = SupabasePregenShim(tokenizer, verifier, ckpt_n, ckpt_hash, repo_id, store)
    engine = MiningEngine(shim, wallet, _EnvStub(environment), validator_url_override=url)

    log.info("entering submit loop (pulling prepared groups from Supabase)")
    # subtensor unused because validator_url_override is set.
    await engine.mine_window(None, 0, use_drand=use_drand)


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        log.info("consumer stopped")
