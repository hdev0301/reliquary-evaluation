"""Miner engine — vLLM generation + HuggingFace GRAIL proof + pregeneration.

Architecture (v2.3, pregeneration fast-fire):

  * A :class:`~reliquary.miner.pregen.Pregenerator` owns the vLLM generator and
    the HF proof model and, on its own daemon thread, keeps a pool of
    **prepared** in-zone groups for the current checkpoint: 8 genuine rollouts
    each, with the randomness-free GRAIL artifacts (per-token buckets +
    token_logprobs) already cached and every behavioural gate pre-screened.

  * This engine runs the async control loop: poll ``/state``, hand checkpoint
    advances to the pregenerator, and at window-open **burst-fire** up to
    ``MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW`` distinct non-cooldown prompts in
    the first drand round. Building a submission from a prepared group is pure
    CPU/crypto — project cached buckets with ``r_vec``, sign the commit + the
    envelope, compute the merkle root, stamp ``drand_round`` at the POST
    instant — so the whole burst lands in well under a second.

  * A verdict poller (`/verdicts/{hotkey}`) surfaces the real ACCEPTED/REJECT
    outcomes seconds after the worker drains each submission.

Generation, proof, zone filtering and anti-rejection screening all live in the
pregenerator; this module is the thin, latency-critical submit path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

import random as _random

from reliquary.constants import (
    GRAIL_PROOF_VERSION,
    LAYER_INDEX,
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
)
from reliquary.infrastructure import chain
from reliquary.protocol.signatures import sign_commit_binding, sign_envelope
from reliquary.protocol.submission import BatchSubmissionRequest, RolloutSubmission

if TYPE_CHECKING:
    from reliquary.environment.base import Environment
    from reliquary.miner.pregen import PreparedGroup, Pregenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (shared with tests; unchanged semantics)
# ---------------------------------------------------------------------------

def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    Retained for tests and as the pregenerator's default candidate sampler
    fallback. Smarter, frontier-predicting samplers are injected into the
    pregenerator instead (see docs/mining.md §Prompt selection strategy).
    """
    rng = rng or _random
    n = len(env)
    if len(cooldown_prompts) < n / 2:
        for _ in range(max_attempts):
            idx = rng.randrange(n)
            if idx not in cooldown_prompts:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(n) if i not in cooldown_prompts]
    if not eligible:
        raise RuntimeError("no eligible prompt — env fully in cooldown")
    return rng.choice(eligible)


def _compute_merkle_root(rollouts) -> str:
    """Compute Merkle root over rollout leaves — returns 64-char hex.

    Canonical JSON (sort_keys, compact) so the root is deterministic and
    refactor-stable. Identical to the pre-pregeneration implementation.
    """
    import hashlib
    import json

    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(json.dumps(r.tokens, separators=(",", ":")).encode())
        h.update(json.dumps(r.reward).encode())
        h.update(json.dumps(r.commit, sort_keys=True, separators=(",", ":")).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        new = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            new.append(hashlib.sha256(left + right).digest())
        leaves = new
    return leaves[0].hex()


def _current_drand_round_at_send() -> int:
    """Drand quicknet round currently in progress at wall-clock now.

    Computed immediately before each POST so the attached round matches the
    validator's view at receipt (the gate is zero-tolerance; see
    docs/mining.md §STALE_ROUND).
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    return compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])


# ---------------------------------------------------------------------------
# Mining engine
# ---------------------------------------------------------------------------

class MiningEngine:
    """Async control + fast submit path over a :class:`Pregenerator`."""

    def __init__(
        self,
        pregen: "Pregenerator",
        wallet,
        env: "Environment",
        *,
        validator_url_override: str | None = None,
        max_per_window: int = MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    ) -> None:
        self.pregen = pregen
        self.wallet = wallet
        self.env = env
        self.tokenizer = pregen.tokenizer
        self.validator_url_override = validator_url_override
        self.max_per_window = max_per_window
        self._verifier = pregen.verifier
        self._latest_cooldown: set[int] = set()
        self._verdicts = {"accepted": 0, "rejected": 0, "by_reason": {}}
        # Decool-sniping: prompts that just EXITED the validator's cooldown were
        # recently rewarded (in-zone), so they are prime, currently-submittable
        # targets. We keep a bounded queue of recently-decooled idxs and hand it
        # to the pregenerator as a priority pool.
        from collections import deque
        self._decool_priority: deque = deque(maxlen=512)

    async def mine_window(self, subtensor, window_start: int = 0, use_drand: bool = True) -> list:
        """Poll /state, drive the pregenerator across checkpoints, and burst-fire
        prepared groups at window-open. Runs until cancelled."""
        import httpx

        from reliquary.constants import POLL_INTERVAL_SECONDS
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url, get_window_state_v2,
        )
        from reliquary.protocol.submission import WindowState

        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        hotkey = self.wallet.hotkey.ss58_address
        self.pregen.set_cooldown_provider(lambda: self._latest_cooldown)
        self.pregen.set_priority_provider(lambda: list(self._decool_priority))
        self.pregen.start()

        results: list = []
        cur_window_n = -1
        submitted_idxs: set[int] = set()

        async with httpx.AsyncClient(timeout=30) as client:
            verdict_task = asyncio.create_task(self._poll_verdicts(url, hotkey, client))
            try:
                while True:
                    try:
                        state = await get_window_state_v2(url, client=client)
                    except SubmissionError:
                        await asyncio.sleep(0.5)
                        continue
                    except Exception as e:
                        logger.debug("state fetch failed: %s", e)
                        await asyncio.sleep(0.5)
                        continue

                    # Detect prompts that just EXITED cooldown → prime snipe
                    # targets (recently rewarded = likely still in-zone, and now
                    # submittable). Diff against the previous snapshot.
                    new_cd = set(state.cooldown_prompts)
                    decooled = self._latest_cooldown - new_cd
                    if decooled:
                        for idx in decooled:
                            self._decool_priority.append(int(idx))
                        logger.info(
                            "decool-snipe: +%d exited cooldown (queue=%d) sample=%s",
                            len(decooled), len(self._decool_priority), sorted(decooled)[:8],
                        )
                    self._latest_cooldown = new_cd

                    # Hand any checkpoint advance to the pregen thread.
                    cur_n, cur_hash, _ = self.pregen.current()
                    if state.checkpoint_n > cur_n and state.checkpoint_revision:
                        self.pregen.request_checkpoint(
                            state.checkpoint_repo_id, state.checkpoint_revision, state.checkpoint_n
                        )

                    if state.state != WindowState.OPEN or not state.randomness:
                        await asyncio.sleep(0.05 if state.state == WindowState.OPEN else 0.5)
                        continue

                    # New window → reset the per-window budget.
                    if state.window_n != cur_window_n:
                        cur_window_n = state.window_n
                        submitted_idxs = set()

                    remaining = self.max_per_window - len(submitted_idxs)
                    if remaining <= 0:
                        await asyncio.sleep(0.1)
                        continue

                    # Only submit when the pregen pool is on the validator's
                    # published checkpoint — otherwise it's WRONG_CHECKPOINT.
                    # Normalise None/"" (bootstrap: no checkpoint published yet).
                    _, cur_hash, cur_model_name = self.pregen.current()
                    target_hash = state.checkpoint_revision or ""
                    if cur_hash != target_hash:
                        await asyncio.sleep(0.1)
                        continue

                    exclude = set(state.cooldown_prompts) | submitted_idxs
                    groups = self.pregen.store.pop_groups(cur_hash, exclude_idxs=exclude, n=remaining)
                    if not groups:
                        await asyncio.sleep(0.1)
                        continue

                    randomness = state.randomness
                    r_vec = self._verifier.generate_r_vec(randomness)
                    fired = await asyncio.gather(
                        *[
                            self._build_and_submit(
                                client, url, g, state.window_n, randomness, r_vec,
                                cur_hash, cur_model_name,
                            )
                            for g in groups
                        ],
                        return_exceptions=True,
                    )
                    for g, res in zip(groups, fired):
                        if isinstance(res, Exception):
                            logger.warning("submit prompt=%d failed: %r", g.prompt_idx, res)
                            continue
                        submitted_idxs.add(g.prompt_idx)
                        results.append(res)
            finally:
                verdict_task.cancel()

        return results

    async def _build_and_submit(
        self, client, url, group: "PreparedGroup", window_n: int,
        randomness: str, r_vec, checkpoint_hash: str, model_name: str,
    ):
        """Assemble one prepared group into a signed request and POST it.

        All work here is CPU: project cached buckets, sign commit + envelope,
        merkle, drand_round. ``drand_round`` is read as late as possible.
        """
        from reliquary.miner.grail_cache import project_buckets
        from reliquary.miner.submitter import submit_batch_v2

        rollout_submissions: list[RolloutSubmission] = []
        for pr in group.rollouts:
            commitments = project_buckets(pr.buckets, r_vec)
            signature = sign_commit_binding(
                pr.all_tokens, randomness, model_name, LAYER_INDEX, commitments, self.wallet,
            )
            commit = {
                "tokens": pr.all_tokens,
                "commitments": commitments,
                "proof_version": GRAIL_PROOF_VERSION,
                "model": {"name": model_name, "layer_index": LAYER_INDEX},
                "signature": signature.hex(),
                "beacon": {"randomness": randomness},
                "rollout": {
                    "prompt_length": pr.prompt_length,
                    "completion_length": pr.completion_length,
                    "success": pr.reward > 0,
                    "total_reward": pr.reward,
                    "advantage": 0.0,
                    "token_logprobs": pr.token_logprobs,
                },
            }
            rollout_submissions.append(
                RolloutSubmission(tokens=pr.all_tokens, reward=pr.reward, commit=commit)
            )

        merkle_root = _compute_merkle_root(rollout_submissions)
        current_round = _current_drand_round_at_send()
        nonce = os.urandom(16).hex()
        envelope_sig = sign_envelope(
            wallet=self.wallet,
            miner_hotkey=self.wallet.hotkey.ss58_address,
            window_start=window_n,
            prompt_idx=group.prompt_idx,
            merkle_root=merkle_root,
            checkpoint_hash=checkpoint_hash,
            drand_round=current_round,
            randomness=randomness,
            nonce=nonce,
        ).hex()
        request = BatchSubmissionRequest(
            miner_hotkey=self.wallet.hotkey.ss58_address,
            prompt_idx=group.prompt_idx,
            window_start=window_n,
            merkle_root=merkle_root,
            rollouts=rollout_submissions,
            checkpoint_hash=checkpoint_hash,
            drand_round=current_round,
            nonce=nonce,
            envelope_signature=envelope_sig,
        )
        resp = await submit_batch_v2(url, request, client=client)
        logger.info(
            "submitted window=%d prompt=%d sigma=%.3f round=%d accepted=%s reason=%s",
            window_n, group.prompt_idx, group.sigma, current_round, resp.accepted,
            resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
        )
        return resp

    async def _poll_verdicts(self, url: str, hotkey: str, client) -> None:
        """Surface real ACCEPTED/REJECT verdicts from /verdicts/{hotkey}.

        Seed ``last_ts`` to now so we only report verdicts for THIS process's
        submissions, not the validator's pre-existing ring-buffer history.
        """
        last_ts = time.time()
        while True:
            try:
                r = await client.get(
                    f"{url}/verdicts/{hotkey}", params={"since": last_ts}, timeout=5.0
                )
                for v in r.json().get("verdicts", []):
                    if v.get("accepted"):
                        self._verdicts["accepted"] += 1
                        logger.info(
                            "verdict ACCEPTED win=%s mr=%s",
                            v.get("window_n"), str(v.get("merkle_root", ""))[:12],
                        )
                    else:
                        self._verdicts["rejected"] += 1
                        reason = v.get("reason", "?")
                        self._verdicts["by_reason"][reason] = (
                            self._verdicts["by_reason"].get(reason, 0) + 1
                        )
                        logger.warning(
                            "verdict REJECTED win=%s mr=%s reason=%s",
                            v.get("window_n"), str(v.get("merkle_root", ""))[:12], reason,
                        )
                    last_ts = max(last_ts, float(v.get("ts", 0.0)))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(5)
