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

# Reject reasons where the prompt NEVER entered the validator pool, so it must NOT
# be added to the persistent re-mine blocklist (it is still freely re-submittable).
# Reasons where the validator rejected BEFORE recording the rollout in its dedup set,
# so the prompt stays re-minable (burning it permanently = self-starvation). "batch_filled"
# is returned at batcher.py:634, BEFORE the dedup hash add at batcher.py:708 -> never deduped,
# safe to retry next window. Burning it was the cause of buffer starvation (every late submit
# burned a curated prompt -> store fills with excluded idxs -> nothing ready at window-open).
_NOT_CONSUMED_REASONS = {"prompt_mismatch", "bad_envelope_signature", "wrong_checkpoint", "batch_filled"}


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
        # Persistent cross-session blocklist of every prompt_idx we've EVER
        # submitted. The validator's cooldown set is rebuilt from a BOUNDED
        # lookback, so prompts burned in older sessions drop out of it while
        # their rollout hashes persist -> re-mining them = hash_duplicate. We
        # keep our own permanent record and exclude it from mining + submission.
        import os as _os_b, json as _json_b
        self._burned_path = _os_b.environ.get("RELIQUARY_BURNED_PATH", "/root/submitted_idx.json")
        try:
            self._burned: set[int] = set(int(i) for i in _json_b.load(open(self._burned_path)))
            logger.info("burned blocklist loaded: %d prompt idxs", len(self._burned))
        except Exception:
            self._burned = set()
        self._burned_dirty = 0
        self._verdicts = {"accepted": 0, "rejected": 0, "by_reason": {}}
        # Decool-sniping: prompts that just EXITED the validator's cooldown were
        # recently rewarded (in-zone), so they are prime, currently-submittable
        # targets. We keep a bounded queue of recently-decooled idxs and hand it
        # to the pregenerator as a priority pool.
        from collections import deque
        self._decool_priority: deque = deque(maxlen=512)

    def _burn(self, idx: int) -> None:
        """Record a prompt_idx as permanently submitted (any session) so we never
        re-mine/re-submit it. Without this, prompts burned beyond the validator's
        bounded cooldown lookback get re-discovered and rejected hash_duplicate.
        Persisted (batched) to survive restarts; self-heals (a colliding prompt is
        recorded on its reject and never retried)."""
        if idx in self._burned:
            return
        self._burned.add(idx)
        self._burned_dirty += 1
        if self._burned_dirty >= 5:
            self._save_burned()

    def _save_burned(self) -> None:
        import json as _json_b
        try:
            _json_b.dump(sorted(self._burned), open(self._burned_path, "w"))
            self._burned_dirty = 0
        except Exception as e:
            logger.debug("burned save failed: %s", e)

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
        self.pregen.set_cooldown_provider(lambda: self._latest_cooldown | self._burned)
        self.pregen.set_priority_provider(lambda: list(self._decool_priority))
        self.pregen.start()

        results: list = []
        cur_window_n = -1
        submitted_idxs: set[int] = set()
        _win_seen_t = time.monotonic()  # when WE first saw the current window (fire-latency probe)
        # SUBMIT-DELAY: top-miner telemetry (reliqua.ai API, rank-1 5DARq6 + rank-2 5F6VZ2)
        # shows accepted submissions land 8-42s into the 56s window (p50 ~20-23s) with 100%
        # acceptance and ZERO batch_filled — while OUR t=0 fires batch_fill in the SAME windows.
        # Firing at the window_n boundary races the prior window's seal. Hold this many seconds
        # after first seeing the window before firing. 0 = legacy fire-ASAP behaviour.
        _submit_delay_s = float(os.environ.get("RELIQUARY_SUBMIT_DELAY_S", "0"))
        # FIRE-PACING: top earner X (5CCnVDzE) spreads its 5-7 submissions across many drand
        # rounds over 46-106s and reaches the zone filter (rejects are out_of_zone, never
        # batch_filled). We instead burst all 8 in ONE drand round at t=0 and get rejected at the
        # SEAL stage. Fire at most _fire_per_burst groups, then wait _fire_pace_s (>= one 3s drand
        # round) before the next burst, so each submission lands in its own round. 0 = legacy burst.
        _fire_per_burst = int(os.environ.get("RELIQUARY_FIRE_PER_BURST", "0"))
        _fire_pace_s = float(os.environ.get("RELIQUARY_FIRE_PACE_S", "4"))
        _last_fire_t = 0.0
        # DETECTION-TIMING PROBE: prove whether the asyncio loop polls /state fast
        # (not GIL-starved) and capture how FULL each window already is the instant
        # we first see it OPEN+randomness-ready (state.valid_submissions). If a window
        # is already at B_BATCH when detected, it sealed before we could act -> over=0
        # is a window-regime/randomness issue, NOT our latency. Gated off by default.
        _probe = os.environ.get("RELIQUARY_DETECT_PROBE", "0") == "1"
        _poll_count = 0
        _last_hb = time.monotonic()

        async with httpx.AsyncClient(timeout=30) as client:
            verdict_task = asyncio.create_task(self._poll_verdicts(url, hotkey, client))
            try:
                while True:
                    try:
                        # DETECTION-LATENCY FIX: the default /state timeout is 60s and
                        # _get_with_retry attempts 3x (=> up to 60*3 + backoff = 183s of the
                        # detection loop BLOCKED on a single slow/unreachable /state moment,
                        # which the validator does have). A blocked loop can't see a window
                        # OPEN -> we detect 30-183s late -> batch_filled even though we fire at
                        # t_since_window_seen=0.00s. Hard-bound the poll to 1.5s so a slow
                        # /state costs ~1.5s and the outer loop immediately re-polls (every
                        # 0.05-0.15s) instead of going blind. Pure client-side timeout; no
                        # protocol/validator-rule impact.
                        state = await asyncio.wait_for(
                            get_window_state_v2(url, client=client, timeout=1.5),
                            timeout=1.8,
                        )
                    except SubmissionError:
                        await asyncio.sleep(0.2)
                        continue
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.debug("state fetch slow/failed: %s", e)
                        await asyncio.sleep(0.2)
                        continue

                    # DETECTION-TIMING PROBE heartbeat: counts polls/interval so we
                    # can see the real loop frequency (GIL-starve check) and what the
                    # loop observes while "waiting" (state/fill/randomness).
                    if _probe:
                        _poll_count += 1
                        _now = time.monotonic()
                        if _now - _last_hb >= 10.0:
                            logger.info(
                                "PROBE-HB polls=%d in %.1fs (%.0f/s) | win=%s state=%s valid_subs=%s rand=%s",
                                _poll_count, _now - _last_hb, _poll_count / max(_now - _last_hb, 1e-9),
                                state.window_n, getattr(state.state, "name", state.state),
                                state.valid_submissions, bool(state.randomness),
                            )
                            _poll_count = 0
                            _last_hb = _now

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
                        await asyncio.sleep(0.05 if state.state == WindowState.OPEN else 0.15)
                        continue

                    # New window → reset the per-window budget.
                    if state.window_n != cur_window_n:
                        cur_window_n = state.window_n
                        submitted_idxs = set()
                        _win_seen_t = time.monotonic()
                        if _probe:
                            # Fill level the INSTANT we first see this window ready.
                            # valid_subs already high => sealed-before-we-saw-it (regime),
                            # valid_subs low => the window is fresh and we have a real shot.
                            logger.info(
                                "PROBE-DETECT win=%d valid_subs=%d state=%s rand_ready=%s",
                                state.window_n, state.valid_submissions,
                                getattr(state.state, "name", state.state), bool(state.randomness),
                            )

                    # SUBMIT-DELAY: don't fire at the window-open boundary (t=0 -> batch_filled).
                    # Wait until the window has been open _submit_delay_s, matching the top
                    # miners' ~20s sweet spot, then fire into the cleanly-open batch. We still
                    # detect promptly (DETECTION-LATENCY fix above) so we never MISS the window.
                    if _submit_delay_s > 0 and (time.monotonic() - _win_seen_t) < _submit_delay_s:
                        await asyncio.sleep(0.25)
                        continue

                    remaining = self.max_per_window - len(submitted_idxs)
                    if remaining <= 0:
                        await asyncio.sleep(0.1)
                        continue

                    # FIRE-PACING gate: cap this burst and space bursts >= one drand round apart
                    # so submissions spread across rounds (mimics top earner X) instead of one
                    # single-round 8-burst that the validator seal-rejects as batch_filled.
                    if _fire_per_burst > 0:
                        if (time.monotonic() - _last_fire_t) < _fire_pace_s:
                            await asyncio.sleep(0.2)
                            continue
                        remaining = min(remaining, _fire_per_burst)

                    # Only submit when the pregen pool is on the validator's
                    # published checkpoint — otherwise it's WRONG_CHECKPOINT.
                    # Normalise None/"" (bootstrap: no checkpoint published yet).
                    _, cur_hash, cur_model_name = self.pregen.current()
                    target_hash = state.checkpoint_revision or ""
                    if cur_hash != target_hash:
                        await asyncio.sleep(0.1)
                        continue

                    exclude = set(state.cooldown_prompts) | submitted_idxs | self._burned
                    groups = self.pregen.store.pop_groups(cur_hash, exclude_idxs=exclude, n=remaining)
                    if not groups:
                        await asyncio.sleep(0.1)
                        continue
                    _last_fire_t = time.monotonic()  # fire-pacing: stamp for the next-burst gate

                    randomness = state.randomness
                    r_vec = self._verifier.generate_r_vec(randomness)
                    logger.info(
                        "FIRE win=%d groups=%d t_since_window_seen=%.2fs valid_subs=%d",
                        state.window_n, len(groups), time.monotonic() - _win_seen_t,
                        state.valid_submissions,
                    )
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
                        # Persistent blocklist (_burn) ONLY for prompts the validator
                        # actually CONSUMED. Pre-pool rejects (prompt_mismatch / bad
                        # signature / wrong checkpoint) never entered the pool -> keep
                        # them re-minable. Critical while the validator hasn't deployed
                        # the boxed-prompt fix yet (every submit currently prompt_mismatches);
                        # without this, we'd permanently burn our curatable prompts for nothing.
                        _reason = getattr(res, "reason", None)
                        _reason = _reason.value if hasattr(_reason, "value") else _reason
                        if getattr(res, "accepted", False) or _reason not in _NOT_CONSUMED_REASONS:
                            self._burn(g.prompt_idx)   # anti hash_duplicate
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
                RolloutSubmission(tokens=pr.all_tokens, reward=pr.reward, commit=commit,
                                  env_name=self.env.name)  # multi-env protocol: per-rollout env tag
            )

        # --- DIAGNOSTIC (bad_termination / clone / hash_duplicate): per-group ground truth.
        # Logs each rollout's reward, completion_length, last 3 token ids (termination tail),
        # and the count of DISTINCT completions in the group (within-group dedup / clone risk).
        try:
            comps = [tuple(pr.all_tokens[pr.prompt_length:]) for pr in group.rollouts]
            distinct = len(set(comps))
            tails = [
                f"r={pr.reward:.0f} clen={pr.completion_length} p_stop={pr.p_stop:.4f} last3={pr.all_tokens[-3:]}"
                for pr in group.rollouts
            ]
            logger.info(
                "GROUPDIAG prompt=%d distinct_completions=%d/%d | %s",
                group.prompt_idx, distinct, len(group.rollouts), " || ".join(tails),
            )
            # Dump exact submitted tokens so we can replicate the validator's
            # has_eos_padding / verify_termination OFFLINE (has_eos_padding needs no GPU).
            # GATED: this is a synchronous multi-MB append in the time-critical submit
            # burst (blocks the event loop -> late submits -> batch_filled). Off by
            # default; set RELIQUARY_REJECT_DUMP=1 only when debugging terminations.
            if os.environ.get("RELIQUARY_REJECT_DUMP") == "1":
                import json as _json_d
                with open("/root/sn81-miner/diagnostics/reject_dump.jsonl", "a") as _df:
                    for pr in group.rollouts:
                        _df.write(_json_d.dumps({
                            "prompt_idx": group.prompt_idx,
                            "prompt_length": pr.prompt_length,
                            "completion_length": pr.completion_length,
                            "reward": pr.reward,
                            "p_stop": pr.p_stop,
                            "n_tokens": len(pr.all_tokens),
                            "tokens": pr.all_tokens,
                        }) + "\n")
        except Exception as _e:
            logger.warning("GROUPDIAG failed: %r", _e)

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
                    # Rich telemetry the validator already returns — pinpoints WHY
                    # batch_filled fires and which lever moves it: submitted vs
                    # arrival drand round, the 8th-distinct seal trigger round and our
                    # delta past it, reject_stage, and the validator-side timings
                    # (queue_wait/verify) that reveal whether our long completions
                    # validate too slowly to make the first-8-distinct seal.
                    _sub, _arr, _trig = v.get("submitted_drand_round"), v.get("arrival_drand_round"), v.get("seal_trigger_round")
                    _over = (_sub - _trig) if (_sub is not None and _trig is not None) else None
                    _diag = ("sub=%s arr=%s trig=%s over=%s tol=%s stage=%s rank=%s qwait=%sms verify=%sms total=%sms"
                             % (_sub, _arr, _trig, _over, v.get("drand_tolerance"), v.get("reject_stage"),
                                v.get("canonical_rank"), v.get("queue_wait_ms"), v.get("verify_ms"), v.get("total_ms")))
                    if v.get("accepted"):
                        self._verdicts["accepted"] += 1
                        logger.info(
                            "verdict ACCEPTED win=%s mr=%s | %s",
                            v.get("window_n"), str(v.get("merkle_root", ""))[:12], _diag,
                        )
                    else:
                        self._verdicts["rejected"] += 1
                        reason = v.get("reason", "?")
                        self._verdicts["by_reason"][reason] = (
                            self._verdicts["by_reason"].get(reason, 0) + 1
                        )
                        logger.warning(
                            "verdict REJECTED win=%s mr=%s reason=%s | %s",
                            v.get("window_n"), str(v.get("merkle_root", ""))[:12], reason, _diag,
                        )
                    last_ts = max(last_ts, float(v.get("ts", 0.0)))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(5)
