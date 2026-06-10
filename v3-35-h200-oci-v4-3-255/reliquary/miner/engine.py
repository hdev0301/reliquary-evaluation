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
_NOT_CONSUMED_REASONS = {"prompt_mismatch", "bad_envelope_signature", "wrong_checkpoint", "batch_filled",
                         # PREDICTIVE-FIRE mispredict signals — the prompt never entered the
                         # validator's dedup/cooldown set, so it must stay re-minable.
                         # CORRECTED (verified server.py:667-682): a wrong window-open-round guess
                         # does NOT surface as WRONG_RANDOMNESS. The validator verifies the envelope
                         # against ITS OWN active_batcher.randomness (server.py:667-679,
                         # ENFORCE_ENVELOPE_SIGNATURE on), so a wrong pred_randomness fails the
                         # synchronous envelope check FIRST -> BAD_ENVELOPE_SIGNATURE (already above),
                         # before the request is queued and long before batcher.py:810. We still list
                         # wrong_randomness (defensive) plus stale_round/future_round (reachable at the
                         # cheap pre-queue drand stage, batcher.py:530-532, if send-round != arrival-round)
                         # and prompt_in_cooldown (a stale-cooldown pre-stage can race a freshly-cooled
                         # prompt; batcher.py:653-654 stage "cooldown") — all reject pre-pool, debt-free.
                         "wrong_randomness", "stale_round", "future_round", "prompt_in_cooldown",
                         # window_mismatch: validator returns it at batcher.py:635-636, BEFORE the
                         # dedup hash add (batcher.py:708) -> never entered the pool. A predicted/raced
                         # window_start that is off by one (the OLD batcher is still active during the
                         # READY->OPEN drand-fetch gap) MUST stay re-minable. Was missing -> a single
                         # mispredict permanently burned a curated prompt (self-starvation).
                         # window_not_active: server.py:750 rejects synchronously when state != OPEN
                         # (the READY/TRAINING/PUBLISHING gap) before the request is ever queued —
                         # pre-pool, debt-free, must stay re-minable.
                         # rate_limited: server.py:824/840 HTTP-rejects past the per-hotkey
                         # MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW=8 quota, BEFORE the validation
                         # pipeline (constants.py:309-313) -> never deduped/cooled. The prompt is
                         # fine; we just spent our window quota (e.g. a relaunch where the prior
                         # instance used the same hotkey's quota, or >8 groups ready). Must stay
                         # re-minable so it can be submitted in a future window, not burned.
                         "window_mismatch", "window_not_active", "rate_limited"}


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


def derive_window_randomness(drand_round: int) -> str:
    """Window randomness for a given drand round — mirrors the validator's
    ``service.py::_derive_randomness`` (drand path, service.py:1643-1654) BYTE-FOR-BYTE.

    Validator pins window randomness to the round publishing AT window OPEN:
        beacon = get_beacon(str(round), use_drand=True)        # drand.py:534, use_fallback=False
        randomness = compute_window_randomness(None, beacon["randomness"], drand_round=beacon["round"])
    i.e. sha256(bytes.fromhex(beacon.randomness) + beacon.round.to_bytes(8,"big")).hexdigest()
    (chain.py:173-180). We pass ``beacon["round"]`` (the round the relay echoes back),
    NOT the requested int, so an adjacent-round relay still matches the validator.
    Raises on drand fetch failure (use_fallback=False) — callers treat a raise as
    "cannot pre-stage this boundary" and fall back to the live /state path.
    """
    from reliquary.infrastructure.drand import get_beacon

    beacon = get_beacon(round_id=str(drand_round), use_drand=True)
    return chain.compute_window_randomness(
        None, beacon["randomness"], drand_round=beacon["round"],
    )


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
        # BURN_COOLDOWN: on SN81 opencode BATCH_PROMPT_COOLDOWN_WINDOWS=1_000_000,
        # so a prompt_in_cooldown reject means the prompt is won and dead for ~19
        # years — NOT a transient freshly-cooled race. Treat it as consumed and
        # burn it, else the pool keeps re-firing the same dead prompts every
        # window (observed: same idx rejected cooldown across consecutive windows).
        self._not_consumed = set(_NOT_CONSUMED_REASONS)
        if _os_b.environ.get("RELIQUARY_BURN_COOLDOWN", "0") == "1":
            self._not_consumed.discard("prompt_in_cooldown")
            logger.info("BURN_COOLDOWN on: prompt_in_cooldown rejects are burned (permanent SN81 cooldown)")
        self._verdicts = {"accepted": 0, "rejected": 0, "by_reason": {}}
        # merkle_root -> prompt_idx for submissions in flight, so the verdict poller
        # can map an async REJECT back to its prompt and UN-BURN it when the reject
        # is transient/re-minable (the prompt never entered cooldown or the hash set).
        # Bounded; old entries are dropped (their verdicts simply won't un-burn).
        self._mr_to_idx: dict[str, int] = {}
        # Reject reasons that mean "re-minable" — the prompt was burned at submit
        # (sync response is always accepted=True/submitted) but the async verdict
        # shows it was NOT won and NOT consumed into cooldown/hash dedup, so it can
        # be mined again (esp. out_of_zone, which is checkpoint-transient, and
        # reward_mismatch, which the value-based reward fix resolves). grail_fail /
        # logprob_mismatch are EXCLUDED by default (systematic stack drift -> retry
        # just re-fails and wastes GPU). Override via RELIQUARY_UNBURN_REASONS.
        self._unburn_reasons = {
            r.strip() for r in os.environ.get(
                "RELIQUARY_UNBURN_REASONS", "out_of_zone,reward_mismatch"
            ).split(",") if r.strip()
        }
        # Decool-sniping: prompts that just EXITED the validator's cooldown were
        # recently rewarded (in-zone), so they are prime, currently-submittable
        # targets. We keep a bounded queue of recently-decooled idxs and hand it
        # to the pregenerator as a priority pool.
        from collections import deque
        self._decool_priority: deque = deque(maxlen=512)

        # --- PREDICTIVE FIRING (default OFF -> control flow byte-for-byte unchanged) ---
        # SAFE mode (RELIQUARY_PREDICT_FIRE=1): pre-build, then ONE /state confirm at the
        # boundary, POST only if confirmed. NOTE: verified ineffective — the validator only
        # exposes OPEN+randomness ~2.8s AFTER the boundary (service.py:1336-1338), by which
        # point the batch is sealed; SAFE confirm thus falls back every window. Kept only as a
        # zero-risk A/B baseline. BLIND mode is the lever.
        self._predict_fire = os.environ.get("RELIQUARY_PREDICT_FIRE", "0") == "1"
        # BLIND mode (RELIQUARY_PREDICT_BLIND=1): POST pre-built into the post-boundary window
        # WITHOUT a /state confirm. The ONLY variant that removes the ~156ms RTT + the multi-second
        # randomness-publish wait. Implies the master switch.
        self._predict_blind = os.environ.get("RELIQUARY_PREDICT_BLIND", "0") == "1"
        if self._predict_blind:
            self._predict_fire = True
        # Pre-stage this many ms before the predicted boundary (the project/sign burst runs here,
        # OFF the boundary critical path).
        self._predict_lead_ms = int(os.environ.get("RELIQUARY_PREDICT_LEAD_MS", "800"))
        # Fire this many ms AFTER the predicted boundary. The new batcher is only activated AFTER
        # the validator's post-boundary drand fetch (_activate_window, service.py:501-511); firing
        # at t_boundary+0 hits WINDOW_NOT_ACTIVE (503) or the OLD batcher (WINDOW_MISMATCH). We aim
        # into the ~0-2.8s post-boundary slot, before the competitor fills it. Tune from telemetry.
        self._predict_post_ms = int(os.environ.get("RELIQUARY_PREDICT_POST_MS", "300"))
        # Don't predict until window length L has been observed this many times.
        self._predict_min_windows = int(os.environ.get("RELIQUARY_PREDICT_MIN_WINDOWS", "3"))
        from collections import deque as _deque
        self._win_open_rounds: _deque = _deque(maxlen=16)   # recent R_open deltas (drand rounds)
        self._last_R_open: int | None = None                # last RECORDED window-open round
        # Seed window length L (drand rounds) so predictive firing can engage after ONE
        # window detection instead of waiting min_windows*~20min to learn it from scratch
        # (RELIQUARY_PREDICT_L>0). Live-observed deltas still refine it once min_windows
        # samples accumulate. 0 = learn from scratch (legacy).
        _L_seed = int(os.environ.get("RELIQUARY_PREDICT_L", "0"))
        self._win_len_L: int | None = _L_seed if _L_seed > 0 else None  # median(deltas) once >= min samples
        self._predicted_windows: set[int] = set()           # window_n values already predict-fired

        # BURST-POLL (RTT mitigation): during the OPEN-but-randomness-not-yet-published
        # gap (~2.8s post-boundary), keep several /state requests continuously in flight
        # so the one the validator processes just AFTER it publishes randomness returns it
        # — we learn randomness at ~one-way latency instead of (poll-gap + full RTT of a
        # post-publish poll). Worth ~one RTT on a high-latency link; pure read, no protocol
        # impact. Set RELIQUARY_BURST_WIDTH=0 to disable (falls back to single-poll).
        self._burst_width = int(os.environ.get("RELIQUARY_BURST_WIDTH", "3"))
        self._burst_stagger_s = float(os.environ.get("RELIQUARY_BURST_STAGGER_S", "0.03"))
        self._burst_budget_s = float(os.environ.get("RELIQUARY_BURST_BUDGET_S", "3.5"))

        # BLIND/PREDICTIVE FIRING IS STRUCTURALLY DEFEATED by the validator's
        # two-phase open: _open_window() builds the next batcher INACTIVE and
        # /submit rejects WINDOW_NOT_ACTIVE (server.py:750) until _activate_window()
        # flips OPEN — which happens only AFTER the ~2.8s post-boundary drand fetch
        # (service.py:1336-1338). So a POST into the [boundary, boundary+2.8s] gap
        # NEVER lands; the earliest a submission is accepted is the OPEN flip, which
        # /state reflects atomically. The only real lever is therefore REACTIVE:
        # detect the OPEN flip with minimum latency (co-location + burst-poll) and
        # POST with zero crypto on the critical path (the READY-anchored prestage
        # below). Leaving _predict_* honoured for A/B, but warn loudly if enabled.
        if self._predict_blind or self._predict_fire:
            logger.warning(
                "RELIQUARY_PREDICT_FIRE/BLIND is set, but predictive firing is "
                "structurally defeated by the validator's two-phase open "
                "(WINDOW_NOT_ACTIVE before _activate_window). It will fire into a "
                "closed window and waste submissions. Prefer co-location + "
                "RELIQUARY_PRESTAGE=1 (reactive pre-build). See engine.py header."
            )

        # READY-ANCHORED PRE-BUILD (RELIQUARY_PRESTAGE=1; default OFF until live A/B).
        # During the READY->OPEN gap, derive the next window's randomness DIRECTLY
        # from drand (we know it the instant the open round's beacon publishes,
        # ~2.8s before the validator republishes it via /state) and pre-build the
        # signed submissions OFF the critical path. The reactive OPEN path then
        # POSTs them as a pure network call — removing project_buckets + 64 ed25519
        # signs + merkle from the latency-critical fire. Fully fallback-safe: the
        # prestage is POSTed ONLY when /state OPEN confirms matching window_n +
        # randomness + checkpoint; on any mismatch the groups are restored to the
        # store and the unchanged legacy build-now path runs.
        self._prestage = os.environ.get("RELIQUARY_PRESTAGE", "0") == "1"
        self._prestaged: dict | None = None   # {window_n, randomness, ckpt, staged:[(g,rs,mr)]}
        self._prestage_round: int | None = None

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

    def _unburn(self, idx: int) -> None:
        """Make a wrongly-burned prompt re-minable again (transient async reject).
        Removes it from the in-memory + persisted blocklist."""
        if idx not in self._burned:
            return
        self._burned.discard(idx)
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

    def _restore_prestage(self) -> None:
        """Return any pre-built (but not yet POSTed) groups to the store so the
        legacy fire path can use them. Lossless: ``store.add`` re-keys by
        (checkpoint_hash, prompt_idx)."""
        if self._prestaged is None:
            return
        for g, _rs, _mr in self._prestaged["staged"]:
            self.pregen.store.add(g)
        self._prestaged = None
        self._prestage_round = None

    async def _maybe_prestage(self, state) -> None:
        """During READY, pre-build the next window's signed submissions for the
        randomness derived directly from drand for the round currently in progress.

        The validator opens the next window at the first drand boundary after it
        re-enters its loop, binding randomness to the round then in progress
        (service.py::_derive_randomness). We don't know that exact round during
        READY, so we build for the round in progress NOW and rebuild whenever the
        round ticks over — by the OPEN flip our latest prestage is for the round
        the validator actually bound, and the OPEN consume path verifies the
        randomness before POSTing (mismatch -> restore + legacy build). This is a
        pure pre-build optimisation: it never fires early and never burns prompts.
        """
        from reliquary.infrastructure.chain import compute_current_drand_round
        from reliquary.infrastructure.drand import get_current_chain

        _, cur_hash, model_name = self.pregen.current()
        target_hash = state.checkpoint_revision or ""
        if not cur_hash or cur_hash != target_hash:
            return  # pregen not on the published checkpoint yet -> would be WRONG_CHECKPOINT

        ci = get_current_chain()
        cur_round = compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])
        # Already staged for this exact (round, checkpoint, next-window)? nothing to do.
        if (
            self._prestaged is not None
            and self._prestage_round == cur_round
            and self._prestaged["ckpt"] == cur_hash
            and self._prestaged["window_n"] == state.window_n + 1
        ):
            return

        # Round advanced / checkpoint changed / window rolled: drop the stale build
        # and rebuild for the round now in progress.
        self._restore_prestage()
        try:
            randomness = derive_window_randomness(cur_round)
        except Exception as e:
            logger.debug("PRESTAGE skip: beacon R=%d unavailable (%s)", cur_round, e)
            return

        exclude = set(state.cooldown_prompts) | self._burned
        groups = self.pregen.store.pop_groups(
            cur_hash, exclude_idxs=exclude, n=self.max_per_window
        )
        if not groups:
            return
        r_vec = self._verifier.generate_r_vec(randomness)
        staged: list = []
        for g in groups:
            try:
                rs, mr = self._build_rollouts_and_merkle(g, randomness, r_vec, model_name)
                staged.append((g, rs, mr))
            except Exception as e:
                logger.warning("PRESTAGE build prompt=%d failed: %r", g.prompt_idx, e)
                self.pregen.store.add(g)  # lossless restore on build failure
        if not staged:
            return
        self._prestage_round = cur_round
        self._prestaged = {
            "window_n": state.window_n + 1,
            "randomness": randomness,
            "ckpt": cur_hash,
            "staged": staged,
        }
        logger.info(
            "PRESTAGE next_win=%d round=%d staged=%d rand=%s..",
            state.window_n + 1, cur_round, len(staged), randomness[:8],
        )

    async def _burst_await_randomness(self, client, url):
        """OPEN-but-no-randomness gap: keep up to ``_burst_width`` /state fetches
        continuously in flight and return the first GrpoBatchState that carries
        randomness (state still OPEN), catching publication within ~one-way latency
        instead of poll-gap + a full post-publish RTT. Returns None if randomness
        doesn't arrive within ``_burst_budget_s`` (caller falls back to single-poll)."""
        import asyncio
        from reliquary.miner.submitter import get_window_state_v2
        from reliquary.protocol.submission import WindowState

        if self._burst_width <= 0:
            return None
        deadline = time.monotonic() + self._burst_budget_s
        pending: set = set()
        try:
            while time.monotonic() < deadline:
                while len(pending) < self._burst_width:
                    pending.add(asyncio.ensure_future(asyncio.wait_for(
                        get_window_state_v2(url, env=self.env.name, client=client, timeout=1.5),
                        timeout=1.8,
                    )))
                done, pending = await asyncio.wait(
                    pending, timeout=self._burst_stagger_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for d in done:
                    try:
                        st = d.result()
                    except Exception:
                        continue
                    if st.randomness and st.state == WindowState.OPEN:
                        return st
            return None
        finally:
            for p in pending:
                p.cancel()

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
                            get_window_state_v2(url, env=self.env.name, client=client, timeout=1.5),
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

                    if state.state != WindowState.OPEN:
                        # READY-anchored pre-build: during the ~3-6s READY->OPEN gap
                        # build the next window's signed submissions for the
                        # drand-derived randomness so the OPEN fire is a pure POST.
                        if self._prestage and state.state == WindowState.READY:
                            try:
                                await self._maybe_prestage(state)
                            except Exception as e:
                                logger.debug("prestage error: %r", e)
                        await asyncio.sleep(0.15)
                        continue
                    if not state.randomness:
                        # BURST-POLL the post-boundary randomness-publish gap: pipeline
                        # overlapping /state fetches and take the first carrying randomness,
                        # then fall THROUGH to fire this same iteration (no extra re-poll).
                        # Catches publication ~one RTT sooner on a high-latency link.
                        _ready = await self._burst_await_randomness(client, url)
                        if _ready is None:
                            await asyncio.sleep(0.05)
                            continue
                        state = _ready

                    # New window → reset the per-window budget.
                    if state.window_n != cur_window_n:
                        cur_window_n = state.window_n
                        submitted_idxs = set()
                        _win_seen_t = time.monotonic()
                        # PREDICTOR: learn window length L (drand rounds). OPEN is drand-boundary
                        # aligned (service.py:1602-1625) so R_open is integral. Record ONLY for
                        # windows we did not predict-fire (those already recorded their R_open at
                        # fire time) — avoids the double-count that corrupts the L estimator.
                        # NB: L (open-to-open) includes the VARIABLE train+publish gap between
                        # windows (service.py:1339-1356) and a between-windows state=None gap was
                        # observed live, so L is noisy, not a fixed cadence — median + min-samples
                        # gate is deliberate, and BLIND mispredicts are debt-free per the safety notes.
                        if self._predict_fire and state.window_n not in self._predicted_windows:
                            self._record_window_open(state.window_n)
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

                    # ---- PREDICTIVE FIRING (additive; default OFF) ----------------------
                    # Predict the NEXT window-open round, pre-build the top-N for its derived
                    # randomness while THIS window is still open, then fire into the post-boundary
                    # slot — beating the ~156ms /state RTT and the multi-second randomness-publish
                    # wait. On any non-readiness/failure we fall through to the unchanged legacy fire.
                    if self._predict_fire and self._win_len_L is not None:
                        did = await self._predictive_fire(
                            client, url, cur_window_n, cur_hash, cur_model_name,
                            state, submitted_idxs, results,
                        )
                        if did:
                            # _predictive_fire fired the predicted window; advance our view of it
                            # so the legacy /state-detection won't re-record this window's R_open.
                            self._predicted_windows.add(did)
                            cur_window_n = did
                            submitted_idxs = set()
                            _win_seen_t = time.monotonic()
                            continue

                    # ---- READY-anchored prestage consume (additive; default OFF) -------
                    # If we pre-built this exact window's submissions during the READY
                    # gap, POST them now as a pure network call (zero crypto on the
                    # critical path). Only when window_n + randomness + checkpoint all
                    # match the live /state; otherwise restore and fall to legacy.
                    if self._prestaged is not None:
                        ps = self._prestaged
                        if (
                            ps["window_n"] == state.window_n
                            and ps["randomness"] == state.randomness
                            and ps["ckpt"] == cur_hash
                        ):
                            self._prestaged = None
                            self._prestage_round = None
                            skip = set(state.cooldown_prompts) | submitted_idxs | self._burned
                            fire_items = [
                                (g, rs, mr) for (g, rs, mr) in ps["staged"]
                                if g.prompt_idx not in skip
                            ]
                            # Any prompt freshly cooled since prestage is no longer
                            # firable here — drop its prepared group back so pregen
                            # can recycle the idx (re-minable, not burned).
                            for g, _rs, _mr in ps["staged"]:
                                if g.prompt_idx in skip:
                                    self.pregen.store.add(g)
                            if fire_items:
                                logger.info(
                                    "FIRE-PRESTAGED win=%d groups=%d t_since_window_seen=%.2fs valid_subs=%d",
                                    state.window_n, len(fire_items),
                                    time.monotonic() - _win_seen_t, state.valid_submissions,
                                )
                                fired = await asyncio.gather(
                                    *[
                                        self._stamp_and_post(
                                            client, url, g, state.window_n, mr, rs,
                                            state.randomness, cur_hash,
                                        )
                                        for (g, rs, mr) in fire_items
                                    ],
                                    return_exceptions=True,
                                )
                                for (g, _rs, _mr), res in zip(fire_items, fired):
                                    if isinstance(res, Exception):
                                        logger.warning("prestaged submit prompt=%d failed: %r", g.prompt_idx, res)
                                        continue
                                    submitted_idxs.add(g.prompt_idx)
                                    _reason = getattr(res, "reason", None)
                                    _reason = _reason.value if hasattr(_reason, "value") else _reason
                                    if getattr(res, "accepted", False) or _reason not in self._not_consumed:
                                        self._burn(g.prompt_idx)
                                    results.append(res)
                                # Loop back to fill any remaining per-window slots from the store.
                                continue
                        else:
                            # Stale prestage (wrong round/window/checkpoint): restore the
                            # popped groups so the legacy path below can use them.
                            self._restore_prestage()

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
                    logger.info(
                        "FIRE-TOTAL win=%d groups=%d wall=%.0fms",
                        state.window_n, len(groups),
                        (time.monotonic() - _last_fire_t) * 1000.0,
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
                        if getattr(res, "accepted", False) or _reason not in self._not_consumed:
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

        Legacy fast-path, behaviour unchanged: build rollouts+merkle, run the GROUPDIAG
        diagnostic (incl. RELIQUARY_REJECT_DUMP), then stamp drand_round + sign envelope +
        POST. Composed from the same two halves the predictive path uses
        (_build_rollouts_and_merkle + _stamp_and_post) so the paths cannot diverge.
        """
        _t0 = time.monotonic()
        rollout_submissions, merkle_root = self._build_rollouts_and_merkle(
            group, randomness, r_vec, model_name,
        )
        _t_build = time.monotonic()

        # --- DIAGNOSTIC (bad_termination / clone / hash_duplicate): per-group ground truth.
        # OFF the critical path by default (RELIQUARY_GROUPDIAG=1 to enable): the decode/
        # format/log of 8 rollouts per group is synchronous CPU that serializes across the
        # gathered groups and delays every POST in the seal race. Reject-dump still honoured.
        if os.environ.get("RELIQUARY_GROUPDIAG") == "1" or os.environ.get("RELIQUARY_REJECT_DUMP") == "1":
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

        _res = await self._stamp_and_post(
            client, url, group, window_n, merkle_root, rollout_submissions,
            randomness, checkpoint_hash,
        )
        # FIRETIME probe: build vs post latency per group (seal-race critical path).
        logger.info(
            "FIRETIME prompt=%d build=%.0fms post=%.0fms",
            group.prompt_idx, (_t_build - _t0) * 1000.0,
            (time.monotonic() - _t_build) * 1000.0,
        )
        return _res

    # ---------------------------------------------------------------------
    # PREDICTIVE FIRING (additive; reached only when RELIQUARY_PREDICT_FIRE=1)
    # ---------------------------------------------------------------------

    def _record_window_open(self, window_n: int) -> None:
        """Record the drand round this window opened on and update L."""
        from reliquary.infrastructure.chain import compute_current_drand_round
        from reliquary.infrastructure.drand import get_current_chain

        ci = get_current_chain()
        r_open = compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])
        if self._last_R_open is not None:
            delta = r_open - self._last_R_open
            if 1 <= delta <= 100000:          # guard dup sightings / restarts
                self._win_open_rounds.append(delta)
        self._last_R_open = r_open
        if len(self._win_open_rounds) >= self._predict_min_windows:
            s = sorted(self._win_open_rounds)
            self._win_len_L = s[len(s) // 2]  # median delta (rounds)
            logger.info(
                "PREDICT-LEARN window=%d R_open=%d L=%d (samples=%d deltas=%s)",
                window_n, r_open, self._win_len_L, len(self._win_open_rounds),
                list(self._win_open_rounds)[-6:],
            )

    async def _predictive_fire(
        self, client, url, cur_window_n: int, checkpoint_hash: str,
        model_name: str, state, submitted_idxs: set, results: list,
    ):
        """Pre-build the next window's burst and fire it into the post-boundary slot.

        Returns the predicted window_n (truthy) if it fired (SAFE-confirmed or BLIND),
        else 0 so the caller runs the unchanged legacy path.
        """
        from reliquary.infrastructure.chain import (
            compute_current_drand_round, seconds_until_next_drand_boundary,
        )
        from reliquary.infrastructure.drand import get_current_chain
        from reliquary.miner.submitter import get_window_state_v2
        from reliquary.protocol.submission import WindowState

        ci = get_current_chain()
        genesis, period = ci["genesis_time"], ci["period"]
        now = time.time()

        if self._last_R_open is None:
            return 0
        pred_R_open = self._last_R_open + self._win_len_L
        t_boundary = genesis + (pred_R_open - 1) * period
        lead_s = self._predict_lead_ms / 1000.0

        # Too early to stage: let the legacy loop service the current window.
        if now < t_boundary - lead_s:
            return 0
        # Way past the boundary (missed / L drifted): re-anchor via legacy detection.
        if now > t_boundary + period:
            return 0

        # window_n increments by exactly 1 per open (service.py:478).
        pred_window_n = cur_window_n + 1

        # --- derive the predicted window randomness (mirror the validator) ---
        try:
            pred_randomness = derive_window_randomness(pred_R_open)
        except Exception as e:
            logger.info("PREDICT skip: beacon for R_open=%d unavailable (%s)", pred_R_open, e)
            return 0

        # --- pre-build top-N (expensive crypto, OFF the boundary critical path) ---
        remaining = self.max_per_window - len(submitted_idxs)
        if remaining <= 0:
            return 0
        # Use the FRESHEST cooldown we have (this poll's state) + burned, NOT a stale
        # snapshot — a freshly-cooled prompt firing into the next window would otherwise
        # hit PROMPT_IN_COOLDOWN. (prompt_in_cooldown is now in _NOT_CONSUMED_REASONS as a
        # belt-and-braces guard, but excluding it up front avoids the wasted slot entirely.)
        exclude = set(state.cooldown_prompts) | submitted_idxs | self._burned
        groups = self.pregen.store.pop_groups(checkpoint_hash, exclude_idxs=exclude, n=remaining)
        if not groups:
            return 0
        r_vec = self._verifier.generate_r_vec(pred_randomness)
        staged = []  # (group, rollout_submissions, merkle_root)
        for g in groups:
            try:
                rs, mr = self._build_rollouts_and_merkle(g, pred_randomness, r_vec, model_name)
                staged.append((g, rs, mr))
            except Exception as e:
                logger.warning("PREDICT prebuild prompt=%d failed: %r", g.prompt_idx, e)
                self.pregen.store.add(g)  # lossless restore on prebuild failure
        logger.info(
            "PREDICT-STAGE pred_win=%d pred_R_open=%d t_boundary=%.3f staged=%d lead=%.0fms rand=%s..",
            pred_window_n, pred_R_open, t_boundary, len(staged), self._predict_lead_ms,
            pred_randomness[:8],
        )
        if not staged:
            return 0

        def _readd(items):
            for g, _rs, _mr in items:
                self.pregen.store.add(g)  # PregenStore.add re-keys by ckpt_hash+prompt_idx (lossless)

        try:
            # --- sleep PAST the boundary into the active-batcher window -------
            # The new batcher activates only AFTER the validator's post-boundary drand
            # fetch (service.py:501-511); firing at t_boundary+0 => WINDOW_NOT_ACTIVE /
            # WINDOW_MISMATCH. Sleep to the boundary, then _predict_post_ms beyond it.
            # Sleep to the PREDICTED window boundary (t_boundary), then _predict_post_ms
            # beyond it. (Was: seconds_until_next_drand_boundary(), which targets the next
            # 3s drand round — correct only when entered BEFORE t_boundary; if entered just
            # AFTER the boundary it overshoots by a full period -> over=+1 -> batch_filled.)
            delay = (t_boundary + (self._predict_post_ms / 1000.0)) - time.time()
            delay = max(delay, 0.001)
            if delay > 0.001:
                await asyncio.sleep(delay)

            # --- SAFE-mode confirm: ONE fast /state read (verified ~useless; baseline only)
            if not self._predict_blind:
                try:
                    st = await asyncio.wait_for(
                        get_window_state_v2(url, env=self.env.name, client=client, timeout=1.5), timeout=1.8,
                    )
                except Exception as e:
                    logger.info("PREDICT confirm /state failed (%s) -> fallback", e)
                    _readd(staged)
                    return 0
                confirmed = (
                    st.state == WindowState.OPEN
                    and st.window_n == pred_window_n
                    and bool(st.randomness)
                    and st.randomness == pred_randomness
                )
                logger.info(
                    "PREDICT-CONFIRM pred_win=%d state=%s state_win=%d rand_match=%s -> %s",
                    pred_window_n, getattr(st.state, "name", st.state), st.window_n,
                    bool(st.randomness) and st.randomness == pred_randomness,
                    "FIRE" if confirmed else "FALLBACK",
                )
                if not confirmed:
                    _readd(staged)
                    return 0

            # --- FIRE: stamp drand_round + sign envelope + POST -----------------
            fired = await asyncio.gather(
                *[
                    self._stamp_and_post(
                        client, url, g, pred_window_n, mr, rs, pred_randomness, checkpoint_hash,
                    )
                    for (g, rs, mr) in staged
                ],
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            _readd(staged)
            raise

        # Record THIS window's R_open now (fire time), so DIFF 4 won't double-count it.
        self._record_window_open(pred_window_n)

        n_acc = 0
        for (g, _rs, _mr), res in zip(staged, fired):
            if isinstance(res, Exception):
                logger.warning("PREDICT submit prompt=%d failed: %r", g.prompt_idx, res)
                self.pregen.store.add(g)            # not posted (or unknown) -> restore
                continue
            submitted_idxs.add(g.prompt_idx)
            _reason = getattr(res, "reason", None)
            _reason = _reason.value if hasattr(_reason, "value") else _reason
            if getattr(res, "accepted", False):
                n_acc += 1
            # Same burn policy as legacy: burn ONLY prompts the validator CONSUMED.
            # Mispredict reasons (bad_envelope_signature / stale_round / future_round /
            # prompt_in_cooldown / wrong_randomness) are in _NOT_CONSUMED_REASONS -> restore.
            if getattr(res, "accepted", False) or _reason not in self._not_consumed:
                self._burn(g.prompt_idx)
            else:
                self.pregen.store.add(g)
            results.append(res)
        logger.info(
            "PREDICT-FIRED pred_win=%d blind=%s posted=%d accepted=%d",
            pred_window_n, self._predict_blind, len(staged), n_acc,
        )
        return pred_window_n

    # ---------------------------------------------------------------------
    # Split builder/poster (legacy _build_and_submit becomes the wrapper, DIFF 7)
    # ---------------------------------------------------------------------

    def _build_rollouts_and_merkle(
        self, group: "PreparedGroup", randomness: str, r_vec, model_name: str,
    ):
        """Randomness-dependent, round-INdependent crypto (pre-stageable):
        project_buckets + sign_commit_binding per rollout + merkle. Identical to
        engine.py:419-481 minus the GROUPDIAG block (diagnostics stay off the
        predictive critical path). Returns (rollout_submissions, merkle_root)."""
        from reliquary.miner.grail_cache import project_buckets

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
                                  env_name=self.env.name)
            )
        merkle_root = _compute_merkle_root(rollout_submissions)
        return rollout_submissions, merkle_root

    async def _stamp_and_post(
        self, client, url, group: "PreparedGroup", window_n: int,
        merkle_root: str, rollout_submissions: "list[RolloutSubmission]",
        randomness: str, checkpoint_hash: str,
    ):
        """Round-dependent finish: stamp drand_round AT POST time, sign the envelope
        (which BINDS drand_round, engine.py:484-494), build the request, POST. drand_round
        is read as late as possible so it equals the validator's arrival round
        (zero-tolerance, server.py:934-968)."""
        from reliquary.miner.submitter import submit_batch_v2

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
        # Map merkle_root -> prompt_idx so the verdict poller can un-burn a
        # transient async reject. Bounded: clear when it grows past a window of
        # history (stale entries just lose their un-burn opportunity).
        if len(self._mr_to_idx) > 2000:
            self._mr_to_idx.clear()
        self._mr_to_idx[merkle_root] = group.prompt_idx
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
                        # Un-burn re-minable transient rejects (out_of_zone is
                        # checkpoint-transient; reward_mismatch is the now-fixed
                        # curation bug). The prompt was burned at submit but never
                        # entered cooldown/hash dedup, so it can be mined again.
                        if reason in self._unburn_reasons:
                            _mr = v.get("merkle_root", "")
                            _idx = self._mr_to_idx.pop(_mr, None)
                            if _idx is not None and _idx in self._burned:
                                self._unburn(_idx)
                                logger.info(
                                    "un-burned prompt=%d (re-minable reject=%s)", _idx, reason,
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
