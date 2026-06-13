"""The hot path: turn a pregenerated, in-zone group into a wire submission.

Everything expensive already happened during pregen. Here we only:
  1. project the cached buckets through the window's r_vec + sign each commit
     (``ProofBuilder.finalize_commit``),
  2. compute the Merkle root,
  3. read the drand round *immediately before* the POST (boundary-safe),
  4. sign the envelope binding,
  5. POST /submit.

This collapses fire-time latency from the ~60–100 s gen+proof of the reference
miner to a few milliseconds, so we land in the earliest drand round and before
the batch fills (``BATCH_FILLED``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)


def current_drand_round() -> int:
    """Drand-quicknet round in progress right now (matches the reference miner)."""
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    return compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])


def seconds_into_round() -> float:
    """How far (s) we are into the current 3 s drand bucket, in [0, period)."""
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    period = float(ci["period"])
    return (time.time() - float(ci["genesis_time"])) % period


async def wait_for_safe_drand_window(margin_s: float) -> None:
    """Sleep past a drand boundary if we're within ``margin_s`` of one.

    With ``DRAND_ROUND_BACKWARD_TOLERANCE = 0`` the validator gates the round
    with zero tolerance, so a POST that crosses the 3 s boundary in flight is
    rejected ``STALE_ROUND``/``FUTURE_ROUND``. Firing in the *first* part of a
    bucket guarantees the round we stamp is the round the validator sees.
    """
    from reliquary.infrastructure.drand import get_current_chain

    period = float(get_current_chain()["period"])
    into = seconds_into_round()
    if period - into < margin_s:
        await asyncio.sleep((period - into) + 0.001)


async def fire_group(
    *,
    url: str,
    client,
    wallet,
    proof_builder,
    group,                      # ReadyGroup (mining.common.pregen)
    randomness: str,
    window_n: int,
    checkpoint_hash: str,
    reward_for: "callable | None" = None,
):
    """Finalize + sign + submit one pregenerated group. Returns the response.

    ``reward_for(rollout_index) -> float`` supplies the per-rollout reward
    field. OpenCode is validator-authoritative so this is the 0.0 placeholder;
    OpenMath would pass its verified local reward here.
    """
    from reliquary.miner.engine import _compute_merkle_root
    from reliquary.miner.submitter import SubmissionError, submit_batch_v2
    from reliquary.protocol.signatures import sign_envelope
    from reliquary.protocol.submission import BatchSubmissionRequest, RolloutSubmission

    commits = [
        proof_builder.finalize_commit(p, randomness, wallet) for p in group.payloads
    ]
    rollouts = []
    for i, (payload, commit) in enumerate(zip(group.payloads, commits)):
        reward = reward_for(i) if reward_for is not None else 0.0
        rollouts.append(
            RolloutSubmission(
                tokens=payload.tokens,
                reward=reward,
                commit=commit,
                env_name=group.env_name,
            )
        )
    merkle_root = _compute_merkle_root(rollouts)

    # Round + envelope must be the LAST things before the POST.
    drand_round = current_drand_round()
    nonce = os.urandom(16).hex()
    envelope_sig = sign_envelope(
        wallet=wallet,
        miner_hotkey=wallet.hotkey.ss58_address,
        window_start=window_n,
        prompt_idx=group.prompt_idx,
        merkle_root=merkle_root,
        checkpoint_hash=checkpoint_hash,
        drand_round=drand_round,
        randomness=randomness or "",
        nonce=nonce,
    ).hex()

    request = BatchSubmissionRequest(
        miner_hotkey=wallet.hotkey.ss58_address,
        prompt_idx=group.prompt_idx,
        window_start=window_n,
        merkle_root=merkle_root,
        rollouts=rollouts,
        checkpoint_hash=checkpoint_hash,
        drand_round=drand_round,
        nonce=nonce,
        envelope_signature=envelope_sig,
    )
    try:
        resp = await submit_batch_v2(url, request, client=client)
        print(
            f"@@FIRED env={group.env_name} window={window_n} prompt={group.prompt_idx} "
            f"round={drand_round} sigma={getattr(group,'sigma',0):.3f} "
            f"accepted={resp.accepted} reason={getattr(resp.reason,'value',resp.reason)}",
            flush=True,
        )
        return resp
    except SubmissionError as exc:
        logger.error("fire submit failed (prompt=%d): %s", group.prompt_idx, exc)
        return None
