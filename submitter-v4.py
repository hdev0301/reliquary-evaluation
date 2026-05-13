"""HTTP client used by miners to push GRPO submissions to the validator.

v4 overlay drop-in for ``reliquary/miner/submitter.py``. Backwards-compatible
with the v1/v3 surface (``submit_batch_v2``, ``get_window_state_v2``,
``discover_validator_url``, ``SubmissionError``, ``NoValidatorFoundError``).

Adds three helpers the v4 engine expects:
- ``discover_validator_urls`` — top-N permitted validators by stake
- ``submit_batch_v2_multi`` — parallel broadcast, one shared serialized body
- ``prewarm_connections`` — issue /health to prime TLS/TCP keep-alive pools

Latency wins on the /submit critical path:
1. ``orjson`` for serialization (~3-5x faster than stdlib on float-heavy bodies).
2. Direct dict-build skips pydantic ``model_dump()`` walk (~1 fewer full
   traversal of the rollout tree).
3. ``token_logprobs`` rounded to ``_LOGPROB_DECIMAL_PLACES`` digits before
   encoding — drops per-float wire size from ~22B to ~8B. With 8 rollouts
   averaging ~5k completion tokens each, this trims the body from ~700KB
   to ~220KB. Body-upload time is THE FIFO race lever — see validator
   ``_submit_worker`` (single-threaded, drains queue in TCP arrival order).
4. ``Content-Type: application/json`` set explicitly with raw bytes body
   so httpx doesn't re-serialize.
5. Multi-broadcast does ZERO retries — a competing miner won't wait for
   our 7s exponential backoff. Single-URL path retains the retry ladder.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import Any

import httpx

try:
    import orjson  # noqa: F401
    _HAS_ORJSON = True
except ImportError:
    _HAS_ORJSON = False

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
)

logger = logging.getLogger(__name__)

# Retry policy for SINGLE-validator submit path (v1/v3 compatibility) and
# for /state GETs. The MULTI-validator path skips retries — see
# ``submit_batch_v2_multi`` for rationale.
_RETRY_DELAYS = (1.0, 2.0, 4.0)
_DEFAULT_TIMEOUT = 60.0

# Float precision for ``token_logprobs``. Validator's LOGPROB_IS_EPS=0.10
# (median exp(|dev|) - 1) is ~4 orders of magnitude looser than 4-decimal
# rounding error, well inside the bf16 noise floor honest miners already
# hit on cross-GPU runs. Body-size win: ~3x smaller wire payload.
_LOGPROB_DECIMAL_PLACES = 4


class NoValidatorFoundError(RuntimeError):
    """No metagraph entry advertises a usable validator endpoint."""


class SubmissionError(RuntimeError):
    """All submission retries exhausted (single-URL path) or fatal HTTP error."""


# ---------------------------------------------------------------------------
# Validator discovery
# ---------------------------------------------------------------------------

def discover_validator_url(metagraph: Any, port: int = VALIDATOR_HTTP_PORT) -> str:
    """Return the HTTP URL of the first validator advertised on the metagraph.

    Kept for v1/v3 callers. Multi-validator code paths use
    ``discover_validator_urls`` instead.
    """
    permits = getattr(metagraph, "validator_permit", None)
    axons = getattr(metagraph, "axons", None)
    if permits is None or axons is None:
        raise NoValidatorFoundError(
            "metagraph missing validator_permit or axons attributes"
        )
    for uid, (permit, axon) in enumerate(zip(permits, axons)):
        if not permit:
            continue
        ip = getattr(axon, "ip", None)
        if not ip or ip in ("0.0.0.0", ""):
            continue
        axon_port = getattr(axon, "port", None) or port
        return f"http://{ip}:{axon_port}"
    raise NoValidatorFoundError("no validator with permit and routable axon")


def discover_validator_urls(
    metagraph: Any,
    *,
    max_n: int = 5,
    port: int = VALIDATOR_HTTP_PORT,
) -> list[str]:
    """Top-``max_n`` permitted validators by stake descending.

    Each validator scores miners independently and contributes to the
    on-chain EMA. Broadcasting to N validators = N independent score
    contributions per submission. Higher-stake validators carry more
    consensus weight, so stake-ordering maximises expected weight.
    """
    permits = getattr(metagraph, "validator_permit", None)
    axons = getattr(metagraph, "axons", None)
    if permits is None or axons is None:
        raise NoValidatorFoundError(
            "metagraph missing validator_permit or axons attributes"
        )

    # ``S`` is bittensor's stake tensor; treat as 0 if absent.
    stakes = getattr(metagraph, "S", None)

    candidates: list[tuple[float, str]] = []
    n = len(permits)
    for uid in range(n):
        permit = permits[uid]
        if not permit:
            continue
        axon = axons[uid]
        ip = getattr(axon, "ip", None)
        if not ip or ip in ("0.0.0.0", ""):
            continue
        axon_port = getattr(axon, "port", None) or port
        url = f"http://{ip}:{axon_port}"
        stake_val = 0.0
        if stakes is not None:
            try:
                stake_val = float(stakes[uid])
            except (TypeError, ValueError, IndexError):
                stake_val = 0.0
        candidates.append((stake_val, url))

    if not candidates:
        raise NoValidatorFoundError("no validator with permit and routable axon")

    # Highest stake first; dedupe URLs (multiple validators behind a single axon).
    candidates.sort(key=lambda kv: -kv[0])
    seen: set[str] = set()
    out: list[str] = []
    for _, url in candidates:
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= max_n:
            break
    return out


# ---------------------------------------------------------------------------
# Fast serialization — the hot path for the FIFO race
# ---------------------------------------------------------------------------

def _round_logprobs(lps: list[float]) -> list[float]:
    """Round each entry to ``_LOGPROB_DECIMAL_PLACES`` digits."""
    p = _LOGPROB_DECIMAL_PLACES
    return [round(x, p) for x in lps]


def _serialize_request_fast(request: BatchSubmissionRequest) -> bytes:
    """Build the on-wire JSON payload as bytes.

    Hot path: bypasses ``model_dump`` (which traverses every nested model)
    by reading already-Python-native fields directly. Rounds
    ``token_logprobs`` to ``_LOGPROB_DECIMAL_PLACES`` digits — cuts wire
    size by ~3x with no validity impact (see module docstring).
    """
    rollouts_out: list[dict] = []
    for r in request.rollouts:
        # ``commit`` is ``dict[str, Any]`` in the pydantic model — already
        # a plain dict, not a model. Shallow-copy only the nodes we mutate.
        commit_in = r.commit
        commit_out = dict(commit_in)
        rollout_meta = commit_in.get("rollout")
        if isinstance(rollout_meta, dict):
            meta_out = dict(rollout_meta)
            lps = meta_out.get("token_logprobs")
            if lps:
                meta_out["token_logprobs"] = _round_logprobs(lps)
            commit_out["rollout"] = meta_out
        rollouts_out.append({
            "tokens": r.tokens,
            "reward": r.reward,
            "commit": commit_out,
        })

    payload = {
        "miner_hotkey": request.miner_hotkey,
        "prompt_idx": request.prompt_idx,
        "window_start": request.window_start,
        "merkle_root": request.merkle_root,
        "rollouts": rollouts_out,
        "checkpoint_hash": request.checkpoint_hash,
    }

    if _HAS_ORJSON:
        import orjson
        return orjson.dumps(payload)
    # stdlib fallback — slower but correct
    import json
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP plumbing — single-URL with retries, multi-URL without
# ---------------------------------------------------------------------------

def _safe_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
        return str(body)[:200]
    except Exception:
        return resp.text[:200]


def _status_to_reject(status_code: int) -> RejectReason:
    """Map HTTP error codes to the canonical reject sentinel."""
    if status_code == 503:
        return RejectReason.WINDOW_NOT_ACTIVE
    if status_code == 409:
        return RejectReason.WINDOW_MISMATCH
    return RejectReason.BAD_PROMPT_IDX


async def _post_bytes_once(
    full_url: str,
    body: bytes,
    *,
    client: httpx.AsyncClient,
    timeout: float,
) -> BatchSubmissionResponse:
    """Single-shot POST, no retries. Network errors raise.

    Used by the multi-broadcast path where each URL is independent and
    retries on one dead validator MUST NOT delay other validators'
    success signal — the engine considers ``accepted=True`` as soon as
    one URL returns SUBMITTED.
    """
    headers = {
        "Content-Type": "application/json",
        # Prefer the well-pooled connection; httpx already does this but
        # being explicit makes the intent obvious in tcpdumps.
        "Connection": "keep-alive",
    }
    resp = await client.post(
        full_url, content=body, headers=headers, timeout=timeout,
    )
    if resp.status_code == 503:
        return BatchSubmissionResponse(
            accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE,
        )
    if 400 <= resp.status_code < 500:
        return BatchSubmissionResponse(
            accepted=False, reason=_status_to_reject(resp.status_code),
        )
    if resp.status_code >= 500:
        raise SubmissionError(f"HTTP {resp.status_code}: {_safe_detail(resp)}")
    return BatchSubmissionResponse.model_validate(resp.json())


async def _post_bytes_with_retry(
    full_url: str,
    body: bytes,
    *,
    client: httpx.AsyncClient,
    timeout: float,
) -> BatchSubmissionResponse:
    """Retrying POST for the single-URL ``submit_batch_v2`` path."""
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    last_exc: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        try:
            resp = await client.post(
                full_url, content=body, headers=headers, timeout=timeout,
            )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            last_exc = e
            logger.warning(
                "submit attempt %d to %s failed: %r (type=%s)",
                attempt, full_url, e, type(e).__name__,
            )
            if attempt < len(_RETRY_DELAYS):
                await asyncio.sleep(delay)
            continue
        if resp.status_code == 503:
            return BatchSubmissionResponse(
                accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE,
            )
        if 400 <= resp.status_code < 500:
            return BatchSubmissionResponse(
                accepted=False, reason=_status_to_reject(resp.status_code),
            )
        if resp.status_code >= 500:
            last_exc = SubmissionError(f"HTTP {resp.status_code}")
            if attempt < len(_RETRY_DELAYS):
                await asyncio.sleep(delay)
            continue
        return BatchSubmissionResponse.model_validate(resp.json())
    raise SubmissionError(f"all retries failed: {last_exc}")


# ---------------------------------------------------------------------------
# Public submit / state API
# ---------------------------------------------------------------------------

async def submit_batch_v2(
    url: str,
    request: BatchSubmissionRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> BatchSubmissionResponse:
    """POST a v2 batch submission to a single validator.

    Uses the fast serializer (orjson + rounded logprobs) and retains the
    v1/v3 retry ladder so single-validator deployments don't regress on
    transient network errors.
    """
    body = _serialize_request_fast(request)
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        return await _post_bytes_with_retry(
            f"{url}/submit", body, client=cli, timeout=timeout,
        )
    finally:
        if own_client:
            await cli.aclose()


@dataclasses.dataclass
class MultiSubmitResult:
    """Aggregate result of a parallel multi-validator broadcast.

    ``accepted``: any URL returned ``accepted=True``.
    ``best_reason``: one canonical reason for log-line summarisation
        (prefers SUBMITTED > ACCEPTED > first reject > WINDOW_NOT_ACTIVE).
    ``per_url``: per-validator breakdown for diagnostics — keyed by URL,
        each value is ``(response_or_none, http_ms, exception_or_none)``.
        A None response with an exception means the request never
        completed for that URL.
    """

    accepted: bool
    best_reason: RejectReason
    per_url: dict[
        str,
        tuple[BatchSubmissionResponse | None, float, BaseException | None],
    ]


async def submit_batch_v2_multi(
    urls: list[str],
    request: BatchSubmissionRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> MultiSubmitResult:
    """Broadcast a v2 batch submission to every URL in parallel.

    Serializes the body ONCE and reuses the same bytes across all URLs
    (saves N-1 serialization passes — the dominant CPU cost for big
    rollout groups).

    Skips retries by design: a dead validator's exponential backoff
    would block ``MultiSubmitResult`` resolution past the window roll
    on the live validators. The engine's outer loop handles transient
    failures by simply submitting on the next OPEN.
    """
    if not urls:
        return MultiSubmitResult(
            accepted=False,
            best_reason=RejectReason.WINDOW_NOT_ACTIVE,
            per_url={},
        )

    body = _serialize_request_fast(request)
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)

    async def _post_one(
        u: str,
    ) -> tuple[
        str,
        BatchSubmissionResponse | None,
        float,
        BaseException | None,
    ]:
        t = time.monotonic()
        try:
            resp = await _post_bytes_once(
                f"{u}/submit", body, client=cli, timeout=timeout,
            )
            return (u, resp, (time.monotonic() - t) * 1000.0, None)
        except (httpx.RequestError, httpx.TimeoutException, SubmissionError) as e:
            return (u, None, (time.monotonic() - t) * 1000.0, e)

    try:
        results = await asyncio.gather(*(_post_one(u) for u in urls))
    finally:
        if own_client:
            await cli.aclose()

    per_url: dict[
        str,
        tuple[BatchSubmissionResponse | None, float, BaseException | None],
    ] = {u: (resp, ms, exc) for (u, resp, ms, exc) in results}

    # Aggregate verdict — accept if any URL accepted.
    accepted_any = any(
        r is not None and r.accepted for r, _, _ in per_url.values()
    )

    # Pick a single ``best_reason`` for the log line. Production validator
    # returns SUBMITTED (queued); ACCEPTED is the TestClient sync sentinel;
    # any other reject is a hard failure on that URL. If every URL
    # network-errored, fall back to WINDOW_NOT_ACTIVE so the engine logs
    # something meaningful while ``record_network_error`` fires separately.
    submitted_seen = False
    accepted_seen = False
    first_reject: RejectReason | None = None
    for resp, _, _ in per_url.values():
        if resp is None:
            continue
        r = resp.reason
        if r == RejectReason.SUBMITTED:
            submitted_seen = True
        elif r == RejectReason.ACCEPTED:
            accepted_seen = True
        elif first_reject is None:
            first_reject = r

    if submitted_seen:
        best_reason: RejectReason = RejectReason.SUBMITTED
    elif accepted_seen:
        best_reason = RejectReason.ACCEPTED
    elif first_reject is not None:
        best_reason = first_reject
    else:
        best_reason = RejectReason.WINDOW_NOT_ACTIVE

    return MultiSubmitResult(
        accepted=accepted_any,
        best_reason=best_reason,
        per_url=per_url,
    )


async def get_window_state_v2(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> GrpoBatchState:
    """GET the validator's current v2 GrpoBatchState. Retries on transient errors."""
    last_exc: Exception | None = None
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                resp = await cli.get(f"{url}/state", timeout=timeout)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            if resp.status_code == 503:
                raise SubmissionError(f"no active window at {url}")
            if resp.status_code == 404:
                raise SubmissionError(f"endpoint not found: {url}/state")
            if 400 <= resp.status_code < 500:
                raise SubmissionError(
                    f"HTTP {resp.status_code}: {_safe_detail(resp)}"
                )
            if resp.status_code >= 500:
                last_exc = SubmissionError(f"HTTP {resp.status_code}")
                if attempt < len(_RETRY_DELAYS):
                    await asyncio.sleep(delay)
                continue
            return GrpoBatchState.model_validate(resp.json())
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()


# ---------------------------------------------------------------------------
# Connection pre-warming
# ---------------------------------------------------------------------------

async def prewarm_connections(
    urls: list[str],
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 5.0,
) -> dict[str, bool]:
    """Fire a /health GET against every URL to prime TLS/TCP pools.

    Called once at miner startup (and ideally periodically during long
    CLOSED phases). Eliminates the connect+TLS round-trip from the
    first /submit of each window — the dominant cost for cold TCP.

    Returns ``{url: True}`` for any URL that responded with 200 within
    ``timeout``; ``False`` otherwise.
    """
    if not urls:
        return {}
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)

    async def _ping(u: str) -> tuple[str, bool]:
        try:
            resp = await cli.get(f"{u}/health", timeout=timeout)
            return (u, resp.status_code == 200)
        except Exception:
            return (u, False)

    try:
        results = await asyncio.gather(*(_ping(u) for u in urls))
    finally:
        if own_client:
            await cli.aclose()
    return {u: ok for u, ok in results}
