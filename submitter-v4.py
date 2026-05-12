"""HTTP client used by miners to push GRPO submissions to the validator (v4).

DEPLOYMENT: this file is the v4 patch to ``reliquary/miner/submitter.py`` —
copy it over the upstream file on the miner box:

    cp submitter-v4.py /root/reliquary/reliquary/miner/submitter.py

Backward-compatible: re-exports every public symbol the v3 engine /
launcher imports (``SubmissionError``, ``NoValidatorFoundError``,
``discover_validator_url``, ``get_window_state_v2``, ``submit_batch_v2``)
so unmodified callers keep working.

Time-efficiency notes (v4.2+)
=============================

- ``submit_batch_v2_multi`` serializes the payload ONCE before fan-out
  to N validators. v3-style ``submit_batch_v2`` called per-URL would
  re-run ``pydantic.model_dump(mode="json")`` for every URL — a
  ~0.5-2 s pydantic walk on a 5-30 MB payload, multiplied by N. Now
  it's one walk + one orjson/json encode → bytes.
- orjson is used opportunistically when installed (``pip install orjson``).
  3-5× faster than stdlib json on large numeric arrays — matters for
  the ~10 MB token_logprobs / commitments arrays in our submission.
- ``prewarm_connections`` performs a cheap GET /state against each URL
  during engine startup so the first real /submit doesn't pay the
  TCP handshake + TLS setup latency (~50-300 ms per validator,
  multiplied by N validators).

What v4 fixes
=============

The v3 submitter has three issues that compound into the
``window_mismatch`` / ``window_not_active`` rejection storm reported
under heavy validator-network load:

1. **Single-validator submission.** ``discover_validator_url`` returns
   the FIRST permitted axon and the engine submits only there. The
   network's other validators never see our submissions → they score
   us 0 → our final on-chain weight (averaged across all validators
   via consensus) collapses to the single validator's view divided by
   N. Top miners broadcast to ALL permitted validators in parallel,
   getting ~N× the weight signal per window.

   v4 adds:
     - ``discover_validator_urls(metagraph, max_n=N)`` — returns up to
       N distinct permitted axon URLs (ordered by uid).
     - ``submit_batch_v2_multi(urls, req)`` — fans out a single
       request to all URLs in parallel. Aggregates per-URL
       (accepted, reason, http_ms) and reports back the worst-case
       reject reason if NONE accepted (so the engine can still
       record actionable telemetry).

2. **Retries waste the window.** v3 retries 3× with 1s + 2s + 4s
   backoff on network errors. A slow validator (90 s upload + 60 s
   timeout) plus 3 retries can spend > 4 minutes waiting for a
   single submit — by which point the OPEN window has rolled at
   least twice and every retry POSTs into a stale window
   (``window_mismatch``).

   v4 reduces to **single-attempt** for /submit (the request body
   is large — 10-30 MB — so retrying is a network re-upload
   penalty, and if the first attempt timed out, the second will
   too). /state polls still use one quick retry because the body
   is < 1 KB and the round-trip is < 1 s.

3. **Default timeout too generous.** ``_DEFAULT_TIMEOUT = 60 s`` lets
   the request hang past the typical 60-90 s OPEN window. By the
   time the response comes back, the window has rolled.

   v4 reduces to 30 s default. The validator's /submit endpoint
   enqueues the request and returns immediately with
   ``SUBMITTED`` once the body is fully uploaded; 30 s is plenty
   for the body upload itself even on a 1 Mbps uplink (a 20 MB
   payload uploads in ~160 s — but if it can't upload in 30 s
   the validator's batcher will reject us anyway, so failing
   fast frees the GPU for the next attempt).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from reliquary.constants import VALIDATOR_HTTP_PORT
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    GrpoBatchState,
    RejectReason,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON encoder — orjson opportunistically, stdlib json as fallback.
# ---------------------------------------------------------------------------
#
# Profile on a 5-MB BatchSubmissionRequest payload (8 rollouts × ~7k
# completion tokens, per-token GRAIL commitments + logprobs):
#   stdlib json.dumps + utf-8 encode :  ~250-500 ms
#   orjson.dumps                     :  ~50-100 ms (5× faster)
#
# orjson also has stricter input typing (it won't accept Python
# ``inf`` / ``nan``) — pydantic ``model_dump(mode="json")`` already
# coerces those to None for JSON safety, so we're fine.
try:
    import orjson as _orjson  # type: ignore
    _HAS_ORJSON = True
except ImportError:
    _orjson = None  # type: ignore[assignment]
    _HAS_ORJSON = False
import json as _json


def _encode_json_bytes(obj: Any) -> bytes:
    """Encode a JSON-serializable dict to bytes via the fastest available
    encoder. Returns UTF-8 bytes ready to ship over the wire.

    Compact form (no whitespace) — saves a few percent of bytes on the
    multi-megabyte payload, which compounds over the OPEN window.
    """
    if _HAS_ORJSON and _orjson is not None:
        return _orjson.dumps(obj)
    return _json.dumps(obj, separators=(",", ":")).encode("utf-8")


def _serialize_request_to_bytes(request: BatchSubmissionRequest) -> bytes:
    """Walk pydantic → dict → JSON bytes EXACTLY ONCE.

    Called by ``submit_batch_v2_multi`` so the cost of
    ``pydantic.model_dump(mode="json")`` (which walks every nested
    rollout/commit and coerces types) is paid once for the whole
    broadcast — not N times, once per validator URL.
    """
    payload_dict = request.model_dump(mode="json")
    return _encode_json_bytes(payload_dict)


# Common headers for the pre-encoded fast path. Setting Content-Type
# explicitly bypasses httpx's slow `json=` re-encoding path.
_JSON_HEADERS = {"content-type": "application/json"}

# ---------------------------------------------------------------------------
# Retry / timeout knobs — see module docstring for rationale.
# ---------------------------------------------------------------------------

# /submit retry policy: SINGLE attempt. Retrying after a 30 s upload
# fail just re-uploads the same 10-30 MB body into a window that's
# almost certainly already rolled. Empty tuple disables retry.
_SUBMIT_RETRY_DELAYS: tuple[float, ...] = ()

# /state retry policy: one quick retry. Body is tiny, round-trip is
# < 1 s, and a single network glitch shouldn't drop us out of the
# poll loop.
_STATE_RETRY_DELAYS: tuple[float, ...] = (0.5,)

# Per-request HTTP timeout. 30 s is enough for the body upload on
# most uplinks; failing fast lets the engine move on to the next
# attempt rather than burning the OPEN window on a doomed POST.
_DEFAULT_TIMEOUT = 30.0

# How many validators ``discover_validator_urls`` returns by default.
# 5 is a balance between coverage (more validators = more weight EMA
# contributions) and uplink cost (every additional validator adds one
# full payload upload per submit). Operator can override with the
# ``max_n`` kwarg.
_DEFAULT_MAX_VALIDATORS = 5


class NoValidatorFoundError(RuntimeError):
    """No metagraph entry advertises a usable validator endpoint."""


class SubmissionError(RuntimeError):
    """All submission retries exhausted."""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_validator_urls(
    metagraph: Any,
    *,
    port: int = VALIDATOR_HTTP_PORT,
    max_n: int = _DEFAULT_MAX_VALIDATORS,
) -> list[str]:
    """Return up to ``max_n`` HTTP URLs of permitted validators.

    Ordered by uid, deduplicated by (ip, port). Returns an empty list
    when no validator is reachable — caller decides whether to raise.

    Multi-validator broadcast is what closes the weight gap to top
    miners: every additional validator that sees our submission adds
    one EMA contribution to our final on-chain weight. v3 used to
    pick just the first; v4 uses all permitted ones.
    """
    permits = getattr(metagraph, "validator_permit", None)
    axons = getattr(metagraph, "axons", None)
    if permits is None or axons is None:
        return []
    seen: set[tuple[str, int]] = set()
    urls: list[str] = []
    for uid, (permit, axon) in enumerate(zip(permits, axons)):
        if not permit:
            continue
        ip = getattr(axon, "ip", None)
        if not ip or ip in ("0.0.0.0", ""):
            continue
        axon_port = getattr(axon, "port", None) or port
        key = (str(ip), int(axon_port))
        if key in seen:
            continue
        seen.add(key)
        urls.append(f"http://{ip}:{axon_port}")
        if len(urls) >= max_n:
            break
    return urls


def discover_validator_url(metagraph: Any, port: int = VALIDATOR_HTTP_PORT) -> str:
    """Backward-compatible single-URL discovery.

    Returns the first URL from ``discover_validator_urls``. Raises
    ``NoValidatorFoundError`` when the list is empty (matches v3
    contract). Existing v3-era callers (``main.py``, the launcher's
    one-shot probe) work unchanged.
    """
    urls = discover_validator_urls(metagraph, port=port, max_n=1)
    if not urls:
        raise NoValidatorFoundError("no validator with permit and routable axon")
    return urls[0]


# ---------------------------------------------------------------------------
# Low-level HTTP helpers — retry-aware, structured-result on known rejects.
# ---------------------------------------------------------------------------

def _safe_detail(resp: httpx.Response) -> str:
    """Best-effort error-detail extraction. Never raises."""
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
        return str(body)[:200]
    except Exception:
        return resp.text[:200]


async def _post_with_retry(
    full_url: str,
    json_payload: dict | None,
    response_model: type,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
    retry_delays: tuple[float, ...] = _SUBMIT_RETRY_DELAYS,
    body_bytes: bytes | None = None,
) -> Any:
    """POST with optional retries.

    Either ``json_payload`` (dict, lets httpx encode) or ``body_bytes``
    (pre-encoded JSON bytes — used by the multi-validator broadcast
    fast path to skip per-URL re-serialization) must be supplied.

    A 503 (no active window) and 4xx (deterministic reject) short-circuit
    to a structured ``BatchSubmissionResponse`` so the caller can record
    the reject reason without a second network round-trip.
    """
    last_exc: Exception | None = None
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    # Attempt count = 1 (the first try) + len(retry_delays). Empty
    # ``retry_delays`` ⇒ single attempt, which is the v4 default for
    # /submit (see module docstring).
    attempts = 1 + len(retry_delays)
    try:
        for attempt in range(1, attempts + 1):
            try:
                if body_bytes is not None:
                    # Fast path: pre-encoded bytes. Skip httpx's
                    # internal json.dumps so the per-URL submit doesn't
                    # re-walk the payload.
                    resp = await cli.post(
                        full_url, content=body_bytes,
                        headers=_JSON_HEADERS, timeout=timeout,
                    )
                else:
                    resp = await cli.post(
                        full_url, json=json_payload, timeout=timeout,
                    )
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(
                    "submit attempt %d/%d to %s failed: %r (type=%s)",
                    attempt, attempts, full_url, e, type(e).__name__,
                )
                if attempt <= len(retry_delays):
                    await asyncio.sleep(retry_delays[attempt - 1])
                continue
            # 503 "no active window" is informational for BatchSubmissionResponse —
            # don't retry, surface as a structured reject.
            if resp.status_code == 503 and response_model is BatchSubmissionResponse:
                return BatchSubmissionResponse(
                    accepted=False, reason=RejectReason.WINDOW_NOT_ACTIVE
                )
            # 4xx means the request is malformed or the validator rejected it
            # for a deterministic reason — retrying is pointless. Parse and return.
            if 400 <= resp.status_code < 500:
                detail = _safe_detail(resp)
                if response_model is BatchSubmissionResponse:
                    if resp.status_code == 409:
                        reason = RejectReason.WINDOW_MISMATCH
                    else:
                        reason = RejectReason.BAD_PROMPT_IDX
                    return BatchSubmissionResponse(accepted=False, reason=reason)
                raise SubmissionError(f"HTTP {resp.status_code}: {detail}")
            if resp.status_code >= 500:
                last_exc = SubmissionError(f"HTTP {resp.status_code}")
                if attempt <= len(retry_delays):
                    await asyncio.sleep(retry_delays[attempt - 1])
                continue
            return response_model.model_validate(resp.json())
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()


async def _get_with_retry(
    full_url: str,
    response_model: type,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
    retry_delays: tuple[float, ...] = _STATE_RETRY_DELAYS,
) -> Any:
    last_exc: Exception | None = None
    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    attempts = 1 + len(retry_delays)
    try:
        for attempt in range(1, attempts + 1):
            try:
                resp = await cli.get(full_url, timeout=timeout)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt <= len(retry_delays):
                    await asyncio.sleep(retry_delays[attempt - 1])
                continue
            if resp.status_code == 503:
                raise SubmissionError(f"no active window at {full_url}")
            if resp.status_code == 404:
                raise SubmissionError(f"endpoint not found: {full_url}")
            if 400 <= resp.status_code < 500:
                raise SubmissionError(
                    f"HTTP {resp.status_code}: {_safe_detail(resp)}"
                )
            if resp.status_code >= 500:
                last_exc = SubmissionError(f"HTTP {resp.status_code}")
                if attempt <= len(retry_delays):
                    await asyncio.sleep(retry_delays[attempt - 1])
                continue
            return response_model.model_validate(resp.json())
        raise SubmissionError(f"all retries failed: {last_exc}")
    finally:
        if own_client:
            await cli.aclose()


# ---------------------------------------------------------------------------
# Single-validator API (backward-compatible with v3)
# ---------------------------------------------------------------------------

async def submit_batch_v2(
    url: str,
    request: BatchSubmissionRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> BatchSubmissionResponse:
    """POST a v2 batch submission. Single attempt by default (v4).

    For the single-URL backward-compatible path we still pre-encode to
    bytes (with orjson when installed) — this gives the same per-submit
    speedup the multi-URL path enjoys, with no behaviour change.
    """
    body = _serialize_request_to_bytes(request)
    return await _post_with_retry(
        f"{url}/submit", None, BatchSubmissionResponse,
        client=client, timeout=timeout, body_bytes=body,
    )


async def _submit_bytes(
    url: str,
    body: bytes,
    *,
    client: httpx.AsyncClient | None,
    timeout: float,
) -> BatchSubmissionResponse:
    """Internal: POST already-serialized JSON bytes to ``url/submit``.

    Used by ``submit_batch_v2_multi`` to share one encoded payload
    across N parallel fan-out requests.
    """
    return await _post_with_retry(
        f"{url}/submit", None, BatchSubmissionResponse,
        client=client, timeout=timeout, body_bytes=body,
    )


async def get_window_state_v2(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> GrpoBatchState:
    """GET the validator's current v2 GrpoBatchState."""
    return await _get_with_retry(
        f"{url}/state", GrpoBatchState,
        client=client, timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Multi-validator broadcast API (v4)
# ---------------------------------------------------------------------------

class MultiSubmitResult:
    """Aggregated result of broadcasting one request to multiple validators.

    Exposes:
      - ``accepted``: True if AT LEAST ONE validator accepted (either
        ``ACCEPTED`` post-verify or ``SUBMITTED`` provisional sentinel).
      - ``per_url``: dict[url, (response, http_ms, exc)] — full per-URL
        breakdown so the engine can record per-validator telemetry.
      - ``best_reason``: a representative ``RejectReason`` to feed the
        single-channel logging path (the worst-case accept if any
        accepted, otherwise the first reject reason seen).
      - ``max_http_ms`` / ``min_http_ms``: span of the fan-out latency.
    """

    __slots__ = ("per_url", "accepted", "best_reason", "max_http_ms", "min_http_ms")

    def __init__(
        self,
        per_url: dict[str, tuple[BatchSubmissionResponse | None, float, Exception | None]],
    ) -> None:
        self.per_url = per_url
        any_ok = False
        first_reject: RejectReason | None = None
        accepted_reason: RejectReason | None = None
        http_times = []
        for url, (resp, http_ms, exc) in per_url.items():
            if http_ms > 0:
                http_times.append(http_ms)
            if resp is None:
                continue
            if resp.accepted:
                any_ok = True
                # ``SUBMITTED`` (queued) is the common case under uvicorn;
                # ``ACCEPTED`` only on the inline TestClient path. Prefer
                # whichever lands first.
                accepted_reason = resp.reason
            else:
                if first_reject is None:
                    first_reject = resp.reason
        self.accepted = any_ok
        self.best_reason = (
            accepted_reason
            or first_reject
            or RejectReason.WINDOW_NOT_ACTIVE
        )
        self.max_http_ms = max(http_times) if http_times else 0.0
        self.min_http_ms = min(http_times) if http_times else 0.0


async def submit_batch_v2_multi(
    urls: list[str],
    request: BatchSubmissionRequest,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> MultiSubmitResult:
    """Fan out ONE submission to every URL in parallel.

    Each validator records the submission independently — accepting at
    one validator does NOT propagate to the others. So broadcasting
    multiplies our weight EMA contribution by the number of validators
    that accept us.

    Time-efficiency: serializes the payload exactly ONCE (pydantic
    walk + JSON encode), then ships the resulting bytes to every URL
    in parallel. With N=5 validators this saves 2-10 s of CPU work
    compared to per-URL ``model_dump`` (a measurable chunk of the
    OPEN-window budget for our >5 MB payloads).

    Returns a ``MultiSubmitResult`` aggregating per-URL responses. The
    engine treats ``result.accepted`` as the operative success flag
    (any acceptor counts) and uses ``result.best_reason`` for the
    single-channel SUB log line.

    Network-level failures and timeouts are captured per-URL — one
    slow validator does not block the others.
    """
    if not urls:
        return MultiSubmitResult({})

    # One-shot serialization. Subsequent per-URL posts ship the same
    # bytes without re-walking pydantic.
    body = _serialize_request_to_bytes(request)

    async def _one(_url: str) -> tuple[str, tuple[BatchSubmissionResponse | None, float, Exception | None]]:
        import time as _time
        t0 = _time.monotonic()
        try:
            resp = await _submit_bytes(
                _url, body, client=client, timeout=timeout,
            )
            return _url, (resp, (_time.monotonic() - t0) * 1000.0, None)
        except (SubmissionError, httpx.RequestError, httpx.TimeoutException) as exc:
            return _url, (None, (_time.monotonic() - t0) * 1000.0, exc)
        except Exception as exc:  # pragma: no cover — defensive
            return _url, (None, (_time.monotonic() - t0) * 1000.0, exc)

    pairs = await asyncio.gather(*[_one(u) for u in urls], return_exceptions=False)
    per_url = dict(pairs)
    return MultiSubmitResult(per_url)


# ---------------------------------------------------------------------------
# Connection pre-warming (v4.2+)
# ---------------------------------------------------------------------------

async def prewarm_connections(
    urls: list[str],
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 5.0,
) -> dict[str, bool]:
    """Pre-warm TCP/TLS to each validator URL by issuing a cheap GET.

    Returns a dict mapping each URL to whether the prewarm succeeded.
    Failures are NOT fatal — they're just logged so the operator can
    see which validators are unreachable at startup. The real /submit
    call later will still try the URL.

    Why this matters
    ----------------
    httpx keeps a per-host connection pool, so once a connection is
    established it's reused across subsequent requests. But the FIRST
    request per host pays:
      - DNS resolution           :  ~10-100 ms
      - TCP handshake (3-way)    :  ~RTT (~50-300 ms intercontinental)
      - TLS handshake (2-RTT)    :  ~2× RTT (HTTP/2 over TLS 1.3 is faster
                                    but still ~RTT minimum)
    On a fresh process, the first /submit therefore eats 200 ms - 1 s
    of cold-connection latency PER validator BEFORE the body upload
    even starts — wasted against the OPEN-window budget.

    By probing /state at startup (a few-KB JSON response) we amortize
    those handshakes outside the timing-critical OPEN phase.
    """
    if not urls:
        return {}

    own_client = client is None
    cli = client or httpx.AsyncClient(timeout=timeout)
    try:
        async def _probe(_u: str) -> tuple[str, bool]:
            try:
                resp = await cli.get(f"{_u}/state", timeout=timeout)
                return _u, (200 <= resp.status_code < 600)
            except Exception:
                return _u, False

        pairs = await asyncio.gather(*[_probe(u) for u in urls], return_exceptions=False)
        return dict(pairs)
    finally:
        if own_client:
            await cli.aclose()
