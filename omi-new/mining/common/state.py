"""Validator state + verdict tracking shared by the env miners.

Thin async helpers over ``reliquary.miner.submitter``. The miner keeps one
long-lived ``httpx.AsyncClient`` (warm keep-alive sockets) so the fire path
pays no TCP/TLS handshake on the hot path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StateView:
    """Latest validator state plus derived per-env cooldown sets."""

    state: "object | None" = None                       # GrpoBatchState
    checkpoint_n: int = 0
    checkpoint_repo_id: str | None = None
    checkpoint_revision: str | None = None
    randomness: str = ""
    window_n: int = 0
    cooldown_per_env: dict[str, set[int]] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        from reliquary.protocol.submission import WindowState

        return self.state is not None and self.state.state == WindowState.OPEN


async def refresh_state(url, client, env_names) -> StateView:
    """Fetch /state and each env's cooldown set into a fresh StateView."""
    from reliquary.miner.submitter import get_window_state_v2

    base = await get_window_state_v2(url, client=client)
    view = StateView(
        state=base,
        checkpoint_n=base.checkpoint_n,
        checkpoint_repo_id=base.checkpoint_repo_id,
        checkpoint_revision=base.checkpoint_revision,
        randomness=base.randomness or "",
        window_n=base.window_n,
    )
    for env_name in env_names:
        try:
            env_state = await get_window_state_v2(url, env=env_name, client=client)
            view.cooldown_per_env[env_name] = set(env_state.cooldown_prompts)
        except Exception:
            view.cooldown_per_env[env_name] = set(base.cooldown_prompts)
    return view


async def fetch_verdicts(url, client, hotkey, since_ts: float) -> list[dict]:
    """GET /verdicts/{hotkey}?since=ts → list of verdict dicts (may be empty).

    The real per-submission outcome (accepted / grail_fail / out_of_zone / ...)
    lands here ~seconds after the provisional /submit ack. We feed it back into
    the frontier model so prediction error self-corrects.
    """
    from urllib.parse import quote

    try:
        r = await client.get(
            f"{url}/verdicts/{quote(hotkey, safe='')}",
            params={"since": since_ts},
            timeout=5.0,
        )
        if r.status_code != 200:
            return []
        return r.json().get("verdicts", [])
    except Exception:
        return []
