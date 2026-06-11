#!/usr/bin/env python3
"""harvest_winners.py — snapshot top OMI miner(s)' recent in-zone winners as a FRONTIER SEED.

Fetches reliqua.ai/api/miners/<HK> for each configured hotkey, extracts every accepted
sample's (prompt_idx, prompt, ground_truth, sigma, completion_length) from the rolling
``window_detail``, dedups by idx, and atomically writes them to the winners file that the
miner's frontier predictor seeds from at (re)start (RELIQUARY_WINNERS_PATH, frontier.py).

Why periodic: the API window is rolling-recent (~last 60 windows), so OVERWRITING each run
keeps the seed tracking the CURRENT checkpoint's in-zone signature instead of going stale
after a checkpoint flip. The live miner reads the seed only at startup, so this keeps the
file fresh for the NEXT restart; the online frontier model adapts from live outcomes between.

Lightweight by design: NO dataset load — frontier.py resolves idx -> problem via the miner's
own env at startup, so we only need the idxs + a few fields from the public API.

Env:
  HARVEST_HOTKEYS        comma-separated ss58 hotkeys to harvest+merge
                         (default: the analyzed top OMI miner 5CX7gQ4...)
  RELIQUARY_WINNERS_PATH output file (default: /root/sn81-miner/data/topminer_winners.jsonl)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

_DEFAULT_HK = "5CX7gQ4faun2ttryqnDTs63vA9Kqxt9pDEDzaUBRirX6pZri"
HOTKEYS = [h.strip() for h in os.environ.get("HARVEST_HOTKEYS", _DEFAULT_HK).split(",") if h.strip()]
OUT = os.environ.get("RELIQUARY_WINNERS_PATH", "/root/sn81-miner/data/topminer_winners.jsonl")


def _fetch(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    by_idx: dict[int, dict] = {}   # dedup by prompt_idx, last write wins
    n_win = 0
    ok = 0
    for hk in HOTKEYS:
        try:
            d = _fetch(f"https://www.reliqua.ai/api/miners/{hk}")
        except Exception as e:
            print(f"  fetch {hk[:10]} failed: {e!r}", file=sys.stderr)
            continue
        ok += 1
        wd = d.get("window_detail") or []
        n_win += len(wd)
        for w in wd:
            for s in (w.get("samples") or []):
                i = s.get("prompt_idx")
                if i is None:
                    continue
                by_idx[int(i)] = {
                    "idx": int(i),
                    "prompt": s.get("prompt"),
                    "ground_truth": s.get("ground_truth"),
                    "sigma": s.get("sigma"),
                    "completion_length": s.get("completion_length"),
                }
    if ok == 0:
        print("harvest: all fetches failed; leaving existing seed untouched", file=sys.stderr)
        return 1
    if not by_idx:
        print("harvest: no winning samples returned; leaving existing seed untouched", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        for r in by_idx.values():
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, OUT)  # atomic: the miner never reads a half-written seed
    print(f"harvested {len(by_idx)} distinct winners from {ok}/{len(HOTKEYS)} hotkey(s), "
          f"{n_win} windows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
