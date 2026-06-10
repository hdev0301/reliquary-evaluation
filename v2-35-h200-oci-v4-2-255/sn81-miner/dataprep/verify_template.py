#!/usr/bin/env python3
"""Verify the rank-1 miner's reward-vector "even-slot template" across the FULL
archived history (not just a hand-sampled window range).

For each archived window it pulls /api/r2/window/<w>, extracts every batch entry
belonging to the target hotkey, recovers the 8-rollout reward vector, and tests
whether the correct answers are placed in a FIXED deterministic order rather than
the random order honest sampling would produce.

Decisive test
-------------
Under honest i.i.d. sampling at T=0.9 the *positions* of the k correct rollouts
are exchangeable: any size-k subset of the 8 slots is equally likely, so
P(a specific arrangement) = 1 / C(8, k). We test each group against the
empirically-inferred placement order CANON = [0,2,4,6,7,5,3,1] (even slots
ascending, then odd slots descending): a group "matches the template" iff the
set of correct positions equals the first-k entries of CANON. We report the
match rate and the combined log10 probability of that many matches under the
honest null.
"""
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

HK = "5F6VZ2roP7ikDQnfzaHUwi54bYL4hmcTBqaPSgzraZ2rMMmy"
BASE = "https://www.reliqua.ai/api"
STORE = "/root/topminer_vectors.jsonl"
CANON = [0, 2, 4, 6, 7, 5, 3, 1]  # inferred correct-answer placement order
# Alternative orders we also score, to stay honest about which template fits best.
ALT_ORDERS = {
    "even_then_odd_desc[0,2,4,6,7,5,3,1]": [0, 2, 4, 6, 7, 5, 3, 1],
    "even_then_odd_asc[0,2,4,6,1,3,5,7]": [0, 2, 4, 6, 1, 3, 5, 7],
    "ascending[0,1,2,3,4,5,6,7]": [0, 1, 2, 3, 4, 5, 6, 7],
}


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def current_window():
    d = fetch(f"{BASE}/miners/{HK}")
    cw = d.get("current_window", {}).get("window")
    hist = [h["window"] for h in d.get("history", []) if isinstance(h.get("window"), int)]
    return cw, hist


def reward_vector(entry):
    rolls = entry.get("rollouts") or []
    rews = []
    for r in rolls:
        v = r.get("reward")
        if v is None:
            return None
        rews.append(1 if float(v) >= 0.5 else 0)
    return rews if len(rews) == 8 else None


def window_rows(w):
    """Return [(reward_vec, sigma, lengths, eos_flags)] for the target hotkey."""
    try:
        data = fetch(f"{BASE}/r2/window/{w}")["data"]
    except Exception as e:
        return ("err", w, str(e)[:60])
    out = []
    for entry in (data.get("batch") or []):
        if entry.get("hotkey") != HK:
            continue
        rv = reward_vector(entry)
        if rv is None:
            continue
        rolls = entry.get("rollouts") or []
        lens = [r.get("completion_length") for r in rolls]
        eos = [r.get("eos_terminated") for r in rolls]
        out.append({
            "window": w,
            "prompt_idx": entry.get("prompt_idx"),
            "rewards": rv,
            "k": sum(rv),
            "sigma": entry.get("sigma"),
            "lengths": lens,
            "eos": eos,
        })
    return ("ok", w, out)


def matches_order(rv, order):
    k = sum(rv)
    correct_pos = {i for i, v in enumerate(rv) if v == 1}
    return correct_pos == set(order[:k])


def is_monotonic(rv):
    return rv == sorted(rv) or rv == sorted(rv, reverse=True)


def main():
    n_back = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    cw, hist = current_window()
    hi = cw if isinstance(cw, int) else (max(hist) if hist else 0)
    lo = hi - n_back
    windows = list(range(hi, lo, -1))
    print(f"target hotkey : {HK}")
    print(f"current window: {cw}; scanning {len(windows)} windows [{lo+1}..{hi}]")

    rows = []
    errs = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(window_rows, w): w for w in windows}
        for i, fut in enumerate(as_completed(futs)):
            status, w, payload = fut.result()
            if status == "err":
                errs += 1
            else:
                rows.extend(payload)
            if (i + 1) % 25 == 0:
                print(f"  ...scanned {i+1}/{len(windows)} (groups so far={len(rows)}, errs={errs})")

    # persist raw vectors for re-analysis
    with open(STORE, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    print(f"\n=== RESULTS: {n} full 8-rollout groups for target across {len(windows)} windows "
          f"({errs} windows unavailable) ===")
    if not n:
        print("no groups recovered (archive may be rate-limited; retry with fewer windows).")
        return

    # k-distribution
    kc = Counter(r["k"] for r in rows)
    print(f"\nk (correct-count) distribution: {dict(sorted(kc.items()))}")
    print(f"  mean k = {sum(r['k'] for r in rows)/n:.2f}/8 ; "
          f"in-zone-only (k in 2..6) = {sum(1 for r in rows if 2<=r['k']<=6)}/{n}")

    # per-slot correct rate
    slot_rate = [sum(r["rewards"][s] for r in rows)/n for s in range(8)]
    print("\nper-slot correct rate (honest sampling -> all ~= mean k/8):")
    print("  slot:  " + "  ".join(f"{s}" for s in range(8)))
    print("  rate:  " + "  ".join(f"{x:.2f}" for x in slot_rate))

    # monotonic fraction
    mono = sum(1 for r in rows if is_monotonic(r["rewards"]))
    print(f"\nmonotonic (ascending/descending) vectors: {mono}/{n} = {mono/n:.0%}")

    # template match rates for each candidate order
    print("\ntemplate match rate (correct positions == first-k of order):")
    best = None
    for name, order in ALT_ORDERS.items():
        m = sum(1 for r in rows if matches_order(r["rewards"], order))
        print(f"  {name:38s}: {m}/{n} = {m/n:.0%}")
        if best is None or m > best[1]:
            best = (name, m, order)

    # combined null probability for the best-fitting template
    name, m, order = best
    log10p = 0.0
    for r in rows:
        if matches_order(r["rewards"], order):
            log10p += -math.log10(math.comb(8, r["k"]))
    print(f"\nbest-fit template: {name}  ({m}/{n} groups match)")
    print(f"combined P(all {m} matches | honest random placement) = 10^{log10p:.1f}")
    if log10p < -20:
        print("  -> astronomically improbable under honest sampling: reward vectors are")
        print("     DELIBERATELY ARRANGED (curated selection + fixed placement), confirming")
        print("     reward-oracle candidate selection rather than natural 8-sample submission.")

    # sigma sanity (should mirror k via sqrt(p(1-p)))
    sig = Counter(round(r["sigma"], 4) for r in rows if r.get("sigma") is not None)
    if sig:
        print(f"\nsigma modes: {dict(sig.most_common(6))}")

    # length / eos sanity (rule out cap-forcing / padding)
    lens = [l for r in rows for l in (r["lengths"] or []) if isinstance(l, int)]
    if lens:
        cap = sum(1 for l in lens if l >= 8192)
        print(f"\nrollout completion_length: n={len(lens)} min={min(lens)} max={max(lens)} "
              f"mean={sum(lens)//len(lens)} at_8192_cap={cap}")
    eos_flags = [e for r in rows for e in (r["eos"] or []) if e is not None]
    if eos_flags:
        print(f"eos_terminated: {dict(Counter(eos_flags))}")

    print(f"\nraw vectors written to {STORE}")


if __name__ == "__main__":
    main()
