#!/usr/bin/env python3
"""Accumulate the rank-1 miner's accepted rollouts over time + infer patterns.

Polls reliqua.ai for the top miner, dedupes accepted (window, prompt_idx) into a
growing JSONL, and logs a rolling inference summary (sources, topics, sigma,
multi-submit windows, idx reuse, rate). Run in the background; re-arm to keep going.
"""
import json, re, time, urllib.request, os, sys
from collections import Counter

HK = "5F6VZ2roP7ikDQnfzaHUwi54bYL4hmcTBqaPSgzraZ2rMMmy"
STORE = "/root/topminer_accepts.jsonl"
URL = f"https://www.reliqua.ai/api/miners/{HK}"
TOPICS = {
    "trig": r"\\(sin|cos|tan|cot|sec|csc)|periodic|angle|radian",
    "matrix": r"\\begin\{pmatrix|matrix|rotation|eigen|determinant|vector",
    "complex": r"complex|imaginary|e\^\{i|polar form|modulus|argument",
    "conic/coord": r"conic|ellipse|parabola|hyperbola|cylindrical|spherical|polar|coordinate",
    "calculus": r"derivative|integral|\\int|\\lim|differentiate",
    "poly/factor": r"factor|polynomial|roots|expand",
    "radical": r"\\sqrt|radical|rationaliz",
    "sequence": r"sequence|series|\\sum|arithmetic|geometric",
    "logexp": r"\\log|\\ln|logarithm|exponential",
    "prob/pct": r"probab|percent|\\%|ratio",
}


def load_seen():
    seen = {}
    if os.path.exists(STORE):
        for line in open(STORE):
            try:
                r = json.loads(line); seen[(r["window"], r["idx"])] = r
            except Exception:
                pass
    return seen


def fetch():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def summarize(seen):
    rows = list(seen.values())
    idxs = [r["idx"] for r in rows]
    win_counts = Counter(r["window"] for r in rows)
    tc = Counter()
    for r in rows:
        p = r.get("prompt", "")
        m = [t for t, pat in TOPICS.items() if re.search(pat, p, re.I)]
        for t in m: tc[t] += 1
        if not m: tc["other"] += 1
    dup = len(idxs) - len(set(idxs))
    print(f"  total accepts={len(rows)} | distinct prompt_idx={len(set(idxs))} | repeated idx={dup}")
    print(f"  sources={dict(Counter(r.get('src') for r in rows))}")
    print(f"  sigma={dict(Counter(round(r.get('sigma',0),3) for r in rows))}")
    print(f"  topics={dict(tc.most_common())}")
    print(f"  per-window accept counts dist={dict(Counter(win_counts.values()))} (max in one window={max(win_counts.values()) if win_counts else 0})")


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seen = load_seen()
    print(f"start: {len(seen)} accepts already stored")
    try:
        import datasets  # noqa
        from reliquary.environment import load_environment
        env = load_environment("openmathinstruct"); ds = env._dataset
        srcs = ds["problem_source"]; N = len(ds)
    except Exception:
        srcs = None; N = 1
    for it in range(iters):
        try:
            d = fetch()
            new = 0
            for w in d.get("window_detail", []):
                if not w.get("accepted"):
                    continue
                for s in w.get("samples", []):
                    idx = s.get("prompt_idx"); win = w["window"]
                    if not isinstance(idx, int) or (win, idx) in seen:
                        continue
                    rec = {"window": win, "idx": idx, "sigma": s.get("sigma"),
                           "gt": str(s.get("ground_truth", ""))[:40],
                           "len": s.get("completion_length"),
                           "prompt": s.get("prompt", "")[:120],
                           "src": (srcs[idx % N] if srcs else None)}
                    seen[(win, idx)] = rec
                    with open(STORE, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    new += 1
            cur_win = d.get("current_window", {}).get("window")
            print(f"[{time.strftime('%H:%M:%S')}] iter {it}: +{new} new accepts (cur_window={cur_win})")
            summarize(seen)
        except Exception as e:
            print(f"[iter {it}] fetch error: {e}")
        if it < iters - 1:
            time.sleep(interval)
    print("=== watch run complete ===")


if __name__ == "__main__":
    main()
