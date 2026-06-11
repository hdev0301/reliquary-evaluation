#!/usr/bin/env python3
"""Harvest the in-zone region from the public R2 archive (ALL hotkeys) and test
what distinguishes in-zone winners from a random augmented_math baseline.

Purpose: the miner's pregen is +0/16 because random augmented_math has ~1% in-zone
density and the online frontier predictor can't bootstrap from that. The archive
is a FREE perfect oracle: every accepted group across every miner is a labelled
in-zone example on the CURRENT checkpoint. We harvest those, map prompt_idx ->
features via the local dataset, and quantify which answer-FORMAT features predict
in-zone (the format-ambiguity hypothesis: a converged model still scatters
in/out of the parser on answers with many equivalent string forms).

Outputs:
  /root/wf_data/winners.jsonl   -- {prompt, ground_truth, idx} for frontier seed
  /root/inzone_idxs.json        -- harvested in-zone prompt_idxs (all hotkeys)
  stdout                        -- winners-vs-baseline feature comparison
"""
import json, os, re, sys, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://www.reliqua.ai/api"
HK_TOP = "5F6VZ2roP7ikDQnfzaHUwi54bYL4hmcTBqaPSgzraZ2rMMmy"


def fetch(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def cur_window():
    d = fetch(f"{BASE}/miners/{HK_TOP}")
    return d.get("current_window", {}).get("window")


def harvest_window(w):
    try:
        data = fetch(f"{BASE}/r2/window/{w}")["data"]
    except Exception:
        return None
    rows = []
    for e in (data.get("batch") or []):
        rolls = e.get("rollouts") or []
        rews = [(1 if (r.get("reward") or 0) >= 0.5 else 0) for r in rolls if r.get("reward") is not None]
        if len(rews) != 8:
            continue
        k = sum(rews)
        if not (2 <= k <= 6):
            continue
        rows.append({
            "window": w, "hotkey": e.get("hotkey"), "idx": e.get("prompt_idx"),
            "k": k, "sigma": e.get("sigma"),
            "eos": [r.get("eos_terminated") for r in rolls],
            "lens": [r.get("completion_length") for r in rolls],
        })
    return rows


# ---- answer-format features (the format-ambiguity hypothesis) ----
def ans_features(a: str) -> dict:
    a = str(a)
    return {
        "plain_int": bool(re.fullmatch(r"-?\d+", a.strip())),
        "decimal": bool(re.fullmatch(r"-?\d+\.\d+", a.strip())),
        "fraction": ("\\frac" in a) or bool(re.search(r"\b\d+/\d+\b", a)),
        "radical": "\\sqrt" in a,
        "pi": "\\pi" in a or "pi" in a.lower(),
        "degree": "\\circ" in a or "degree" in a.lower() or "°" in a,
        "tuple_list": ("," in a) or (";" in a) or bool(re.search(r"[()\[\]]", a)),
        "matrix": "pmatrix" in a or "bmatrix" in a or "matrix" in a,
        "text_unit": "\\text" in a or "\\mbox" in a,
        "has_var": bool(re.search(r"[a-zA-Z]", re.sub(r"\\(text|frac|sqrt|pi|circ|begin|end|pmatrix|cdot|times|left|right|sin|cos|tan|log|ln)", "", a))),
    }
# "format-ambiguous" = answer has >=1 representation-variant axis (model can be
# right yet parser-wrong on some rollouts -> natural in-zone with reliable EOS)
AMBIG_KEYS = ["fraction", "radical", "pi", "degree", "tuple_list", "matrix", "text_unit"]


def summarize(label, idxs, ans_col):
    n = len(idxs)
    if not n:
        print(f"  [{label}] empty"); return
    feat_counts = Counter()
    ambig = 0
    for i in idxs:
        if i is None or i >= len(ans_col):
            continue
        f = ans_features(ans_col[i])
        for k, v in f.items():
            if v:
                feat_counts[k] += 1
        if any(f[k] for k in AMBIG_KEYS):
            ambig += 1
    print(f"  [{label}] n={n}  FORMAT-AMBIGUOUS={ambig/n:.0%}")
    print("    " + "  ".join(f"{k}={feat_counts[k]/n:.0%}" for k in
          ["plain_int","decimal","fraction","radical","pi","degree","tuple_list","matrix","text_unit","has_var"]))


def main():
    n_back = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    hi = cur_window()
    windows = list(range(hi, hi - n_back, -1))
    print(f"current window={hi}; harvesting {len(windows)} windows (all hotkeys, in-zone k=2..6)")

    all_rows, errs = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(harvest_window, w) for w in windows]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r is None:
                errs += 1
            else:
                all_rows.extend(r)
            if (i + 1) % 40 == 0:
                print(f"  ...{i+1}/{len(windows)} (in-zone rows={len(all_rows)}, unavailable={errs})")

    by_hotkey = Counter(r["hotkey"] for r in all_rows)
    idx_set = sorted({r["idx"] for r in all_rows if isinstance(r["idx"], int)})
    json.dump(idx_set, open("/root/inzone_idxs.json", "w"))
    print(f"\nharvested {len(all_rows)} in-zone groups, {len(idx_set)} distinct prompt_idx, "
          f"{len(by_hotkey)} hotkeys ({errs} windows unavailable)")
    print(f"top hotkeys by in-zone volume: {by_hotkey.most_common(6)}")
    print(f"k distribution (all miners): {dict(sorted(Counter(r['k'] for r in all_rows).items()))}")

    # ---- map idx -> features via local dataset; compare vs random baseline ----
    try:
        from reliquary.environment import load_environment
        env = load_environment("openmathinstruct")
        ds = env._dataset
        cols = ds.column_names
        print(f"\ndataset rows={len(ds)} columns={cols}")
        ans_key = "expected_answer" if "expected_answer" in cols else cols[-1]
        src_key = "problem_source" if "problem_source" in cols else None
        prob_key = "problem" if "problem" in cols else ("question" if "question" in cols else cols[0])
        ans_col = ds[ans_key]
        src_col = ds[src_key] if src_key else None
        prob_col = ds[prob_key]
        N = len(ds)

        # winners restricted to augmented_math (what we mine), in-range
        win_idxs = [i for i in idx_set if i < N and (src_col is None or src_col[i] == "augmented_math")]
        # random augmented_math baseline
        import random
        random.seed(0)
        if src_col is not None:
            am_pool = [i for i, s in enumerate(src_col) if s == "augmented_math"]
        else:
            am_pool = list(range(N))
        base_idxs = random.sample(am_pool, min(3000, len(am_pool)))

        print(f"\n=== ANSWER-FORMAT comparison (augmented_math) ===")
        summarize("RANDOM baseline", base_idxs, ans_col)
        summarize("IN-ZONE winners", win_idxs, ans_col)

        # write winners.jsonl for frontier seeding (prompt + ground_truth)
        os.makedirs("/root/wf_data", exist_ok=True)
        with open("/root/wf_data/winners.jsonl", "w") as f:
            for i in win_idxs:
                f.write(json.dumps({"idx": i, "prompt": prob_col[i], "ground_truth": ans_col[i]}) + "\n")
        print(f"\nwrote {len(win_idxs)} winners to /root/wf_data/winners.jsonl (frontier seed)")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n(dataset feature analysis skipped: {e})")
        print(f"harvested idxs saved to /root/inzone_idxs.json")


if __name__ == "__main__":
    main()
