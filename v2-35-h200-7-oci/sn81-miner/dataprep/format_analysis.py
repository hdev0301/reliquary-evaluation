#!/usr/bin/env python3
"""Offline test of the format-ambiguity hypothesis (no API; uses cached winners
+ local dataset). Why pregen is +0/16: the converged model is BIMODAL (8/8 or
0/8), so the only honest route into the 2..6 zone is prompts whose ANSWER has
many equivalent string forms -> a confident model still scatters in/out of the
parser. If true, in-zone winners' ground_truths are far more format-ambiguous
(and far less plain-integer) than random augmented_math.

Also writes /root/wf_data/winners.jsonl (frontier seed) and a curated
non-cooldown candidate pool /root/inzone_pool.json (format-ambiguous easy idxs).
"""
import json, os, re
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ---- in-zone winners harvested earlier (idx-level), all augmented_math ----
def load_inzone_idxs():
    idxs = set()
    for p in ["/root/topminer_vectors.jsonl", "/root/topminer_accepts.jsonl"]:
        if os.path.exists(p):
            for line in open(p):
                try:
                    r = json.loads(line)
                    i = r.get("idx")
                    if isinstance(i, int):
                        idxs.add(i)
                except Exception:
                    pass
    if os.path.exists("/root/inzone_idxs.json"):
        try:
            idxs.update(json.load(open("/root/inzone_idxs.json")))
        except Exception:
            pass
    return sorted(idxs)


def ans_features(a: str) -> dict:
    a = str(a)
    core = re.sub(r"\\(text|frac|sqrt|pi|circ|begin|end|pmatrix|bmatrix|cdot|times|left|right|sin|cos|tan|log|ln|theta|alpha|beta)", "", a)
    return {
        "plain_int": bool(re.fullmatch(r"-?\d+", a.strip())),
        "decimal": bool(re.fullmatch(r"-?\d+\.\d+", a.strip())),
        "fraction": ("\\frac" in a) or bool(re.search(r"\b\d+/\d+\b", a)),
        "radical": "\\sqrt" in a,
        "pi": "\\pi" in a,
        "degree": "\\circ" in a or "degree" in a.lower() or "°" in a,
        "tuple_list": ("," in a) or (";" in a) or bool(re.search(r"[()\[\]]", a)),
        "matrix": "matrix" in a,
        "text_unit": "\\text" in a or "\\mbox" in a,
        "has_var": bool(re.search(r"[a-zA-Z]", core)),
    }
AMBIG = ["fraction", "radical", "pi", "degree", "tuple_list", "matrix", "text_unit"]


def is_ambiguous(a):
    f = ans_features(a)
    return any(f[k] for k in AMBIG) or f["has_var"]


def summarize(label, idxs, ans_col, prob_col=None):
    n = 0
    fc = Counter(); ambig = 0; plen = []
    for i in idxs:
        if not isinstance(i, int) or i >= len(ans_col):
            continue
        n += 1
        f = ans_features(ans_col[i])
        for k, v in f.items():
            if v:
                fc[k] += 1
        if is_ambiguous(ans_col[i]):
            ambig += 1
        if prob_col is not None:
            plen.append(len(str(prob_col[i])))
    if not n:
        print(f"  [{label}] no usable idxs"); return
    print(f"  [{label}] n={n}  FORMAT-AMBIGUOUS={ambig/n:.0%}" +
          (f"  mean prompt_chars={sum(plen)//len(plen)}" if plen else ""))
    print("    " + "  ".join(f"{k}={fc[k]/n:.0%}" for k in
          ["plain_int","decimal","fraction","radical","pi","degree","tuple_list","matrix","text_unit","has_var"]))


def main():
    win = load_inzone_idxs()
    print(f"loaded {len(win)} distinct in-zone winner idxs from local caches")
    try:
        from reliquary.environment import load_environment
        env = load_environment("openmathinstruct")
        ds = env._dataset
        cols = ds.column_names
        print(f"dataset rows={len(ds)} cols={cols}")
        ans_key = "expected_answer" if "expected_answer" in cols else cols[-1]
        prob_key = "problem" if "problem" in cols else ("question" if "question" in cols else cols[0])
        src_key = "problem_source" if "problem_source" in cols else None
        ans_col = ds[ans_key]; prob_col = ds[prob_key]
        src_col = ds[src_key] if src_key else None
        N = len(ds)

        win = [i for i in win if i < N]
        import random; random.seed(0)
        am_pool = [i for i, s in enumerate(src_col) if s == "augmented_math"] if src_col else list(range(N))
        base = random.sample(am_pool, min(4000, len(am_pool)))

        print("\n=== ANSWER-FORMAT: in-zone winners vs random augmented_math ===")
        summarize("RANDOM augmented_math", base, ans_col, prob_col)
        summarize("IN-ZONE winners      ", win, ans_col, prob_col)

        # Build a curated NON-redundant candidate pool: augmented_math + ambiguous
        # answer + short-ish prompt (proxy for 'easy/terminating'). These are the
        # prompts that should both terminate AND scatter into the zone.
        amb_pool = [i for i in am_pool if is_ambiguous(ans_col[i]) and len(str(prob_col[i])) <= 400]
        json.dump(amb_pool, open("/root/inzone_pool.json", "w"))
        print(f"\ncurated candidate pool (augmented_math + ambiguous answer + prompt<=400 chars): "
              f"{len(amb_pool)} prompts -> /root/inzone_pool.json")

        os.makedirs("/root/wf_data", exist_ok=True)
        with open("/root/wf_data/winners.jsonl", "w") as f:
            for i in win:
                f.write(json.dumps({"idx": i, "prompt": prob_col[i], "ground_truth": ans_col[i]}) + "\n")
        print(f"wrote {len(win)} winners -> /root/wf_data/winners.jsonl (frontier seed)")
    except Exception as e:
        import traceback; traceback.print_exc()
        # fallback: use truncated gt straight from topminer_accepts.jsonl
        print(f"\n(dataset unavailable: {e}; using truncated gt from topminer_accepts.jsonl)")
        gts = []
        if os.path.exists("/root/topminer_accepts.jsonl"):
            for line in open("/root/topminer_accepts.jsonl"):
                try:
                    gts.append(json.loads(line).get("gt", ""))
                except Exception:
                    pass
        if gts:
            fc = Counter(); ambig = 0
            for a in gts:
                for k, v in ans_features(a).items():
                    if v: fc[k] += 1
                if is_ambiguous(a): ambig += 1
            n = len(gts)
            print(f"in-zone winners (truncated gt) n={n} FORMAT-AMBIGUOUS={ambig/n:.0%}")
            print("  " + "  ".join(f"{k}={fc[k]/n:.0%}" for k in
                  ["plain_int","decimal","fraction","radical","pi","degree","tuple_list","matrix","text_unit","has_var"]))


if __name__ == "__main__":
    main()
