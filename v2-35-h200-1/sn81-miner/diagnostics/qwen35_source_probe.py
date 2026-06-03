"""DECISIVE source comparison on the LIVE Qwen3.5 (817). For each source measure:
  - box_rate     : completions containing \\boxed/\\fbox
  - term_rate    : completions that hit EOS (finish=stop)
  - extract_corr : completions the validator reward scores >=1 (THE thing we need)
  - gt_present   : completions containing the gt string (model 'knows' the answer)
Plus dump the actual ENDING (last 220 chars) of a few completions per source so we
can SEE the format. This decides which pool (if any) is minable with this checkpoint.

Run with miner STOPPED. ~5 prompts/source, n=6 each, 8192 budget.
Wrapped in __main__ guard: vLLM uses spawn and re-imports this module."""
import json, re, random, traceback

SNAP = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/7926d852f1d955f44443fac1476681e0e0fdde92/"
SOURCES = ["gsm8k", "augmented_gsm8k", "math", "augmented_math"]
PER_SRC = 5
N = 6


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from reliquary.constants import T_PROTO, TOP_P_PROTO, TOP_K_PROTO
    from reliquary.environment import load_environment
    from reliquary.environment.openmathinstruct import _normalize_answer
    from reliquary.protocol.tokens import encode_prompt
    import pyarrow as pa

    tok = AutoTokenizer.from_pretrained(SNAP)
    env = load_environment("openmathinstruct")
    tbl = env._dataset.data.table
    src_col = tbl["problem_source"].cast(pa.string()).combine_chunks().to_pylist()
    by = {}
    for i, s in enumerate(src_col):
        if s in SOURCES:
            by.setdefault(s, []).append(i)
    print("available per source:", {k: len(v) for k, v in by.items()}, flush=True)
    random.seed(5)
    pick = []
    for s in SOURCES:
        if by.get(s):
            for idx in random.sample(by[s], min(PER_SRC, len(by[s]))):
                pick.append((s, idx))

    llm = LLM(model=SNAP, tokenizer=SNAP, dtype="bfloat16", gpu_memory_utilization=0.80,
              max_model_len=9216, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    top_k = -1 if TOP_K_PROTO <= 0 else TOP_K_PROTO
    sp = SamplingParams(n=N, temperature=T_PROTO, top_p=TOP_P_PROTO, top_k=top_k,
                        max_tokens=8192, stop_token_ids=[248044, 248046])

    agg = {s: dict(comps=0, term=0, box=0, corr=0, gtp=0) for s in SOURCES}
    endings = {s: [] for s in SOURCES}
    for s, idx in pick:
        p = env.get_problem(idx)
        gt = _normalize_answer(p.get("ground_truth", ""))
        out = llm.generate({"prompt_token_ids": encode_prompt(tok, p["prompt"])}, sp)
        c = corr = term = box = gtp = 0
        for o in out[0].outputs:
            txt = o.text
            agg[s]["comps"] += 1; c += 1
            if o.finish_reason == "stop":
                agg[s]["term"] += 1; term += 1
            if "\\boxed{" in txt or "\\fbox{" in txt:
                agg[s]["box"] += 1; box += 1
            r = float(env.compute_reward(p, txt))
            if r >= 1.0:
                agg[s]["corr"] += 1; corr += 1
            if gt and re.search(r"(?<![\w.])" + re.escape(gt) + r"(?![\w.])", txt):
                agg[s]["gtp"] += 1; gtp += 1
            if len(endings[s]) < 4:
                endings[s].append((r, o.finish_reason, repr(txt[-220:])))
        print(f"[{s}] idx={idx} gt={gt!r} corr={corr}/{N} term={term}/{N} box={box}/{N} gt_in_text={gtp}/{N}", flush=True)

    print("\n================ PER-SOURCE AGGREGATE ================")
    for s in SOURCES:
        a = agg[s]
        if a["comps"]:
            print(f"{s:18s} comps={a['comps']:3d}  term={100*a['term']/a['comps']:3.0f}%  "
                  f"box={100*a['box']/a['comps']:3.0f}%  EXTRACT_CORR={100*a['corr']/a['comps']:3.0f}%  "
                  f"gt_present={100*a['gtp']/a['comps']:3.0f}%")
    print("\n================ SAMPLE ENDINGS (last 220 chars) ================")
    for s in SOURCES:
        print(f"\n----- {s} -----")
        for r, fr, e in endings[s]:
            print(f"  reward={r} finish={fr}\n     ...{e}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
