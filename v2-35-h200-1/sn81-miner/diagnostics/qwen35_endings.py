"""FAST: dump the exact ENDINGS of live-Qwen3.5 completions on NUMERIC-answer prompts,
split by reward, to see what format blocks extraction and whether a selection lever
exists. Few prompts, n=10, 6144 budget (gsm8k terminates well within this)."""
import re, random, json, traceback
SNAP = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/7926d852f1d955f44443fac1476681e0e0fdde92/"
POOL = "/root/sn81-miner/data/inzone_pool_qwen35.json"
N = 8
N_PROMPTS = 4


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from reliquary.constants import T_PROTO, TOP_P_PROTO, TOP_K_PROTO
    from reliquary.environment import load_environment
    from reliquary.environment.openmathinstruct import _normalize_answer
    from reliquary.protocol.tokens import encode_prompt

    tok = AutoTokenizer.from_pretrained(SNAP)
    env = load_environment("openmathinstruct")
    pool = json.load(open(POOL))
    random.seed(3)
    idxs = random.sample(pool, N_PROMPTS)
    llm = LLM(model=SNAP, tokenizer=SNAP, dtype="bfloat16", gpu_memory_utilization=0.80,
              max_model_len=9216, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    top_k = -1 if TOP_K_PROTO <= 0 else TOP_K_PROTO
    sp = SamplingParams(n=N, temperature=T_PROTO, top_p=TOP_P_PROTO, top_k=top_k,
                        max_tokens=2048, stop_token_ids=[248044, 248046])  # thinking OFF -> short direct answers
    n_extractable = n_term = n_total = n_box = 0
    print("PROMPT[0] (confirm boxed instruction present):", repr(env.get_problem(idxs[0])["prompt"][-80:]))
    for idx in idxs:
        p = env.get_problem(idx)
        gt = _normalize_answer(p.get("ground_truth", ""))
        out = llm.generate({"prompt_token_ids": encode_prompt(tok, p["prompt"])}, sp)
        print("\n" + "=" * 90)
        print(f"IDX {idx}  GT={gt!r}  Q={p['prompt'][:110]!r}")
        for i, o in enumerate(out[0].outputs):
            txt = o.text
            n_total += 1
            if o.finish_reason == "stop":
                n_term += 1
            r = float(env.compute_reward(p, txt))
            if r >= 1.0:
                n_extractable += 1
            has_box = ("\\boxed{" in txt) or ("\\fbox{" in txt)
            if has_box:
                n_box += 1
            last_line = txt.strip().split("\n")[-1].strip()
            tag = "OK " if r >= 1.0 else "XX "
            bx = "BOX" if has_box else "   "
            print(f"  {tag}{bx} r={r} fin={o.finish_reason} LAST_LINE={last_line[:90]!r}")
    print("\n================ SUMMARY ================")
    print(f"total={n_total}  terminated={n_term} ({100*n_term/n_total:.0f}%)  "
          f"boxed={n_box} ({100*n_box/n_total:.0f}%)  "
          f"extractable_correct={n_extractable} ({100*n_extractable/n_total:.0f}%)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
