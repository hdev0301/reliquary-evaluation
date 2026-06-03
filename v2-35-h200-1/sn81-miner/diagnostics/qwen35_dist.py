"""Measure the LIVE Qwen3.5 checkpoint (7926) correctness distribution on inzone prompts.
Decides whether a curatable (mixed 2-6 correct) band exists for curation."""
import json, statistics as st, traceback
SNAP = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/7926d852f1d955f44443fac1476681e0e0fdde92/"
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from reliquary.constants import T_PROTO, TOP_P_PROTO, TOP_K_PROTO
    from reliquary.environment import load_environment
    from reliquary.protocol.tokens import encode_prompt
    tok = AutoTokenizer.from_pretrained(SNAP)
    env = load_environment("openmathinstruct")
    import random
    random.seed(7)
    pool = json.load(open("/root/inzone_pool.json"))
    idxs = random.sample(pool, 12)
    llm = LLM(model=SNAP, tokenizer=SNAP, dtype="bfloat16", gpu_memory_utilization=0.80,
              max_model_len=9216, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    top_k = -1 if TOP_K_PROTO <= 0 else TOP_K_PROTO
    sp = SamplingParams(n=8, temperature=T_PROTO, top_p=TOP_P_PROTO, top_k=top_k,
                        max_tokens=8192, stop_token_ids=[248044, 248046])
    buckets = {"7-8 corr": 0, "mixed 2-6": 0, "0-1 corr": 0}
    print("RESULTS idx | corr/8 term/8 meanlen | gt")
    for idx in idxs:
        p = env.get_problem(idx)
        out = llm.generate({"prompt_token_ids": encode_prompt(tok, p["prompt"])}, sp)
        corr = term = 0
        lens = []
        for o in out[0].outputs:
            r = float(env.compute_reward(p, o.text))
            corr += int(r >= 1)
            term += int(o.finish_reason == "stop")
            lens.append(len(o.token_ids))
        b = "7-8 corr" if corr >= 7 else ("mixed 2-6" if 2 <= corr <= 6 else "0-1 corr")
        buckets[b] += 1
        print(f"RESULTS {idx} | {corr}/8 {term}/8 {int(st.mean(lens))} | {str(p.get('ground_truth'))[:18]!r}")
    print("BUCKETS", buckets)
    print("CURATABLE-BAND (mixed 2-6 with also >=2 wrong) is what curation needs.")
except Exception:
    traceback.print_exc()
