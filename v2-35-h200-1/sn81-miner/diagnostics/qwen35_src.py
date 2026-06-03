"""Box-rate by SOURCE on live 7926: does the model close </think> + \\boxed on easier
sources (gsm8k) but ramble forever on augmented_math? Tests the source-selection lever."""
import json, traceback
SNAP = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/7926d852f1d955f44443fac1476681e0e0fdde92/"
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from reliquary.constants import T_PROTO, TOP_P_PROTO, TOP_K_PROTO
    from reliquary.environment import load_environment
    from reliquary.protocol.tokens import encode_prompt
    import random
    tok = AutoTokenizer.from_pretrained(SNAP)
    env = load_environment("openmathinstruct")
    srcs = env._dataset["problem_source"]
    by = {}
    for i, s in enumerate(srcs):
        by.setdefault(s, []).append(i)
    print("sources available:", {k: len(v) for k, v in by.items()})
    random.seed(1)
    pick = []
    for s in ("gsm8k", "augmented_gsm8k", "math", "augmented_math"):
        if by.get(s):
            for idx in random.sample(by[s], min(2, len(by[s]))):
                pick.append((s, idx))
    llm = LLM(model=SNAP, tokenizer=SNAP, dtype="bfloat16", gpu_memory_utilization=0.80,
              max_model_len=9216, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    tk = -1 if TOP_K_PROTO <= 0 else TOP_K_PROTO
    sp = SamplingParams(n=8, temperature=T_PROTO, top_p=TOP_P_PROTO, top_k=tk,
                        max_tokens=8192, stop_token_ids=[248044, 248046])
    print("SRC | boxed/8 closed_think/8 correct/8 boxed&corr/8 term/8 meanlen")
    for s, idx in pick:
        p = env.get_problem(idx)
        out = llm.generate({"prompt_token_ids": encode_prompt(tok, p["prompt"])}, sp)
        b = c = corr = bc = term = 0; ln = 0
        for o in out[0].outputs:
            t = o.text; hb = "\\boxed" in t; cl = "</think>" in t
            r = float(env.compute_reward(p, t))
            b += int(hb); c += int(cl); corr += int(r >= 1); bc += int(hb and r >= 1)
            term += int(o.finish_reason == "stop"); ln += len(o.token_ids)
        print(f"SRC {s:16} idx={idx} | boxed={b}/8 closed={c}/8 corr={corr}/8 box&corr={bc}/8 term={term}/8 len={ln//8}")
except Exception:
    traceback.print_exc()
