"""Dump actual live-7926 completion text to see WHAT it generates on inzone prompts."""
import json, traceback
SNAP = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/7926d852f1d955f44443fac1476681e0e0fdde92/"
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from reliquary.constants import T_PROTO, TOP_P_PROTO, TOP_K_PROTO
    from reliquary.environment import load_environment
    from reliquary.protocol.tokens import encode_prompt
    tok = AutoTokenizer.from_pretrained(SNAP)
    env = load_environment("openmathinstruct")
    pool = json.load(open("/root/inzone_pool.json"))
    idxs = pool[:2]
    llm = LLM(model=SNAP, tokenizer=SNAP, dtype="bfloat16", gpu_memory_utilization=0.80,
              max_model_len=9216, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    top_k = -1 if TOP_K_PROTO <= 0 else TOP_K_PROTO
    sp = SamplingParams(n=3, temperature=T_PROTO, top_p=TOP_P_PROTO, top_k=top_k,
                        max_tokens=8192, stop_token_ids=[248044, 248046])
    for idx in idxs:
        p = env.get_problem(idx)
        ptoks = encode_prompt(tok, p["prompt"])
        print("\n" + "=" * 80)
        print("IDX", idx, "GT=", repr(p.get("ground_truth")))
        print("PROMPT(decoded chat-template):", repr(tok.decode(ptoks)[:400]))
        out = llm.generate({"prompt_token_ids": ptoks}, sp)
        for i, o in enumerate(out[0].outputs):
            txt = o.text
            r = float(env.compute_reward(p, txt))
            print(f"\n--- comp[{i}] len={len(o.token_ids)} finish={o.finish_reason} reward={r} ---")
            print("HEAD:", repr(txt[:250]))
            print("TAIL:", repr(txt[-400:]))
except Exception:
    traceback.print_exc()
