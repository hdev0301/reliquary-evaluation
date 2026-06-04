"""DECISIVE probe: is avg_n_correct=0 a CAPABILITY miss (model gets gsm8k wrong)
or an EXTRACTION miss (model solves correctly but ends in prose the reward can't
parse)? For each completion record reward, </think>-closed, \\boxed present, the
LAST line (what the fallback regex sees), and whether the normalized ground-truth
number appears ANYWHERE in the text (=> model knew it, format blocked the score).

Run with the miner STOPPED (needs the GPU). Writes a compact aggregate + samples.
"""
import json, re, random, traceback
SNAP = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/7926d852f1d955f44443fac1476681e0e0fdde92/"
POOL = "/root/sn81-miner/data/inzone_pool_qwen35.json"
N_PROMPTS = 24
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from reliquary.constants import T_PROTO, TOP_P_PROTO, TOP_K_PROTO
    from reliquary.environment import load_environment
    from reliquary.environment.openmathinstruct import _normalize_answer
    from reliquary.protocol.tokens import encode_prompt

    tok = AutoTokenizer.from_pretrained(SNAP)
    env = load_environment("openmathinstruct")
    pool = json.load(open(POOL))
    random.seed(11)
    idxs = random.sample(pool, N_PROMPTS)
    llm = LLM(model=SNAP, tokenizer=SNAP, dtype="bfloat16", gpu_memory_utilization=0.80,
              max_model_len=9216, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    top_k = -1 if TOP_K_PROTO <= 0 else TOP_K_PROTO
    sp = SamplingParams(n=8, temperature=T_PROTO, top_p=TOP_P_PROTO, top_k=top_k,
                        max_tokens=8192, stop_token_ids=[248044, 248046])

    def gt_in_text(gt: str, text: str) -> bool:
        if not gt:
            return False
        # gt appears as a standalone number somewhere in the text
        return re.search(r"(?<![\d.])" + re.escape(gt) + r"(?![\d.])", text) is not None

    tot_term = tot_corr = tot_box = tot_closed = 0
    tot_comps = 0
    tot_gt_present = tot_extract_miss = 0   # extract_miss = gt in text but reward 0
    n_prompts_with_2corr = 0
    samples = []
    print(f"probing {N_PROMPTS} prompts x 8 ...", flush=True)
    for idx in idxs:
        p = env.get_problem(idx)
        gt = _normalize_answer(p.get("ground_truth", ""))
        out = llm.generate({"prompt_token_ids": encode_prompt(tok, p["prompt"])}, sp)
        corr = term = box = closed = gtp = emiss = 0
        last_lines = []
        for o in out[0].outputs:
            txt = o.text
            tot_comps += 1
            terminated = o.finish_reason == "stop"
            if terminated:
                term += 1; tot_term += 1
            r = float(env.compute_reward(p, txt))
            if r >= 1.0:
                corr += 1; tot_corr += 1
            if "\\boxed{" in txt or "\\fbox{" in txt:
                box += 1; tot_box += 1
            if "</think>" in txt:
                closed += 1; tot_closed += 1
            present = gt_in_text(gt, txt)
            if present:
                gtp += 1; tot_gt_present += 1
                if r < 1.0:
                    emiss += 1; tot_extract_miss += 1
            last_lines.append(txt.strip().split("\n")[-1].strip()[:80])
        if corr >= 2:
            n_prompts_with_2corr += 1
        samples.append((idx, gt, corr, term, box, closed, gtp, emiss, last_lines[:3]))
        print(f"idx={idx} gt={gt!r} corr={corr}/8 term={term}/8 boxed={box} closed={closed} "
              f"gt_in_text={gtp} extract_miss={emiss}", flush=True)

    print("\n================ AGGREGATE ================")
    print(f"completions={tot_comps}  terminated={tot_term} ({100*tot_term/tot_comps:.0f}%)  "
          f"closed</think>={tot_closed} ({100*tot_closed/tot_comps:.0f}%)  "
          f"boxed={tot_box} ({100*tot_box/tot_comps:.0f}%)")
    print(f"reward-correct={tot_corr} ({100*tot_corr/tot_comps:.0f}%)")
    print(f"gt_present_in_text={tot_gt_present} ({100*tot_gt_present/tot_comps:.0f}%)  "
          f"EXTRACT_MISS(gt present but reward 0)={tot_extract_miss} ({100*tot_extract_miss/tot_comps:.0f}%)")
    print(f"prompts with >=2 reward-correct (curatable for k=2): {n_prompts_with_2corr}/{N_PROMPTS}")
    print("\nVERDICT: if EXTRACT_MISS high vs reward-correct low => FORMAT problem (model solves, "
          "doesn't end with bare number). If gt_present low too => capability/too-hard.")
    print("\n---- sample last-lines (what the fallback regex sees) ----")
    for idx, gt, corr, term, box, closed, gtp, emiss, lls in samples[:8]:
        print(f"\nidx={idx} gt={gt!r} corr={corr}/8 extract_miss={emiss}")
        for ll in lls:
            print("   LAST:", repr(ll))
except Exception:
    traceback.print_exc()
