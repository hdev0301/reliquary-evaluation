#!/usr/bin/env python3
"""Screen a candidate pool against the LIVE checkpoint and report the in-zone
k-distribution — so we know whether the pool sits at the model's learning
frontier (k=2..6 correct of 8, sigma>=0.43) BEFORE committing a mining run.

Uses the SAME protocol sampling params (T_PROTO/top_p/top_k, EOS set) and the
validator-equivalent VALUE-based reward (reliquary.miner.pregen.value_match_reward)
so the measured k matches what the validator will recompute.

Run (GPU must be free):
  cd /root/reliquary && .venv/bin/python /root/sn81-miner/dataprep/screen_pool.py \
      --pool /root/sn81-miner/data/inzone_pool_v2.json --n 64 --max-tokens 512
"""
import argparse, json, random, re, sys
from collections import Counter


def shape_of(gt: str) -> str:
    s = (gt or "").strip()
    if re.fullmatch(r"[\-\+]?\d+", s):
        return "int"
    if re.fullmatch(r"[\-\+]?\d+\.\d+", s):
        return "decimal"
    if "\\frac" in s or re.fullmatch(r"[\-\+]?\d+/\d+", s):
        return "fraction"
    if "\\sqrt" in s:
        return "radical"
    if "pi" in s:
        return "pi"
    if any(c.isalpha() for c in s) or "(" in s:
        return "var/text"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--snap", default="/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/e7324d79fdf2bfcca7949f033d5464cb2dbbfa1a/")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from reliquary.constants import T_PROTO, TOP_P_PROTO, TOP_K_PROTO
    from reliquary.environment import load_environment
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.miner.pregen import value_match_reward  # validator-equivalent reward

    tok = AutoTokenizer.from_pretrained(args.snap)
    env = load_environment("openmathinstruct")
    pool = json.load(open(args.pool))
    random.seed(args.seed)
    idxs = random.sample(pool, min(args.n, len(pool)))

    llm = LLM(model=args.snap, tokenizer=args.snap, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, max_model_len=max(2048, args.max_tokens + 1024),
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    top_k = -1 if TOP_K_PROTO <= 0 else TOP_K_PROTO
    sp = SamplingParams(n=8, temperature=T_PROTO, top_p=TOP_P_PROTO, top_k=top_k,
                        max_tokens=args.max_tokens, stop_token_ids=[248044, 248046])

    khist = Counter()                       # k(correct/8) -> count
    by_shape = {}                           # shape -> [in_zone, total]
    inzone = eight = zero = 0
    print(f"screening {len(idxs)} prompts from {args.pool} @ {args.snap.split('/')[-2][:12]} max_tok={args.max_tokens}")
    # Build all prompts and generate in ONE batched call (vLLM continuous batching)
    # so 64*8=512 seqs run together instead of 64 serial 8-seq calls.
    probs, reqs = [], []
    for idx in idxs:
        p = env.get_problem(idx)
        probs.append((idx, p))
        reqs.append({"prompt_token_ids": encode_prompt(tok, p["prompt"])})
    outs = llm.generate(reqs, sp)           # one RequestOutput per prompt, in order

    print("idx | k/8 term/8 shape | gt")
    for (idx, p), ro in zip(probs, outs):
        gt = p.get("ground_truth", "")
        k = term = 0
        for o in ro.outputs:
            txt = o.text
            if o.finish_reason == "stop":
                term += 1
            if value_match_reward(p, txt) >= 1.0:
                k += 1
        khist[k] += 1
        sh = shape_of(gt)
        s = by_shape.setdefault(sh, [0, 0]); s[1] += 1
        if 2 <= k <= 6:
            inzone += 1; s[0] += 1
        if k == 8:
            eight += 1
        if k == 0:
            zero += 1
        print(f"{idx} | {k}/8 {term}/8 {sh:8s} | {gt[:30]}")

    n = sum(khist.values()) or 1
    print("\n=== k-distribution (correct of 8) ===")
    for k in range(9):
        bar = "#" * khist.get(k, 0)
        tag = "  <- in-zone" if 2 <= k <= 6 else ("  <- 8/8 waste" if k == 8 else ("  <- 0/8 waste" if k == 0 else ""))
        print(f"  k={k}: {khist.get(k,0):3d} {bar}{tag}")
    print(f"\nIN-ZONE (k=2..6): {inzone}/{n} = {100*inzone/n:.0f}%   |   8/8 waste: {100*eight/n:.0f}%   0/8 waste: {100*zero/n:.0f}%")
    print("\n=== in-zone rate by answer shape ===")
    for sh, (iz, tot) in sorted(by_shape.items(), key=lambda x: -x[1][1]):
        print(f"  {sh:9s}: {iz:3d}/{tot:3d} in-zone = {100*iz/max(1,tot):.0f}%")
    # verdict
    print()
    if inzone / n >= 0.20:
        print(f"VERDICT: HEALTHY ({100*inzone/n:.0f}% in-zone) — pool sits at the frontier; launch it.")
    elif inzone / n >= 0.10:
        print(f"VERDICT: MARGINAL ({100*inzone/n:.0f}% in-zone) — usable but consider tuning shape mix toward the best-yielding shapes above.")
    else:
        print(f"VERDICT: POOR ({100*inzone/n:.0f}% in-zone) — wrong frontier for this checkpoint; retune ratios.")


if __name__ == "__main__":
    sys.exit(main())
