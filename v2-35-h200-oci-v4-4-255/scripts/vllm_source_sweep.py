#!/usr/bin/env python3
"""Per-source difficulty diagnostic — is ANY prompt regime in-zone-feasible?

gsm8k/aug_gsm8k already shown: terminate ⟺ correct (P(correct|term)=1.0), so no
in-zone groups. This tests harder sources (math, augmented_math) where the model
may terminate on a WRONG answer (P(correct|term) < 1.0) — the only structural
path to a genuine 2..6/8 in-zone group. Terminated-only grading via the exact
env.compute_reward the validator re-runs.
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("src_sweep")

CKPT = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/2a66ba495544fd75c2cd0fe373aa9ef46b4c6b45"
SOURCES = ["gsm8k", "augmented_gsm8k", "augmented_math", "math"]
N_PROMPTS_PER_SRC = 12
N_SAMPLES = 64
MAX_TOKENS = 2048
TEMP = 1.0
SEED = 1234


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from reliquary.environment import load_environment

    env = load_environment("openmathinstruct")
    ds = env._dataset
    srcs = ds["problem_source"]
    by_src = {s: [] for s in SOURCES}
    for i, s in enumerate(srcs):
        if s in by_src and len(by_src[s]) < N_PROMPTS_PER_SRC * 40:
            by_src[s].append(i)

    chosen = []   # (source, problem)
    for s in SOURCES:
        idxs = by_src[s]
        if not idxs:
            logger.warning("source %s: 0 prompts found", s)
            continue
        stride = max(1, len(idxs) // N_PROMPTS_PER_SRC)
        for i in idxs[::stride][:N_PROMPTS_PER_SRC]:
            chosen.append((s, env.get_problem(i)))
    logger.info("Picked %d prompts across %s", len(chosen), SOURCES)

    tok = AutoTokenizer.from_pretrained(CKPT)
    vllm_prompts = [{"prompt_token_ids": tok.encode(p["prompt"], add_special_tokens=False)}
                    for _, p in chosen]

    llm = LLM(model=CKPT, tokenizer=CKPT, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=2560,
              enable_prefix_caching=True, tensor_parallel_size=1, seed=SEED)

    sp = SamplingParams(n=N_SAMPLES, temperature=TEMP, top_p=1.0, top_k=-1, max_tokens=MAX_TOKENS)
    outputs = llm.generate(vllm_prompts, sp)

    per_src = {s: {"n_prompts": 0, "term": 0, "corr_term": 0, "samples": 0,
                   "ge8_term": 0, "in_zone_feasible": 0, "prompts": []} for s in SOURCES}
    for (s, p), out in zip(chosen, outputs):
        tc = []
        for comp in out.outputs:
            per_src[s]["samples"] += 1
            if comp.finish_reason == "stop":
                r = env.compute_reward(p, comp.text)
                tc.append(1 if r >= 1.0 else 0)
        n_term = len(tc); n_corr = sum(tc)
        d = per_src[s]
        d["n_prompts"] += 1
        d["term"] += n_term
        d["corr_term"] += n_corr
        if n_term >= 8:
            d["ge8_term"] += 1
            pc = n_corr / n_term
            if 0.15 <= pc <= 0.85:
                d["in_zone_feasible"] += 1
        d["prompts"].append({"id": p["id"], "n_term": n_term, "n_corr": n_corr,
                             "p_corr": round(n_corr / n_term, 3) if n_term else None})

    summary = {}
    for s in SOURCES:
        d = per_src[s]
        if d["n_prompts"] == 0:
            continue
        summary[s] = {
            "n_prompts": d["n_prompts"],
            "term_rate": round(d["term"] / d["samples"], 4) if d["samples"] else 0,
            "P_correct_given_term": round(d["corr_term"] / d["term"], 4) if d["term"] else None,
            "prompts_ge8_term": d["ge8_term"],
            "in_zone_feasible_prompts": d["in_zone_feasible"],
            "prompts": d["prompts"],
        }
        logger.info("src=%-16s term_rate=%.4f P(correct|term)=%s ge8term=%d/%d in_zone_feasible=%d",
                    s, summary[s]["term_rate"], summary[s]["P_correct_given_term"],
                    d["ge8_term"], d["n_prompts"], d["in_zone_feasible"])

    with open("/root/vllm_source_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== PER-SOURCE DIFFICULTY DIAGNOSTIC (T=1.0, terminated-only) ===")
    for s in SOURCES:
        if s in summary:
            r = summary[s]
            print(f"{s:16s} term_rate={r['term_rate']:.4f} P(correct|term)={r['P_correct_given_term']} "
                  f"ge8term={r['prompts_ge8_term']}/{r['n_prompts']} in_zone_feasible={r['in_zone_feasible_prompts']}")
    logger.info("wrote /root/vllm_source_sweep.json")


if __name__ == "__main__":
    main()
