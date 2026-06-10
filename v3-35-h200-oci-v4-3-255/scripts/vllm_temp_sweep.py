#!/usr/bin/env python3
"""vLLM temperature sweep — terminated-only correctness (production-faithful).

Mirrors the miner: oversample n completions per prompt at the protocol cap,
keep only EOS-terminated ones (finish_reason == "stop"), grade with the exact
env.compute_reward the validator re-runs. Reports, per temperature:

  * termination rate (P(terminate) over n samples)
  * P(correct | terminated) per prompt + distribution
  * in-zone achievability: prompts where terminated completions are a 2..6/8 mix

If P(correct|terminated) collapses to ~1.0 at every T, the checkpoint only
terminates when correct -> in-zone is unreachable by temperature alone. If a
higher T pushes it toward ~0.3..0.7, that T is the in-zone lever.

Within rules: honest EOS-terminated samples, real rewards, validator recomputes.
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("vllm_sweep")

CKPT = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/2a66ba495544fd75c2cd0fe373aa9ef46b4c6b45"
SOURCES = {"gsm8k", "augmented_gsm8k"}
N_PROMPTS = 16
N_SAMPLES = 80          # oversample to reliably collect >=8 terminated
MAX_TOKENS = 1536
TEMPS = [0.9, 1.2, 1.5]
SEED = 1234


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from reliquary.environment import load_environment

    env = load_environment("openmathinstruct")
    ds = env._dataset
    srcs = ds["problem_source"]
    idxs = [i for i, s in enumerate(srcs) if s in SOURCES]
    stride = max(1, len(idxs) // N_PROMPTS)
    chosen = idxs[::stride][:N_PROMPTS]
    problems = [env.get_problem(i) for i in chosen]
    logger.info("Picked %d prompts from %s", len(problems), SOURCES)

    tok = AutoTokenizer.from_pretrained(CKPT)
    prompt_token_ids = [tok.encode(p["prompt"], add_special_tokens=False) for p in problems]
    vllm_prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

    llm = LLM(model=CKPT, tokenizer=CKPT, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=2048,
              enable_prefix_caching=True, tensor_parallel_size=1, seed=SEED)

    results = {}
    for T in TEMPS:
        sp = SamplingParams(n=N_SAMPLES, temperature=T, top_p=1.0, top_k=-1,
                            max_tokens=MAX_TOKENS)
        outputs = llm.generate(vllm_prompts, sp)
        per_prompt = []
        tot_term = 0
        tot_corr_term = 0
        tot_samples = 0
        in_zone_first8 = 0      # first-8-terminated lands 2..6 correct (miner rule)
        in_zone_possible = 0    # >=8 terminated AND p_correct in [0.2,0.8]
        for p, out in zip(problems, outputs):
            term_correct = []   # 1/0 correctness of each terminated completion (in order)
            for comp in out.outputs:
                tot_samples += 1
                if comp.finish_reason == "stop":      # EOS-terminated
                    r = env.compute_reward(p, comp.text)
                    term_correct.append(1 if r >= 1.0 else 0)
            n_term = len(term_correct)
            n_corr = sum(term_correct)
            tot_term += n_term
            tot_corr_term += n_corr
            p_corr = (n_corr / n_term) if n_term else None
            first8 = term_correct[:8]
            first8_correct = sum(first8) if len(first8) == 8 else None
            if first8_correct is not None and 2 <= first8_correct <= 6:
                in_zone_first8 += 1
            if n_term >= 8 and p_corr is not None and 0.2 <= p_corr <= 0.8:
                in_zone_possible += 1
            per_prompt.append({
                "id": p["id"], "n_term": n_term, "n_corr_term": n_corr,
                "p_correct_given_term": round(p_corr, 3) if p_corr is not None else None,
                "first8_correct": first8_correct,
            })
        results[str(T)] = {
            "term_rate": round(tot_term / tot_samples, 4),
            "p_correct_given_term_overall": round(tot_corr_term / tot_term, 4) if tot_term else None,
            "prompts_with_ge8_term": sum(1 for x in per_prompt if x["n_term"] >= 8),
            "in_zone_first8": in_zone_first8,
            "in_zone_possible": in_zone_possible,
            "per_prompt": per_prompt,
        }
        logger.info("T=%.1f | term_rate=%.3f | P(correct|term)=%s | >=8term=%d/%d | in_zone_first8=%d | in_zone_possible=%d",
                    T, results[str(T)]["term_rate"],
                    results[str(T)]["p_correct_given_term_overall"],
                    results[str(T)]["prompts_with_ge8_term"], len(problems),
                    in_zone_first8, in_zone_possible)

    with open("/root/vllm_temp_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== vLLM TEMP SWEEP (terminated-only) ===")
    for T in TEMPS:
        r = results[str(T)]
        print(f"T={T}: term_rate={r['term_rate']} P(correct|term)={r['p_correct_given_term_overall']} "
              f">=8term={r['prompts_with_ge8_term']}/{len(problems)} "
              f"in_zone_first8={r['in_zone_first8']} in_zone_possible={r['in_zone_possible']}")
    logger.info("wrote /root/vllm_temp_sweep.json")


if __name__ == "__main__":
    main()
