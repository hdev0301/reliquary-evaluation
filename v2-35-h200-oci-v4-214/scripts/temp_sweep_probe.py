#!/usr/bin/env python3
"""Temperature sweep: does higher sampling diversity break the 8/8 bimodality?

The converged checkpoint solves our sampled prompts 8/8 at T_PROTO=0.9, so no
group lands in the in-zone band (2-6 of 8 correct, sigma>=0.43). This probe
generates 8 completions per prompt at several temperatures and grades each with
the SAME env.compute_reward the validator re-runs, reporting the per-prompt
correct-count histogram and in-zone yield at each temperature.

Within rules: honest EOS-terminated samples, real rewards; only the sampling
temperature changes. The validator recomputes rewards + logprobs itself.
"""

from __future__ import annotations

import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reliquary.environment import load_environment

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("temp_sweep")

CKPT = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/2a66ba495544fd75c2cd0fe373aa9ef46b4c6b45"
SOURCES = {"gsm8k", "augmented_gsm8k"}   # shorter completions -> less truncation noise
N_PROMPTS = 16
N_SAMPLES = 8
MAX_NEW = 768
TEMPS = [0.9, 1.1, 1.3, 1.5]
EOS_IDS = {151643, 151645}
SEED = 1234


def pick_prompts(env):
    ds = env._dataset
    srcs = ds["problem_source"]
    idxs = [i for i, s in enumerate(srcs) if s in SOURCES]
    # deterministic stride sample across the filtered pool
    stride = max(1, len(idxs) // N_PROMPTS)
    chosen = idxs[::stride][:N_PROMPTS]
    return [env.get_problem(i) for i in chosen]


def main():
    torch.manual_seed(SEED)
    logger.info("Loading env...")
    env = load_environment("openmathinstruct")
    problems = pick_prompts(env)
    logger.info("Picked %d prompts from %s", len(problems), SOURCES)

    tok = AutoTokenizer.from_pretrained(CKPT)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda:0").eval()

    results = {}
    for T in TEMPS:
        hist = {k: 0 for k in range(N_SAMPLES + 1)}   # n_correct -> #prompts
        in_zone = 0
        term_frac_sum = 0.0
        per_prompt = []
        for p in problems:
            ids = tok(p["prompt"], return_tensors="pt", add_special_tokens=False).to("cuda:0")
            with torch.inference_mode():
                out = model.generate(
                    **ids,
                    do_sample=True,
                    temperature=T,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=MAX_NEW,
                    num_return_sequences=N_SAMPLES,
                    pad_token_id=tok.pad_token_id,
                )
            prompt_len = ids["input_ids"].shape[1]
            gen = out[:, prompt_len:]
            n_correct = 0
            n_term = 0
            for row in gen:
                row_list = row.tolist()
                terminated = any(e in row_list for e in EOS_IDS)
                if terminated:
                    n_term += 1
                text = tok.decode(row, skip_special_tokens=True)
                r = env.compute_reward(p, text)
                if r >= 1.0:
                    n_correct += 1
            hist[n_correct] += 1
            term_frac_sum += n_term / N_SAMPLES
            if 2 <= n_correct <= 6:
                in_zone += 1
            per_prompt.append({"id": p["id"], "n_correct": n_correct, "n_term": n_term})
        results[str(T)] = {
            "in_zone_prompts": in_zone,
            "in_zone_frac": round(in_zone / len(problems), 3),
            "correct_hist": hist,
            "avg_term_frac": round(term_frac_sum / len(problems), 3),
            "per_prompt": per_prompt,
        }
        logger.info("T=%.1f -> in_zone=%d/%d hist=%s avg_term=%.2f",
                    T, in_zone, len(problems), hist, term_frac_sum / len(problems))

    with open("/root/temp_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== TEMPERATURE SWEEP (in-zone = 2..6 of 8 correct) ===")
    for T in TEMPS:
        r = results[str(T)]
        print(f"T={T}: in_zone={r['in_zone_prompts']}/{len(problems)} "
              f"hist={r['correct_hist']} avg_term={r['avg_term_frac']}")
    logger.info("wrote /root/temp_sweep.json")


if __name__ == "__main__":
    main()
