#!/usr/bin/env python3
"""Replay KNOWN in-zone winner prompts on the current checkpoint.

winners.jsonl holds prompts that were in-zone (2..6/8 correct) when recorded,
with their reward_vector. 279/323 were 6/8 (upper edge). If the checkpoint has
since improved, those tip to 7-8/8 (out of zone). This regenerates on the deeper
in-zone winners (recorded 2..5/8 — most margin to stay in-zone) at the production
temperature and measures CURRENT correctness among terminated completions.

Verdict:
  * If many winners are STILL 2..6/8 now -> point the miner at these prompt_idxs
    and we can submit.
  * If they've shifted to 7-8/8 -> checkpoint advanced past its frontier;
    in-zone unreachable until a new checkpoint publishes.
"""

from __future__ import annotations

import os as _os
_os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import json
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("winners_replay")

CKPT = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/2a66ba495544fd75c2cd0fe373aa9ef46b4c6b45"
WINNERS = "/root/wf_data/winners.jsonl"
N_SAMPLES = 64
MAX_TOKENS = 1536
TEMP = 0.9     # production temperature
SEED = 1234
MAX_PROMPTS = 40


def rec_nc(rv: str) -> int:
    return sum(1 for c in str(rv) if c == "1")


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from reliquary.environment import load_environment

    env = load_environment("openmathinstruct")  # for compute_reward parity
    rows = [json.loads(l) for l in open(WINNERS)]
    for r in rows:
        r["_nc"] = rec_nc(r["reward_vector"])
    # Prioritise deeper in-zone winners (more margin), then fill with 5s/6s.
    rows.sort(key=lambda r: r["_nc"])  # 2s first
    chosen = rows[:MAX_PROMPTS]
    logger.info("Replaying %d winners | recorded nc hist=%s",
                len(chosen), dict(sorted(Counter(r["_nc"] for r in chosen).items())))

    tok = AutoTokenizer.from_pretrained(CKPT)
    vllm_prompts = [{"prompt_token_ids": tok.encode(r["prompt"], add_special_tokens=False)}
                    for r in chosen]

    llm = LLM(model=CKPT, tokenizer=CKPT, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=2048,
              enable_prefix_caching=True, tensor_parallel_size=1, seed=SEED)
    sp = SamplingParams(n=N_SAMPLES, temperature=TEMP, top_p=1.0, top_k=-1, max_tokens=MAX_TOKENS)
    outputs = llm.generate(vllm_prompts, sp)

    still_in_zone = 0
    flipped_high = 0      # >=7/8 now (out, top edge)
    too_few_term = 0
    per = []
    for r, out in zip(chosen, outputs):
        problem = {"prompt": r["prompt"], "ground_truth": r["ground_truth"]}
        tc = []
        for comp in out.outputs:
            if comp.finish_reason == "stop":
                tc.append(1 if env.compute_reward(problem, comp.text) >= 1.0 else 0)
        n_term = len(tc)
        first8 = tc[:8]
        cur_nc = sum(first8) if len(first8) == 8 else None
        p_corr = round(sum(tc) / n_term, 3) if n_term else None
        status = "few_term"
        if cur_nc is not None:
            if 2 <= cur_nc <= 6:
                status = "in_zone"; still_in_zone += 1
            elif cur_nc >= 7:
                status = "flipped_high"; flipped_high += 1
            else:
                status = "flipped_low"
        else:
            too_few_term += 1
        per.append({"idx": r.get("prompt_idx"), "src": r.get("source"),
                    "rec_nc": r["_nc"], "n_term": n_term,
                    "cur_first8_nc": cur_nc, "p_corr_term": p_corr, "status": status})
        logger.info("idx=%s src=%-15s rec=%d/8 -> n_term=%2d cur_first8=%s p_corr=%s [%s]",
                    r.get("prompt_idx"), r.get("source"), r["_nc"], n_term,
                    cur_nc, p_corr, status)

    out_summary = {
        "n_replayed": len(chosen),
        "still_in_zone_now": still_in_zone,
        "flipped_to_7or8": flipped_high,
        "too_few_terminated(<8)": too_few_term,
        "per_prompt": per,
    }
    with open("/root/winners_replay.json", "w") as f:
        json.dump(out_summary, f, indent=2)
    print("\n=== WINNERS REPLAY (current checkpoint, T=0.9) ===")
    print(f"replayed={len(chosen)}  STILL_IN_ZONE={still_in_zone}  "
          f"flipped_to_7-8/8={flipped_high}  too_few_term(<8)={too_few_term}")
    logger.info("wrote /root/winners_replay.json")


if __name__ == "__main__":
    main()
