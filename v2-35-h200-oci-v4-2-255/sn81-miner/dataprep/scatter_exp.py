#!/usr/bin/env python3
"""Format-scatter feasibility experiment.

For scatterable-GT prompts, generate 8 rollouts at several sampling temps,
score with the EXACT env.compute_reward, and ALSO compute the validator's
per-token chosen-probability at T_PROTO=0.9 to check the distribution floor
(median>=0.30, q10>=0.025) and the boxed-answer floor (>=0.001).

Reports per temp:
  * in-zone rate (group with 2..6 of 8 correct, sigma>=0.43)
  * of those in-zone groups, how many would PASS the distribution floor on
    ALL 8 rollouts (the gate that the scatter mechanism risks tripping)
  * scatter diagnostics: did the wrong rollouts fail via parser-equivalent
    alt-form (the honest scatter) vs genuine math error vs non-termination
"""
from __future__ import annotations
import os as _os
_os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import json, re, logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("scatter")

CKPT = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/2a66ba495544fd75c2cd0fe373aa9ef46b4c6b45"
T_PROTO = 0.9
SAMPLING_MEDIAN_LOW_MAX = 0.30
SAMPLING_LOW_Q10_MAX = 0.025
SAMPLING_MIN_STEPS = 30
BOXED_MIN = 0.001
EOS_IDS = {151643, 151645}
N = 8
MAX_TOKENS = 2048
TEMPS = [0.9, 1.2]
SEED = 1234
N_PROMPTS_PER_TYPE = 12


def last_boxed(text):
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx < 0:
        return None
    try:
        o = text.index("{", idx)
    except ValueError:
        return None
    d = 0
    for j in range(o, len(text)):
        if text[j] == "{":
            d += 1
        elif text[j] == "}":
            d -= 1
            if d == 0:
                return text[o+1:j]
    return None


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from reliquary.environment import load_environment
    from reliquary.environment.openmathinstruct import _normalize_answer

    env = load_environment("openmathinstruct")
    idxs = json.load(open("/root/scatter_idxs.json"))
    chosen = []  # (type, problem_idx, problem)
    for typ, lst in idxs.items():
        for pi in lst[:N_PROMPTS_PER_TYPE]:
            chosen.append((typ, pi, env.get_problem(pi)))
    log.info("prompts: %d", len(chosen))

    tok = AutoTokenizer.from_pretrained(CKPT)
    vllm_prompts = [{"prompt_token_ids": tok.encode(p["prompt"], add_special_tokens=False)}
                    for _, _, p in chosen]

    llm = LLM(model=CKPT, tokenizer=CKPT, dtype="bfloat16",
              gpu_memory_utilization=0.22, max_model_len=3072,
              enable_prefix_caching=True, tensor_parallel_size=1, seed=SEED)

    # HF model for validator-faithful chosen-prob @ T_PROTO
    hf = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda:0").eval()

    results = {}
    for T in TEMPS:
        sp = SamplingParams(n=N, temperature=T, top_p=1.0, top_k=-1,
                            max_tokens=MAX_TOKENS, seed=SEED)
        outs = llm.generate(vllm_prompts, sp)
        in_zone = 0
        in_zone_floor_ok = 0
        scatter_groups = 0  # in-zone where >=1 wrong rollout is parser-equiv alt-form
        per = []
        for (typ, pi, p), out in zip(chosen, outs):
            gt_norm = _normalize_answer(p["ground_truth"])
            recs = []
            for comp in out.outputs:
                txt = comp.text
                terminated = comp.finish_reason == "stop"
                r = env.compute_reward(p, txt)
                boxed = last_boxed(txt)
                cand_norm = _normalize_answer(boxed) if boxed else None
                # scatter classification: wrong but boxed something non-empty
                # whose stripped numeric core equals gt's numeric core
                alt_form = False
                if r < 1.0 and cand_norm:
                    a = re.sub(r"[^0-9]", "", cand_norm)
                    b = re.sub(r"[^0-9]", "", gt_norm)
                    if a and a == b and cand_norm != gt_norm:
                        alt_form = True
                recs.append({"r": r, "term": terminated, "boxed": boxed,
                             "cand": cand_norm, "alt": alt_form,
                             "ids": comp.token_ids})
            ncorr = sum(1 for x in recs if x["r"] >= 1.0)
            rewards = [x["r"] for x in recs]
            mean = sum(rewards)/len(rewards)
            sigma = (sum((x-mean)**2 for x in rewards)/len(rewards))**0.5
            iz = (2 <= ncorr <= 6) and sigma >= 0.43
            floor_ok = None
            if iz:
                in_zone += 1
                n_alt = sum(1 for x in recs if x["alt"])
                if n_alt >= 1:
                    scatter_groups += 1
                # validator-faithful distribution floor on ALL 8 rollouts @ T_PROTO
                floor_ok = True
                for x in recs:
                    ids = list(x["ids"])
                    if len(ids) < SAMPLING_MIN_STEPS:
                        floor_ok = False; break
                    full = tok.encode(p["prompt"], add_special_tokens=False) + ids
                    inp = torch.tensor([full], device="cuda:0")
                    with torch.no_grad():
                        logits = hf(inp).logits[0]  # [L, V]
                    plen = len(full) - len(ids)
                    # chosen prob for each completion token t at pos t-1, scaled by T_PROTO
                    probs_chosen = []
                    for j, t in enumerate(ids):
                        pos = plen + j - 1
                        if pos < 0:
                            continue
                        pr = torch.softmax(logits[pos].float()/T_PROTO, dim=-1)
                        probs_chosen.append(float(pr[t]))
                    arr = np.array(probs_chosen)
                    med = float(np.median(arr)); q10 = float(np.quantile(arr, 0.10))
                    if med < SAMPLING_MEDIAN_LOW_MAX or q10 < SAMPLING_LOW_Q10_MAX:
                        floor_ok = False; break
                    # EOS p_stop
                    if terminated:
                        last_pos = len(full) - 2
                        prs = torch.softmax(logits[last_pos].float(), dim=-1)
                        pstop = float(sum(prs[e] for e in EOS_IDS))
                        if pstop < 0.01:
                            floor_ok = False; break
                if floor_ok:
                    in_zone_floor_ok += 1
            per.append({"type": typ, "pi": pi, "ncorr": ncorr, "sigma": round(sigma,3),
                        "in_zone": iz, "floor_ok": floor_ok,
                        "n_alt": sum(1 for x in recs if x["alt"]),
                        "n_term": sum(1 for x in recs if x["term"])})
        results[str(T)] = {
            "in_zone": in_zone, "in_zone_floor_ok": in_zone_floor_ok,
            "scatter_groups": scatter_groups, "n_prompts": len(chosen),
            "per": per,
        }
        log.info("T=%.1f in_zone=%d/%d floor_ok=%d scatter=%d",
                 T, in_zone, len(chosen), in_zone_floor_ok, scatter_groups)
        # per-type breakdown
        from collections import Counter
        izt = Counter(x["type"] for x in per if x["in_zone"])
        fok = Counter(x["type"] for x in per if x["floor_ok"])
        log.info("  in_zone by type: %s | floor_ok by type: %s", dict(izt), dict(fok))

    json.dump(results, open("/root/scatter_results.json", "w"), indent=2, default=str)
    print("\n=== FORMAT-SCATTER FEASIBILITY ===")
    for T in TEMPS:
        r = results[str(T)]
        print(f"T={T}: in_zone={r['in_zone']}/{r['n_prompts']} "
              f"floor_ok(accepted)={r['in_zone_floor_ok']} scatter_attributed={r['scatter_groups']}")


if __name__ == "__main__":
    main()
