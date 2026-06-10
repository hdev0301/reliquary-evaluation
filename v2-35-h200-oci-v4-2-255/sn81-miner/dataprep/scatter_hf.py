#!/usr/bin/env python3
"""HF-only format-scatter experiment (no vLLM, avoids GPU contention).

Sample 8 rollouts/prompt at each T via HF generate, score with env reward,
and compute the validator-faithful chosen-prob @ T_PROTO=0.9 in the SAME
forward to check the distribution floor + boxed floor. Reuses one model.
"""
from __future__ import annotations
import json, re, logging
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("scatter_hf")

CKPT = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/2a66ba495544fd75c2cd0fe373aa9ef46b4c6b45"
T_PROTO = 0.9
MED_FLOOR, Q10_FLOOR, MIN_STEPS = 0.30, 0.025, 30
EOS_IDS = {151643, 151645}
N = 8
MAX_NEW = 1024
TEMPS = [0.9, 1.2]
SEED = 1234
NPER = 10


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
    from reliquary.environment import load_environment
    from reliquary.environment.openmathinstruct import _normalize_answer
    env = load_environment("openmathinstruct")
    idxs = json.load(open("/root/scatter_idxs.json"))
    chosen = []
    for typ, lst in idxs.items():
        for pi in lst[:NPER]:
            chosen.append((typ, pi, env.get_problem(pi)))
    log.info("prompts: %d", len(chosen))

    tok = AutoTokenizer.from_pretrained(CKPT)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to("cuda:0").eval()

    results = {}
    for T in TEMPS:
        torch.manual_seed(SEED)
        in_zone = floor_ok_cnt = scatter_groups = 0
        per = []
        for typ, pi, p in chosen:
            gt_norm = _normalize_answer(p["ground_truth"])
            ids = tok(p["prompt"], return_tensors="pt", add_special_tokens=False).to("cuda:0")
            plen = ids["input_ids"].shape[1]
            with torch.inference_mode():
                out = model.generate(
                    **ids, do_sample=True, temperature=T, top_p=1.0, top_k=0,
                    max_new_tokens=MAX_NEW, num_return_sequences=N,
                    pad_token_id=tok.pad_token_id,
                )
            recs = []
            for row in out:
                comp_ids = row[plen:].tolist()
                # trim at first EOS (honest miner truncates there)
                terminated = False
                for k, t in enumerate(comp_ids):
                    if t in EOS_IDS:
                        comp_ids = comp_ids[:k+1]
                        terminated = True
                        break
                text = tok.decode(comp_ids, skip_special_tokens=True)
                r = env.compute_reward(p, text)
                boxed = last_boxed(text)
                cand = _normalize_answer(boxed) if boxed else None
                alt = False
                if r < 1.0 and cand:
                    a = re.sub(r"[^0-9]", "", cand); b = re.sub(r"[^0-9]", "", gt_norm)
                    if a and a == b and cand != gt_norm:
                        alt = True
                recs.append({"r": r, "term": terminated, "ids": comp_ids, "alt": alt})
            ncorr = sum(1 for x in recs if x["r"] >= 1.0)
            rew = [x["r"] for x in recs]; m = sum(rew)/len(rew)
            sig = (sum((x-m)**2 for x in rew)/len(rew))**0.5
            iz = (2 <= ncorr <= 6) and sig >= 0.43
            floor_ok = None
            if iz:
                in_zone += 1
                if any(x["alt"] for x in recs):
                    scatter_groups += 1
                floor_ok = True
                worst_med = 1.0; worst_q10 = 1.0
                for x in recs:
                    cids = x["ids"]
                    if len(cids) < MIN_STEPS:
                        floor_ok = False; break
                    full = ids["input_ids"][0].tolist() + cids
                    inp = torch.tensor([full], device="cuda:0")
                    with torch.inference_mode():
                        lg = model(inp).logits[0]
                    ch = []
                    for j, t in enumerate(cids):
                        pos = plen + j - 1
                        pr = torch.softmax(lg[pos].float()/T_PROTO, dim=-1)
                        ch.append(float(pr[t]))
                    arr = np.array(ch)
                    med = float(np.median(arr)); q10 = float(np.quantile(arr, 0.10))
                    worst_med = min(worst_med, med); worst_q10 = min(worst_q10, q10)
                    if med < MED_FLOOR or q10 < Q10_FLOOR:
                        floor_ok = False; break
                if floor_ok:
                    floor_ok_cnt += 1
            per.append({"type": typ, "pi": pi, "ncorr": ncorr, "sigma": round(sig,3),
                        "in_zone": iz, "floor_ok": floor_ok,
                        "n_alt": sum(1 for x in recs if x["alt"]),
                        "n_term": sum(1 for x in recs if x["term"]),
                        "worst_med": round(worst_med,3) if iz else None,
                        "worst_q10": round(worst_q10,4) if iz else None})
        results[str(T)] = {"in_zone": in_zone, "floor_ok": floor_ok_cnt,
                           "scatter_groups": scatter_groups, "n_prompts": len(chosen), "per": per}
        from collections import Counter
        log.info("T=%.1f in_zone=%d/%d floor_ok=%d scatter=%d", T, in_zone, len(chosen), floor_ok_cnt, scatter_groups)
        log.info("  in_zone by type: %s", dict(Counter(x['type'] for x in per if x['in_zone'])))
        log.info("  floor_ok by type: %s", dict(Counter(x['type'] for x in per if x['floor_ok'])))
        log.info("  alt-form present by type: %s", dict(Counter(x['type'] for x in per if x['n_alt']>0)))
    json.dump(results, open("/root/scatter_hf_results.json","w"), indent=2, default=str)
    log.info("wrote /root/scatter_hf_results.json")


if __name__ == "__main__":
    main()
