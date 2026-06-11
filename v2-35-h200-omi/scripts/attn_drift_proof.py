#!/usr/bin/env python3
"""Measure GRAIL sketch + logprob drift between attention kernels.

The miner's PROOF model (HF) must reproduce the validator's forward pass
within tolerance. The validator recomputes with attn_implementation=
flash_attention_2 (docs/mining.md: "Do not override on mainnet"). This
script quantifies what happens if the miner's proof model uses sdpa or
eager instead: it loads the SAME checkpoint with each kernel over the
SAME fixed token sequence and compares, per position,

  * GRAIL sketch mod-diff vs adaptive_sketch_tolerance(pos, seq_len)
  * chosen-token logprob deviation dev = exp(|Δlogprob|) - 1 vs LOGPROB_IS_EPS

If miner(sdpa) drifts from validator(flash) beyond tolerance -> GRAIL_FAIL.
Only token IDs flow from vLLM; this concerns the HF proof load only.
"""

from __future__ import annotations

import gc
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reliquary.constants import (
    LAYER_INDEX,
    LOGPROB_IS_EPS,
    PRIME_Q,
)
from reliquary.protocol.grail_verifier import (
    GRAILVerifier,
    adaptive_sketch_tolerance,
)
from reliquary.shared.forward import forward_single_layer
from reliquary.shared.hf_compat import resolve_hidden_size

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("attn_drift")

CKPT = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/2a66ba495544fd75c2cd0fe373aa9ef46b4c6b45"
RANDOMNESS = "a" * 64
PROMPT = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May? Solve step by step and put the final answer in \\boxed{}."


def load(attn_impl):
    return (
        AutoModelForCausalLM.from_pretrained(
            CKPT, dtype=torch.bfloat16, attn_implementation=attn_impl
        )
        .to("cuda:0")
        .eval()
    )


def forward_proof(model, input_ids, attn_mask, r_vec):
    """Return (sketches:list[int], chosen_logprobs:list[float]) for the seq."""
    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    with torch.inference_mode():
        h, logits = forward_single_layer(model, input_ids, attn_mask, LAYER_INDEX)
    h_layer = h[0]  # (seq, hid)
    r_vec_dev = r_vec.to(h_layer.device)
    commits = verifier.create_commitments_batch(h_layer, r_vec_dev)
    sketches = [c["sketch"] for c in commits]

    # chosen-token logprob: logits[i] predicts token[i+1]
    lg = logits[0].float()  # (seq, vocab)
    logprobs = torch.log_softmax(lg, dim=-1)
    ids = input_ids[0]
    chosen = []
    for i in range(ids.shape[0] - 1):
        chosen.append(float(logprobs[i, ids[i + 1]].item()))
    return sketches, chosen


def mod_diff(a, b):
    d = abs(a - b)
    return min(d, PRIME_Q - d)


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)

    # 1) Get a realistic token sequence: greedy-generate once under flash.
    logger.info("Loading flash model to produce a fixed realistic sequence...")
    m = load("flash_attention_2")
    pid = torch.tensor([prompt_ids], device="cuda:0")
    with torch.inference_mode():
        gen = m.generate(pid, max_new_tokens=180, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    full_ids = gen[0].tolist()
    seq_len = len(full_ids)
    prompt_len = len(prompt_ids)
    logger.info("Fixed sequence: prompt_len=%d total=%d (completion=%d)",
                prompt_len, seq_len, seq_len - prompt_len)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda:0")
    attn_mask = torch.ones_like(input_ids)

    verifier = GRAILVerifier(hidden_dim=resolve_hidden_size(m))
    r_vec = verifier.generate_r_vec(RANDOMNESS)

    # flash result first (reuse loaded model)
    results = {}
    logger.info("Computing proof under flash_attention_2 ...")
    results["flash_attention_2"] = forward_proof(m, input_ids, attn_mask, r_vec)
    del m; gc.collect(); torch.cuda.empty_cache()

    for impl in ("sdpa", "eager"):
        logger.info("Loading + computing proof under %s ...", impl)
        m = load(impl)
        results[impl] = forward_proof(m, input_ids, attn_mask, r_vec)
        del m; gc.collect(); torch.cuda.empty_cache()

    # Compare every kernel vs flash (the validator's kernel).
    ref_sketch, ref_lp = results["flash_attention_2"]
    summary = {}
    # focus on COMPLETION positions (what the proof actually challenges)
    comp_lo = prompt_len
    for impl in ("sdpa", "eager"):
        sk, lp = results[impl]
        # sketch drift over completion positions
        sk_fail = 0
        sk_diffs = []
        for pos in range(comp_lo, seq_len):
            d = mod_diff(ref_sketch[pos], sk[pos])
            tol = adaptive_sketch_tolerance(pos, seq_len)
            sk_diffs.append(d)
            if d > tol:
                sk_fail += 1
        sk_diffs_sorted = sorted(sk_diffs)
        n = len(sk_diffs_sorted)
        # logprob deviation over completion positions (lp has seq_len-1 entries)
        devs = []
        for i in range(comp_lo, seq_len - 1):
            devs.append(torch.exp(torch.tensor(abs(ref_lp[i] - lp[i]))).item() - 1.0)
        devs_sorted = sorted(devs)
        m_ = len(devs_sorted)
        summary[f"{impl}_vs_flash"] = {
            "sketch_positions": n,
            "sketch_mean_diff": round(sum(sk_diffs) / n, 1),
            "sketch_p50": sk_diffs_sorted[n // 2],
            "sketch_p95": sk_diffs_sorted[min(n - 1, int(n * 0.95))],
            "sketch_max": sk_diffs_sorted[-1],
            "sketch_tol_at_last": adaptive_sketch_tolerance(seq_len - 1, seq_len),
            "sketch_positions_OVER_tol": sk_fail,
            "sketch_frac_fail": round(sk_fail / n, 4),
            "logprob_median_dev": round(devs_sorted[m_ // 2], 5),
            "logprob_p95_dev": round(devs_sorted[min(m_ - 1, int(m_ * 0.95))], 5),
            "logprob_max_dev": round(devs_sorted[-1], 5),
            "logprob_IS_EPS_threshold": LOGPROB_IS_EPS,
            "logprob_median_FAILS": devs_sorted[m_ // 2] > LOGPROB_IS_EPS,
        }

    print("\n=== ATTN KERNEL DRIFT vs flash_attention_2 (validator kernel) ===")
    print(json.dumps(summary, indent=2))
    with open("/root/attn_drift_proof.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote /root/attn_drift_proof.json")


if __name__ == "__main__":
    main()
