"""Empirical σ-by-source measurement on the LIVE validator checkpoint.

Answers the core mining question for a well-trained policy: on which
OpenMathInstruct-2 `problem_source` buckets does Qwen3.5 (checkpoint 923)
actually sit at its learning frontier — i.e. naturally produce a mix of
correct AND incorrect EOS-terminated answers so a σ≥0.43 group is formable?

Generates at the protocol sampling params (T=0.9, top_p=1.0, full vocab) via
the SAME vLLM worker the miner uses, grades with the validator's own
`_compute_omi_reward`, and reports per-source the fraction of prompts that are
"minable" (≥2 natural-EOS fails AND ≥2 natural-EOS passes among ≥8 EOS rollouts
→ an 8-subset with k∈[2,6] exists).
"""

from __future__ import annotations

import collections
import os
import random
import shutil
import sys
import time

REPO_ID = os.environ.get("MEAS_REPO_ID", "R0mAI/reliquary-sn-v23")
REV = os.environ.get("MEAS_REV", "7f0a568d468652681259352c6cbec1ec7d53fd01")
N_GEN = int(os.environ.get("MEAS_N", "24"))
PER_SRC = int(os.environ.get("MEAS_PER_SRC", "30"))
MAX_TOK = int(os.environ.get("MEAS_MAX_TOK", "2750"))
SOURCES = tuple(s for s in os.environ.get("MEAS_SOURCES", "gsm8k,augmented_gsm8k,math,augmented_math").split(",") if s)
os.environ.setdefault("RELIQUARY_OMI_SHARDS", "4")
os.environ.setdefault("RELIQUARY_VLLM_GPU_MEM_UTIL", "0.55")
os.environ.setdefault("RELIQUARY_VLLM_MAX_NUM_SEQS", "384")


def _popstd(xs):
    if not xs:
        return 0.0
    mu = sum(xs) / len(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5


def best_sigma_8(rewards, eos_flags):
    """Max σ over any 8-subset of the EOS-terminated rollouts (≤1 non-EOS ok)."""
    eos_idx = [i for i, e in enumerate(eos_flags) if e]
    if len(eos_idx) < 8:
        # allow backfilling exactly one truncated rollout (validator allows ≤1)
        non = [i for i, e in enumerate(eos_flags) if not e][:1]
        pool = eos_idx + non
    else:
        pool = eos_idx
    if len(pool) < 8:
        return 0.0, 0, 0
    pool.sort(key=lambda i: rewards[i])  # low→high reward
    best = -1.0
    for j in range(0, 9):  # j fails (lowest) + (8-j) passes (highest)
        sel = pool[:j] + pool[len(pool) - (8 - j):]
        if len(set(sel)) != 8:
            continue
        s = _popstd([rewards[i] for i in sel])
        best = max(best, s)
    pass_eos = sum(1 for i in eos_idx if rewards[i] >= 0.5)
    fail_eos = sum(1 for i in eos_idx if rewards[i] < 0.5)
    return max(best, 0.0), pass_eos, fail_eos


def main():
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.environment.openmathinstruct import (
        OpenMathInstructEnvironment,
        _compute_omi_reward,
    )
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import (
        load_text_generation_model,
        load_tokenizer,
        resolve_eos_token_ids,
    )
    from mining.common.config import MinerConfig
    from mining.common.vllm_generator import make_generator

    print(f"[meas] downloading {REPO_ID}@{REV[:8]} ...", flush=True)
    path = snapshot_download(REPO_ID, revision=REV)
    for fn in ("preprocessor_config.json", "video_preprocessor_config.json"):
        dst = os.path.join(path, fn)
        if not os.path.exists(dst):
            try:
                shutil.copy(hf_hub_download("Qwen/Qwen3.5-4B", filename=fn), dst)
            except Exception as e:
                print(f"[meas] stage warn {fn}: {e}", flush=True)

    tok = load_tokenizer(path)
    print("[meas] loading HF model briefly for canonical EOS set ...", flush=True)
    hf = (
        load_text_generation_model(
            path, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION
        )
        .to("cuda:0")
        .eval()
    )
    eos = sorted(set(resolve_eos_token_ids(hf, tok)))
    print(f"[meas] eos ids = {eos}", flush=True)
    del hf
    torch.cuda.empty_cache()

    cfg = MinerConfig()
    gen = make_generator(cfg, eos, repo_dir=os.getcwd())
    gen.load(path)

    env = OpenMathInstructEnvironment()
    ds = env._dataset
    n_total = len(ds)
    print(f"[meas] dataset rows = {n_total}", flush=True)

    sources = SOURCES
    pools = collections.defaultdict(list)  # source -> [idx]
    rng = random.Random(1234)
    tries = 0
    need = {s: PER_SRC for s in sources}
    while any(len(pools[s]) < need[s] for s in sources) and tries < n_total:
        tries += 1
        idx = rng.randrange(n_total)
        src = str(ds[idx]["problem_source"])
        if src in pools and len(pools[src]) < need[src]:
            pools[src].append(idx)
    for s in sources:
        print(f"[meas] {s}: {len(pools[s])} prompts (after {tries} samples)", flush=True)

    report = {}
    for src in sources:
        idxs = pools[src]
        if not idxs:
            continue
        prompts = []
        gts = []
        for i in idxs:
            p = env.get_problem(i)
            prompts.append(encode_prompt(tok, p["prompt"]))
            gts.append(str(p.get("ground_truth", "")))
        t0 = time.time()
        groups = gen.generate_groups(prompts, n=N_GEN, max_tokens=MAX_TOK)
        dt = time.time() - t0

        minable = 0
        sig_list = []
        passrate_list = []
        eosfrac_list = []
        any_fail = 0
        fail_eos_hist = collections.Counter()
        comp_lens = []          # completion length of every rollout
        eos_comp_lens = []      # length of EOS-terminated rollouts only
        trunc_fail = 0          # wrong AND non-EOS (the unusable failures)
        eos_fail = 0            # wrong AND EOS (the usable failures)
        n_roll = 0
        for g, gt in zip(groups, gts):
            comps = [tok.decode(r.tokens[r.prompt_length:]) for r in g]
            rewards = [_compute_omi_reward({"ground_truth": gt}, c) for c in comps]
            eos_flags = [r.finished_with_eos for r in g]
            for r, rew, e in zip(g, rewards, eos_flags):
                L = len(r.tokens) - r.prompt_length
                comp_lens.append(L)
                n_roll += 1
                if e:
                    eos_comp_lens.append(L)
                if rew < 0.5:
                    if e:
                        eos_fail += 1
                    else:
                        trunc_fail += 1
            sig, pass_eos, fail_eos = best_sigma_8(rewards, eos_flags)
            sig_list.append(sig)
            passrate_list.append(sum(rewards) / len(rewards))
            eosfrac_list.append(sum(eos_flags) / len(eos_flags))
            fail_eos_hist[min(fail_eos, 6)] += 1
            if fail_eos >= 1:
                any_fail += 1
            # minable: an 8-subset with k∈[2,6] all-EOS exists
            if pass_eos >= 2 and fail_eos >= 2 and (pass_eos + fail_eos) >= 8:
                minable += 1
        m = len(idxs)

        def _pct(xs, q):
            if not xs:
                return 0
            s = sorted(xs)
            return s[min(len(s) - 1, int(q * len(s)))]

        report[src] = dict(
            n=m,
            minable=minable,
            minable_frac=round(minable / m, 3),
            any_natural_fail_frac=round(any_fail / m, 3),
            mean_accuracy=round(sum(passrate_list) / m, 3),
            mean_eos_frac=round(sum(eosfrac_list) / m, 3),
            mean_best_sigma=round(sum(sig_list) / m, 3),
            max_best_sigma=round(max(sig_list), 3),
            fail_eos_hist=dict(sorted(fail_eos_hist.items())),
            trunc_fail_frac=round(trunc_fail / max(1, n_roll), 3),
            eos_fail_frac=round(eos_fail / max(1, n_roll), 3),
            eos_len_p50=_pct(eos_comp_lens, 0.50),
            eos_len_p90=_pct(eos_comp_lens, 0.90),
            len_p90=_pct(comp_lens, 0.90),
            len_p99=_pct(comp_lens, 0.99),
            gen_s=round(dt, 1),
        )
        print(f"[meas] {src}: {report[src]}", flush=True)

    import json as _json

    lines = []
    lines.append("================ SUMMARY (N_gen=%d, per_src=%d, max_tok=%d) ================" % (N_GEN, PER_SRC, MAX_TOK))
    lines.append("source            minable%  anyfail%  acc    eos%   meanσ  maxσ   fail_eos_hist")
    for src in sources:
        r = report.get(src)
        if not r:
            continue
        lines.append(
            f"{src:18s} {r['minable_frac']*100:5.1f}   {r['any_natural_fail_frac']*100:5.1f}   "
            f"{r['mean_accuracy']:.2f}  {r['mean_eos_frac']*100:4.0f}  {r['mean_best_sigma']:.3f}  "
            f"{r['max_best_sigma']:.3f}  {r['fail_eos_hist']}"
        )
    lines.append("=" * 80)
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    with open("/root/meas_result.txt", "w") as f:
        f.write(text + "\n\n" + _json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
