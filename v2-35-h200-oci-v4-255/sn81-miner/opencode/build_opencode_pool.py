#!/usr/bin/env python3
"""Empirical in-zone pool builder for the OPENCODE env.

Unlike the math builders (build_inzone_v2.py), opencode curatability CANNOT be
read off the prompt: the reward is passed/total over HIDDEN unit tests, so a
prompt is "in-zone-feasible" only if the model SCATTERS on it (some of M
rollouts pass, some fail) -> a group can be curated to k correct + (8-k) wrong
with sigma >= SIGMA_MIN (0.43). This script finds those prompts empirically:

  1. Resolve each candidate prompt's STRUCTURED CASES (validator-replica):
       reconstruct = stream the PUBLIC nvidia/OpenCodeInstruct, run the in-repo
       build pipeline (build_opencodeinstruct_subset.process_row), join on `id`
       to the prompt mirror (R0mAI/opencodeinstruct-prompts). This reproduces
       the validator's cases byte-exact (see opencode/verify_opencode_gate.py).
       Cached to data/oci_cases_cache.json so it's one-time.
       subset      = read structured_cases straight from a structured subset you
       have read access to (--cases-source subset, fast path).
  2. Generate M completions/prompt with vLLM (Qwen3.5, thinking disabled,
     protocol sampling), extract python (env._extract_python), grade locally
     against the cases with the SAME call/compare semantics as the validator
     grader.
  3. KEEP prompts whose fraction-correct is intermediate (p_low <= f <= p_high):
     not pure 0/M (too hard) and not pure M/M (bimodal trap) -> curatable.

OUTPUT (under data/): inzone_pool_opencode.json   (sorted prompt_idx list)
  Consumed by run_miner.sh's inzone_pool_*.json rotation. MINE IT WITH THE
  OPENCODE ENV:  RELIQUARY_ENVIRONMENTS=opencodeinstruct RELIQUARY_OCI_PROMPT_ONLY=1
  (the indices are positions in the prompt mirror, which the miner loads).

Run (on the box):
  cd /root/reliquary && .venv/bin/python /root/sn81-miner/opencode/build_opencode_pool.py \
      --max-candidates 3000 --m 16

SECURITY: local grading exec's MODEL-generated code in a subprocess pool with a
SIGALRM timeout — NOT a gVisor sandbox. Fine for screening YOUR OWN model output
on YOUR OWN box. Use --use-grader to route through a running grader server
(GraderClient over GRADER_SOCKET_PATH) for sandboxed, exact-parity grading.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/root/reliquary")
sys.path.insert(0, "/root/reliquary/scripts")

from reliquary.constants import (  # noqa: E402
    T_PROTO, TOP_P_PROTO, TOP_K_PROTO, M_ROLLOUTS, SIGMA_MIN,
    GRADER_EVAL_TIMEOUT_SECONDS, DEFAULT_BASE_MODEL,
)
from reliquary.environment.opencodeinstruct import (  # noqa: E402
    OpenCodeInstructEnvironment, _extract_python,
)
from reliquary.protocol.tokens import encode_prompt  # noqa: E402

DATA = os.environ.get("RELIQUARY_DATA_DIR", "/root/sn81-miner/data")
DIAG = os.environ.get("RELIQUARY_DIAG_DIR", "/root/sn81-miner/diagnostics")


# ---------------------------------------------------------------------------
# Local grading — mirrors the validator grader's call/compare semantics
# (reliquary/environment/grader/worker.py + build_opencodeinstruct_subset).
# ---------------------------------------------------------------------------

def _eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) or isinstance(b, float):
            return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _grade_one(task):
    """Run one (code, cases) and return passed/total in [0,1]. Worker-process safe.

    SIGALRM bounds wall time; exec failures and per-case errors score 0 for that
    case (never raise). Network/FS are NOT sandboxed here — see module docstring.
    """
    code, cases, timeout = task
    if not cases:
        return 0.0
    import signal

    def _timeout(signum, frame):
        raise TimeoutError()

    passed = 0
    ns: dict = {}
    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            exec(code, ns)  # noqa: S102 - intentional, see docstring
        except Exception:
            pass
        for c in cases:
            try:
                e = c["entry"]
                if e["kind"] == "function":
                    fn = ns[e["name"]]
                else:
                    fn = getattr(ns[e["class_name"]](), e["method"])
                out = fn(*c.get("args", []), **c.get("kwargs", {}))
                if _eq(out, c.get("expected")):
                    passed += 1
            except Exception:
                pass
    except TimeoutError:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    total = len(cases)
    return passed / total if total else 0.0


# ---------------------------------------------------------------------------
# Case resolution
# ---------------------------------------------------------------------------

def resolve_cases(cand_idx, mirror, args):
    """Return {idx: {"prompt": str, "cases": list}} for candidate indices.

    None-valued entries mark indices the validator filter would DROP (no valid
    structured cases) — they are excluded from screening.
    """
    if args.cases_source == "subset":
        # Validator mode: structured_cases are in the loaded dataset itself.
        env = OpenCodeInstructEnvironment()
        ds = env._dataset
        out = {}
        for idx in cand_idx:
            row = ds[idx % len(ds)]
            raw = row.get("structured_cases")
            cases = json.loads(raw) if isinstance(raw, str) else (raw or [])
            out[idx] = {"prompt": row["input"], "cases": cases} if cases else None
        return out

    # reconstruct from public nvidia/OpenCodeInstruct, cached
    cache_path = os.path.join(DATA, "oci_cases_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    id_to_idx = {}
    for idx in cand_idx:
        if str(idx) in cache:
            continue
        id_to_idx[mirror[idx]["id"]] = idx
    if id_to_idx:
        from build_opencodeinstruct_subset import process_row
        import datasets as hf
        targets = set(id_to_idx)
        print(f"reconstructing cases for {len(targets)} ids (streaming nvidia/OpenCodeInstruct, max_scan={args.max_scan})...")
        it = hf.load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
        for i, row in enumerate(it):
            if i >= args.max_scan or not targets:
                break
            rid = row.get("id")
            if rid not in targets:
                continue
            targets.discard(rid)
            idx = id_to_idx[rid]
            proc = process_row(row)
            if proc is None:
                cache[str(idx)] = None
            else:
                cache[str(idx)] = {"prompt": proc["input"], "cases": json.loads(proc["structured_cases"])}
            if (len(id_to_idx) - len(targets)) % 200 == 0:
                print(f"  resolved {len(id_to_idx) - len(targets)}/{len(id_to_idx)} (scanned {i})")
        if targets:
            print(f"  WARNING: {len(targets)} ids not found within max_scan; raise --max-scan")
        os.makedirs(DATA, exist_ok=True)
        json.dump(cache, open(cache_path, "w"))
    return {idx: cache.get(str(idx)) for idx in cand_idx}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT_PATH") or DEFAULT_BASE_MODEL)
    ap.add_argument("--cases-source", choices=["reconstruct", "subset"], default="reconstruct")
    ap.add_argument("--max-candidates", type=int, default=3000, help="how many mirror prompts to screen")
    ap.add_argument("--idx-file", default=None, help="re-screen/refine an existing pool json instead of random sampling")
    ap.add_argument("--m", type=int, default=max(16, M_ROLLOUTS), help="rollouts per prompt (oversample for screen)")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--p-low", type=float, default=0.06, help="min fraction-correct to keep (>0 => not 0/M)")
    ap.add_argument("--p-high", type=float, default=0.94, help="max fraction-correct to keep (<1 => not M/M)")
    ap.add_argument("--strict-zone", action="store_true",
                    help="tighten the band to k in [2,6]/8 (sigma>=SIGMA_MIN) instead of the wide screen band")
    ap.add_argument("--gpu-mem-util", type=float, default=0.65)
    ap.add_argument("--max-scan", type=int, default=2_000_000)
    ap.add_argument("--grade-workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--use-grader", action="store_true", help="grade via a running grader server (sandboxed parity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA, "inzone_pool_opencode.json"))
    args = ap.parse_args()

    if args.strict_zone:
        args.p_low, args.p_high = 0.25, 0.75  # k in [2,6]/8 -> sigma>=0.433

    import random
    random.seed(args.seed)

    # --- candidate indices into the prompt mirror -------------------------
    import datasets as hf
    repo = OpenCodeInstructEnvironment._DEFAULT_PROMPT_REPO
    rev = OpenCodeInstructEnvironment._DEFAULT_PROMPT_REVISION
    print(f"loading prompt mirror {repo}@{rev[:8]} ...")
    mirror = hf.load_dataset(repo, revision=rev, split="train")
    N = len(mirror)
    if args.idx_file:
        cand = [i for i in json.load(open(args.idx_file)) if 0 <= i < N]
    else:
        cand = list(range(N))
    if len(cand) > args.max_candidates:
        cand = sorted(random.sample(cand, args.max_candidates))
    print(f"mirror rows={N}; screening {len(cand)} candidates; M={args.m}; band=[{args.p_low},{args.p_high}]")

    # --- resolve structured cases ----------------------------------------
    cmap = resolve_cases(cand, mirror, args)
    usable = [i for i in cand if cmap.get(i) and cmap[i].get("cases")]
    print(f"usable (have structured cases): {len(usable)}/{len(cand)}")
    if not usable:
        print("no usable candidates — check --cases-source / --max-scan / HF access")
        return

    # --- generate M completions per prompt with vLLM ----------------------
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    try:
        from vllm.inputs import TokensPrompt
    except Exception:
        TokensPrompt = None

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    token_ids = [encode_prompt(tok, cmap[i]["prompt"]) for i in usable]
    sp = SamplingParams(
        n=args.m, temperature=T_PROTO, top_p=TOP_P_PROTO,
        top_k=(TOP_K_PROTO if TOP_K_PROTO and TOP_K_PROTO > 0 else -1),
        max_tokens=args.max_new_tokens,
    )
    # Qwen3.5 is a MULTIMODAL-wrapper checkpoint used text-only; the snapshot has no
    # image processor. limit_mm_per_prompt={image:0,video:0} makes vLLM skip loading it
    # and run the LM only (mirrors miner vllm_backend._multimodal_kwargs). Without it:
    # OSError "Can't load image processor". enforce_eager already set (CUDA-graph capture
    # mis-reads vocab_size nested under text_config on the wrapper).
    llm = LLM(model=args.checkpoint, dtype="bfloat16", max_model_len=9216,
              gpu_memory_utilization=args.gpu_mem_util, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "video": 0})
    print(f"generating {len(usable)} prompts x {args.m} rollouts ...")
    if TokensPrompt is not None:
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids]
        outputs = llm.generate(prompts, sp)
    else:
        outputs = llm.generate(prompt_token_ids=token_ids, sampling_params=sp)

    # --- grade every completion ------------------------------------------
    grade_tasks = []      # (code, cases, timeout)
    span = []             # (idx, n_rollouts) so we can regroup rewards per prompt
    for i, out in zip(usable, outputs):
        cases = cmap[i]["cases"]
        comps = [o.text for o in out.outputs]
        for c in comps:
            grade_tasks.append((_extract_python(c or ""), cases, float(GRADER_EVAL_TIMEOUT_SECONDS)))
        span.append((i, len(comps)))

    if args.use_grader:
        from reliquary.environment.grader_client import GraderClient
        from concurrent.futures import ThreadPoolExecutor
        gc = GraderClient()
        # EXACT-PARITY grading via the real grader SERVER (validator semantics:
        # worker.py entry/args/compare/import-whitelist). GraderClient is stateless
        # (one Unix socket per call) -> thread-safe; fan out across the server's worker
        # pool so 20000 calls don't run serially. ThreadPoolExecutor.map preserves order.
        _gw = max(16, args.grade_workers)
        print(f"grading {len(grade_tasks)} completions via grader server ({_gw} threads) ...")
        with ThreadPoolExecutor(max_workers=_gw) as _ex:
            rewards = list(_ex.map(lambda t: gc.evaluate_cases(t[0], t[1], t[2]), grade_tasks))
    else:
        print(f"grading {len(grade_tasks)} completions on {args.grade_workers} workers ...")
        # SPAWN, not fork: the parent has vLLM/CUDA initialized; fork-inherited CUDA
        # context deadlocks the workers (they never run -> main blocks on a futex
        # forever). spawn gives clean CUDA-free workers. Safe here: vLLM is imported
        # inside main() (not module-level) and there's a __main__ guard, so re-import
        # in spawned workers is light and won't re-run main().
        import multiprocessing as _mp
        with ProcessPoolExecutor(max_workers=args.grade_workers,
                                 mp_context=_mp.get_context("spawn")) as ex:
            rewards = list(ex.map(_grade_one, grade_tasks, chunksize=8))

    # --- select intermediate-pass prompts --------------------------------
    pos = 0
    kept, stats = [], {}
    dist = Counter()
    for idx, k in span:
        rs = rewards[pos:pos + k]
        pos += k
        binary = [1 if r >= 0.5 else 0 for r in rs]
        frac = sum(binary) / k
        dist[round(frac, 3)] += 1
        stats[idx] = {"n": k, "frac_correct": round(frac, 3),
                      "mean_reward": round(sum(rs) / k, 3), "n_pass": sum(binary)}
        if args.p_low <= frac <= args.p_high:
            kept.append(idx)

    kept.sort()
    os.makedirs(DATA, exist_ok=True)
    json.dump(kept, open(args.out, "w"))
    os.makedirs(DIAG, exist_ok=True)
    json.dump(stats, open(os.path.join(DIAG, "opencode_pool_meta.json"), "w"), indent=0)

    # --- report ----------------------------------------------------------
    n_zero = sum(1 for s in stats.values() if s["frac_correct"] == 0)
    n_full = sum(1 for s in stats.values() if s["frac_correct"] == 1)
    print(f"\nscreened {len(stats)} prompts:")
    print(f"  pure 0/M (too hard / bimodal-low) : {n_zero} ({n_zero/len(stats):.0%})")
    print(f"  pure M/M (solved / bimodal-high)  : {n_full} ({n_full/len(stats):.0%})")
    print(f"  IN-BAND [{args.p_low},{args.p_high}] -> KEPT : {len(kept)} ({len(kept)/len(stats):.0%})")
    print(f"  frac-correct histogram: {dict(sorted(dist.items()))}")
    print(f"\nwrote {len(kept)} idxs -> {args.out}")
    print(f"      per-prompt stats   -> {os.path.join(DIAG, 'opencode_pool_meta.json')}")
    print("\nMINE IT:  RELIQUARY_ENVIRONMENTS=opencodeinstruct RELIQUARY_OCI_PROMPT_ONLY=1 \\")
    print(f"          ... mine --prompt-idx-file {args.out}   (sigma target: k in [2,6]/8, SIGMA_MIN={SIGMA_MIN})")


if __name__ == "__main__":
    main()
