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
       the validator's cases byte-exact (see diagnostics/verify_opencode_gate.py).
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
  cd /root/reliquary && .venv/bin/python /root/sn81-miner/dataprep/build_opencode_pool.py \
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


# Model-generated code routinely calls exit()/quit()/sys.exit() (-> SystemExit, a
# BaseException) or blocks on input(). Inside a ProcessPoolExecutor worker those
# ESCAPE `except Exception`, kill the worker, and one dead worker breaks the WHOLE
# pool -> the entire grading run dies with BrokenProcessPool (losing all the GPU work).
# Defense: (1) _grade_one catches BaseException everywhere and runs under a neutered
# builtins; (2) _grade_all runs each exec in a DISPOSABLE forked child under hard
# rlimits, so nothing a model emits (exit/os._exit/segfault/OOM/blocked-signal hang)
# can take down the parent — the parent just SIGKILLs the child and scores 0.0.

def _safe_globals():
    """Fresh exec globals: full builtins EXCEPT process-killing / blocking calls."""
    import builtins as _b

    def _blocked(*_a, **_k):
        raise RuntimeError("call disabled in grading sandbox")

    bd = dict(_b.__dict__)
    for _n in ("exit", "quit", "input"):
        bd[_n] = _blocked
    return {"__builtins__": bd}


def _grade_one(task):
    """Run one (code, cases) and return passed/total in [0,1].

    Catches BaseException around BOTH the exec and every per-case call so that
    exit()/sys.exit()/SystemExit from model code score 0 instead of propagating.
    SIGALRM bounds wall time; the hard CPU/mem caps are applied by the forked child
    in _grade_all. Network/FS are NOT sandboxed here — see module docstring.
    """
    code, cases, timeout = task
    if not cases:
        return 0.0
    import signal

    def _timeout(_signum, _frame):
        raise TimeoutError()

    passed = 0
    total = len(cases)
    ns = _safe_globals()
    try:
        old = signal.signal(signal.SIGALRM, _timeout)
    except Exception:
        old = None
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        try:
            exec(code, ns)  # noqa: S102 - intentional, see module docstring
        except BaseException:
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
            except BaseException:
                pass
    except BaseException:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if old is not None:
            try:
                signal.signal(signal.SIGALRM, old)
            except Exception:
                pass
    return passed / total if total else 0.0


def _grade_all(tasks, workers, slack=2.0, mem_cap_gib=2):
    """CRASH-PROOF parallel grading, driven from the (single-threaded) main process.

    Each completion is exec'd in a DISPOSABLE forked child under hard rlimits
    (RLIMIT_CPU + RLIMIT_AS + RLIMIT_FSIZE=0). A child that calls os._exit, segfaults,
    OOMs, or hangs (even with SIGALRM blocked) only kills ITSELF — the parent SIGKILLs
    it at its wall-clock deadline and scores 0.0. No ProcessPoolExecutor is used, so a
    bad completion can never raise BrokenProcessPool and abort the run.

    Returns (rewards_in_order, n_hard_crashed).
    """
    import os
    import select
    import struct
    import signal as _sig
    import resource
    import time as _time

    n = len(tasks)
    rewards = [0.0] * n
    crashed = 0
    inflight = {}      # pid -> [task_index, read_fd, deadline, buf]
    fd_to_pid = {}     # read_fd -> pid
    next_i = 0
    done = 0

    def _spawn(i):
        code, cases, timeout = tasks[i]
        if not cases:
            return False
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            # ---------------- child (disposable, locked-down) ----------------
            try:
                os.close(r)
                cpu = max(1, int(timeout) + 1)
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
                except Exception:
                    pass
                try:
                    mem = int(mem_cap_gib) * 1024 * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                except Exception:
                    pass
                try:
                    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
                except Exception:
                    pass
                score = _grade_one((code, cases, timeout))
                os.write(w, struct.pack("d", float(score)))
            except BaseException:
                try:
                    os.write(w, struct.pack("d", 0.0))
                except Exception:
                    pass
            finally:
                os._exit(0)
        # ---------------- parent ----------------
        os.close(w)
        inflight[pid] = [i, r, _time.monotonic() + float(timeout) + slack, b""]
        fd_to_pid[r] = pid
        return True

    def _reap(pid):
        nonlocal crashed
        i, r, _dl, buf = inflight.pop(pid)
        fd_to_pid.pop(r, None)
        try:
            os.close(r)
        except Exception:
            pass
        try:
            os.kill(pid, _sig.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        if len(buf) == 8:
            rewards[i] = struct.unpack("d", buf)[0]
        else:
            crashed += 1  # child died before writing a result

    while done < n:
        while len(inflight) < workers and next_i < n:
            spawned = _spawn(next_i)
            next_i += 1
            if not spawned:
                done += 1
        if not inflight:
            continue
        now = _time.monotonic()
        nearest = min(v[2] for v in inflight.values())
        wait = nearest - now
        ready, _, _ = select.select(list(fd_to_pid), [], [], wait if wait > 0 else 0.02)
        for r in ready:
            pid = fd_to_pid.get(r)
            if pid is None:
                continue
            need = 8 - len(inflight[pid][3])
            try:
                chunk = os.read(r, need)
            except Exception:
                chunk = b""
            if chunk:
                inflight[pid][3] += chunk
            if (not chunk) or len(inflight[pid][3]) >= 8:
                _reap(pid)
                done += 1
        now = _time.monotonic()
        for pid in [p for p, v in inflight.items() if now >= v[2]]:
            _reap(pid)
            done += 1
    return rewards, crashed


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
    ap.add_argument("--regrade-from", default=None,
                    help="re-grade persisted completions (data/oci_gen_cache_seed*.json) WITHOUT "
                         "re-running generation — use after a grading change or crash")
    ap.add_argument("--cases-only", action="store_true",
                    help="reconstruct + cache test cases then STOP before loading the GPU — "
                         "CPU/network only, safe to run alongside a live miner")
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

    if args.cases_only:
        # CPU/network only — no GPU touched. Safe to run while the miner is mining.
        # The case cache accrues across runs; rebuild the local subset to pick up new cases.
        cache_path = os.path.join(DATA, "oci_cases_cache.json")
        total = len(json.load(open(cache_path))) if os.path.exists(cache_path) else len(usable)
        print(f"\ncases-only: cached {len(usable)} usable case-sets this run "
              f"(cache now ~{total} entries) -> {cache_path}")
        print("skipped GPU generation/grading. Next: rebuild the local subset, then restart the miner:")
        print("  .venv/bin/python /root/sn81-miner/opencode/build_local_subset.py")
        return

    # --- obtain M completions per prompt (generate on GPU, or reload from cache) ---
    gen_cache = os.path.join(DATA, f"oci_gen_cache_seed{args.seed}.json")
    comps_by_idx: dict = {}
    if args.regrade_from:
        # FAST PATH: re-grade persisted completions; skip the GPU entirely. This is how
        # you iterate on grading after a crash WITHOUT re-running the expensive generation.
        blob = json.load(open(args.regrade_from))
        comps_by_idx = {int(k): v for k, v in blob.get("completions", {}).items()}
        usable = [i for i in usable if i in comps_by_idx]
        print(f"REGRADE: loaded completions for {len(comps_by_idx)} prompts from "
              f"{args.regrade_from} (no generation); usable now {len(usable)}")
        if not usable:
            print("no overlap between persisted completions and resolved cases — aborting")
            return
    else:
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
        # The SN checkpoint (R0mAI/reliquary-sn-v23) is a MULTIMODAL Qwen3_5ForConditionalGeneration
        # (~297 vision tensors). Load it TEXT-ONLY exactly like the miner does
        # (vllm_backend: enforce_eager + limit_mm_per_prompt={'image':0,'video':0}) so vLLM does not
        # reserve a multimodal encoder cache / run dummy-mm profiling — otherwise the load can OOM or
        # diverge from the miner's runtime, making the screened scatter unrepresentative.
        llm = LLM(model=args.checkpoint, dtype="bfloat16", max_model_len=9216,
                  gpu_memory_utilization=args.gpu_mem_util, enforce_eager=True,
                  limit_mm_per_prompt={"image": 0, "video": 0})
        print(f"generating {len(usable)} prompts x {args.m} rollouts ...")
        if TokensPrompt is not None:
            prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids]
            outputs = llm.generate(prompts, sp)
        else:
            outputs = llm.generate(prompt_token_ids=token_ids, sampling_params=sp)
        comps_by_idx = {i: [o.text for o in out.outputs] for i, out in zip(usable, outputs)}

        # PERSIST raw completions BEFORE grading so a grading failure never wastes the GPU
        # run — re-grade with `--regrade-from <this file>` (no regeneration needed).
        try:
            json.dump({"checkpoint": args.checkpoint, "m": args.m,
                       "completions": {str(i): comps_by_idx[i] for i in usable}},
                      open(gen_cache, "w"))
            print(f"persisted completions for {len(usable)} prompts -> {gen_cache}")
        except Exception as e:  # persistence is best-effort; never block grading on it
            print(f"WARNING: could not persist completions ({e})")

        # RELEASE the vLLM engine before the CPU-bound grading phase: frees the GPU and
        # leaves a CUDA-free, single-threaded parent to fork the grading sandbox from.
        try:
            del outputs, llm
            import gc as _gc
            import torch
            _gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # --- grade every completion (CRASH-PROOF sandboxed grading) ----------
    grade_tasks = []      # (code, cases, timeout)
    span = []             # (idx, n_rollouts) so we can regroup rewards per prompt
    for i in usable:
        cases = cmap[i]["cases"]
        comps = comps_by_idx.get(i, [])
        for c in comps:
            grade_tasks.append((_extract_python(c or ""), cases, float(GRADER_EVAL_TIMEOUT_SECONDS)))
        span.append((i, len(comps)))

    if args.use_grader:
        from reliquary.environment.grader_client import GraderClient
        gcl = GraderClient()
        rewards = [gcl.evaluate_cases(code, cases, ts) for code, cases, ts in grade_tasks]
    else:
        print(f"grading {len(grade_tasks)} completions on {args.grade_workers} sandboxed workers ...")
        rewards, n_crashed = _grade_all(grade_tasks, args.grade_workers)
        if n_crashed:
            print(f"  note: {n_crashed} completion(s) hard-crashed their sandbox child "
                  f"(os._exit/segfault/OOM/timeout) -> scored 0.0 (NOT silently dropped)")

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
