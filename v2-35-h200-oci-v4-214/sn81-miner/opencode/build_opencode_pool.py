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
import re
import sys
import time
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


def _grade_one_inproc(code, cases, timeout):
    """In-process grader: SIGALRM soft bound + per-case scoring -> passed/total.

    Runs inside a forked child (see _grade_one) so a C-level runaway that
    SIGALRM cannot interrupt is hard-killed by the parent rather than wedging
    the worker pool. exec failures and per-case errors score 0 (never raise).
    """
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


def _grade_one(task):
    """Hard-bounded grader: run the in-process grader in a forked child and
    SIGKILL it if it overruns. SIGALRM alone can't interrupt C-level runaways
    (e.g. ``10**(10**8)``, catastrophic regex backtracking); only a process kill
    can, so a single pathological completion no longer wedges the pool.
    """
    code, cases, timeout = task
    if not cases:
        return 0.0
    import os
    import pickle
    import select
    import signal

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: compute, ship result, hard-exit (skip atexit/flush)
        os.close(r)
        try:
            frac = _grade_one_inproc(code, cases, timeout)
        except BaseException:
            frac = 0.0
        try:
            os.write(w, pickle.dumps(float(frac)))
        except BaseException:
            pass
        os._exit(0)

    # parent: wait up to a hard deadline, then SIGKILL the child unconditionally
    os.close(w)
    frac = 0.0
    try:
        ready, _, _ = select.select([r], [], [], float(timeout) + 1.0)
        if ready:
            buf = b""
            while True:
                chunk = os.read(r, 65536)
                if not chunk:
                    break
                buf += chunk
            if buf:
                frac = float(pickle.loads(buf))
    except BaseException:
        frac = 0.0
    finally:
        try:
            os.close(r)
        except BaseException:
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except BaseException:
            pass
        try:
            os.waitpid(pid, 0)
        except BaseException:
            pass
    return frac


def select_max_std_sigma(rewards, size=8):
    """Max population-std achievable by ANY `size`-subset of `rewards`.

    EXACT mirror of the miner's consensus.select_max_std (pregen ships the
    max-std 8-subset). The max-variance subset is j smallest + (size-j) largest
    values, so we sweep j and keep the best std. This is the SAME number the
    validator recomputes -> selecting the build pool on this == selecting on
    what actually ships, fixing the binary-frac vs exact-sigma mismatch."""
    import statistics
    if len(rewards) < size:
        return 0.0
    s = sorted(rewards)
    n = len(s)
    best = 0.0
    for j in range(0, size + 1):
        hi = size - j
        idxs = list(range(j)) + list(range(n - hi, n)) if hi > 0 else list(range(j))
        if len(set(idxs)) != size:
            continue
        sig = statistics.pstdev([s[i] for i in idxs])
        if sig > best:
            best = sig
    return best


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
# Strategy A selection — prefilter the mirror to easy, short, Sample-I/O,
# single-archetype CS-101 prompts so a far higher fraction land in the curatable
# [0.25,0.75] band (denser pool). This is the HONEST half of Strategy A: better
# prompt SELECTION only. It deliberately does NOT implement duplicate-archetype
# harvesting or any distribution_suspicious / dedup evasion.
# ---------------------------------------------------------------------------

_ARCHETYPES = (
    # classic algorithms (unambiguous, CS-101)
    "fibonacci", "prime number", "is prime", "longest increasing", "sieve",
    "merge two sorted", "merge sorted", "palindrome", "anagram", "factorial",
    "gcd", "lcm", "binary search", "bubble sort", "selection sort",
    "insertion sort", "two sum", "fisher-yates", "caesar cipher",
    "roman numeral", "fizzbuzz", "fizz buzz", "histogram", "fibonacci sequence",
    # simple single-class OOP
    "calculator", "bankaccount", "bank account", "inventory system",
    "linked list", "linkedlist", "trie", "implement a stack", "implement a queue",
    # small, scalar-output utilities
    "temperature", "celsius", "fahrenheit", "vowel", "factorial of",
    "reverse the digits", "sum of digits", "count the number of vowels",
    "perfect number", "armstrong", "leap year", "roman to integer",
    "integer to roman",
)
_EXCLUDE = (
    "flask", "django", "fastapi", "api endpoint", "rest api", "http request",
    "http server", "database", "sqlite", "postgres", "mysql", "mongodb",
    "redis", "asyncio", "async def", "await ", "threading", "multiprocess",
    "concurren", "socket", "microservice", "docker", "kubernetes", "selenium",
    "web scrap", "scrape", "gui", "tkinter", "pygame", "matplotlib", "plot the",
    "tensorflow", "pytorch", "numpy", "pandas", "scikit", "machine learning",
    "neural network", "optimize for performance", "must run in o(",
    "time complexity should be", "multiple files", "across files", "across modules",
    "compiler for", "interpreter for", "regex engine", "operating system",
)


def _sample_output_block(text):
    m = re.search(r"sample\s*output[^\n]*\n+```[a-z]*\n(.*?)```", text, re.I | re.S)
    return m.group(1) if m else None


def _strategy_a_keep(text):
    """True iff the prompt fits Strategy A's easy/short/sample-I/O/archetype profile."""
    if not text:
        return False
    t = text.lower()
    if "sample input" not in t or "sample output" not in t:
        return False
    if not (400 <= len(text) <= 1400):
        return False
    if any(x in t for x in _EXCLUDE):
        return False
    if not any(a in t for a in _ARCHETYPES):
        return False
    # scalar / short sample output: a few lines, not a long dump
    so = _sample_output_block(text)
    if so is not None:
        lines = [ln for ln in so.strip().splitlines() if ln.strip()]
        if len(lines) > 8 or len(so) > 300:
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT_PATH") or DEFAULT_BASE_MODEL)
    ap.add_argument("--cases-source", choices=["reconstruct", "subset"], default="reconstruct")
    ap.add_argument("--max-candidates", type=int, default=3000, help="how many mirror prompts to screen")
    ap.add_argument("--strategy-a", action="store_true",
                    help="Strategy A selection: prefilter the mirror to easy/short/Sample-I/O/"
                         "single-archetype CS-101 prompts before sampling (denser curatable pool)")
    ap.add_argument("--idx-file", default=None, help="re-screen/refine an existing pool json instead of random sampling")
    ap.add_argument("--m", type=int, default=max(16, M_ROLLOUTS), help="rollouts per prompt (oversample for screen)")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--p-low", type=float, default=0.06, help="min fraction-correct to keep (>0 => not 0/M)")
    ap.add_argument("--p-high", type=float, default=0.94, help="max fraction-correct to keep (<1 => not M/M)")
    ap.add_argument("--strict-zone", action="store_true",
                    help="tighten the band to k in [2,6]/8 (sigma>=SIGMA_MIN) instead of the wide screen band")
    ap.add_argument("--two-stage", action=argparse.BooleanOptionalAction, default=None,
                    help="stage 1: n=8 screen per candidate, advance iff full-pass count k in [1,7] "
                         "of >=8 EOS-terminated rollouts; stage 2: full n=--m generation + the "
                         "exact-sigma keep gate. Skips candidates the checkpoint fully solves/fails "
                         "cheaply. Defaults ON with --strict-zone (--no-two-stage disables).")
    ap.add_argument("--zone-sigma", type=float, default=0.0,
                    help="EXACT-sigma keep gate: keep a prompt iff its max-std 8-subset of exact "
                         "rewards has pstdev >= this (== what the miner ships). Replaces the binary "
                         "frac band, fixing the build/mine criterion mismatch. --strict-zone defaults "
                         "this to 0.45 (ABOVE the 0.43 ship floor, with margin, since the build now "
                         "grades at the miner's depth). 0 = legacy binary band.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.65)
    ap.add_argument("--max-scan", type=int, default=2_000_000)
    ap.add_argument("--grade-workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--use-grader", action="store_true", help="grade via a running grader server (sandboxed parity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA, "inzone_pool_opencode.json"))
    ap.add_argument("--append", action="store_true",
                    help="UNION the kept idxs into the existing --out pool (incremental chunk build) "
                         "instead of overwriting. Lets many small chunks accumulate one growing pool.")
    ap.add_argument("--screened-file", default=None,
                    help="json list of already-screened idxs (this checkpoint) to EXCLUDE from candidate "
                         "sampling; the chunk's sampled candidates are added back, so successive chunks "
                         "cover NEW prompts. Reset this (+ the pool) when the checkpoint changes.")
    args = ap.parse_args()

    if args.strict_zone:
        args.p_low, args.p_high = 0.25, 0.75  # k in [2,6]/8 -> sigma>=0.433 (diagnostic band)
        if args.zone_sigma <= 0.0:
            # 0.45: gate ABOVE the 0.43 ship floor, with margin. The build now grades at
            # the SAME depth as the miner (POOL_M == miner oversample) with the SAME
            # validator-parity worker, so build sigma8 ~= mine sigma8. A gate BELOW the
            # floor (the old 0.33) kept near-floor prompts that, under sampling jitter,
            # collapse to sigma=0 at mine time -> a full-but-DEAD pool (observed ckpt 907,
            # 94% kept yet 0 ships). The margin keeps only prompts that clear the floor
            # with slack so they re-scatter at mine.
            args.zone_sigma = 0.45
    if args.two_stage is None:
        args.two_stage = bool(args.strict_zone)

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
    if args.strategy_a:
        _texts = mirror["input"]
        _before = len(cand)
        cand = [i for i in cand if _strategy_a_keep(_texts[i])]
        print(f"strategy-a prefilter: {len(cand)}/{_before} prompts kept "
              f"(easy/short/sample-IO/archetype)")
    # Skip prompts already screened on THIS checkpoint, so successive chunks
    # cover NEW ground instead of regrading the same prompts (incremental build).
    if args.screened_file and os.path.exists(args.screened_file):
        _scr = set(json.load(open(args.screened_file)))
        _before = len(cand)
        cand = [i for i in cand if i not in _scr]
        print(f"screened-file: excluding {_before - len(cand)} already-screened "
              f"-> {len(cand)} fresh candidates remain")
    # Skip prompts already recorded as saturated (solve rate 0.0 or 1.0) on the
    # CURRENT snapshot — re-screening them cannot yield scatterers.
    solve_stats_path = os.path.join(DATA, "oci_solve_stats.json")
    snap_id = os.path.basename(os.path.normpath(args.checkpoint))
    try:
        _solve_prev = json.load(open(solve_stats_path)) if os.path.exists(solve_stats_path) else {}
    except Exception:
        _solve_prev = {}
    _sat = {int(k) for k, v in _solve_prev.items()
            if isinstance(v, dict) and v.get("snapshot_hash") == snap_id
            and v.get("solve_rate") in (0.0, 1.0)}
    if _sat:
        _before = len(cand)
        cand = [i for i in cand if i not in _sat]
        print(f"solve-stats: excluding {_before - len(cand)} saturated idxs "
              f"(solve rate 0/1 on this snapshot) -> {len(cand)} remain")
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
    from transformers import AutoTokenizer, GenerationConfig
    try:
        from vllm.inputs import TokensPrompt
    except Exception:
        TokensPrompt = None

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    token_ids = [encode_prompt(tok, cmap[i]["prompt"]) for i in usable]
    # EOS set resolved like the miner (cli/main.py): generation_config eos_token_id(s)
    # union tokenizer.eos_token_id -> stop_token_ids parity with vllm_backend.
    eos_set: set = set()
    try:
        gcfg = GenerationConfig.from_pretrained(args.checkpoint)
        geos = getattr(gcfg, "eos_token_id", None)
        if isinstance(geos, int):
            eos_set.add(int(geos))
        elif geos:
            eos_set.update(int(e) for e in geos)
    except Exception:
        pass
    if tok.eos_token_id is not None:
        eos_set.add(int(tok.eos_token_id))
    # Qwen3.5 is a MULTIMODAL-wrapper checkpoint used text-only; the snapshot has no
    # image processor. limit_mm_per_prompt={image:0,video:0} makes vLLM skip loading it
    # and run the LM only (mirrors miner vllm_backend._multimodal_kwargs). Without it:
    # OSError "Can't load image processor". enforce_eager already set (CUDA-graph capture
    # mis-reads vocab_size nested under text_config on the wrapper).
    llm = LLM(model=args.checkpoint, dtype="bfloat16", max_model_len=9216,
              gpu_memory_utilization=args.gpu_mem_util, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "video": 0})
    _tid = dict(zip(usable, token_ids))

    def _generate(idxs, n):
        sp = SamplingParams(
            n=n, temperature=T_PROTO, top_p=TOP_P_PROTO,
            top_k=(TOP_K_PROTO if TOP_K_PROTO and TOP_K_PROTO > 0 else -1),
            max_tokens=args.max_new_tokens,
            stop_token_ids=sorted(eos_set),
        )
        ids = [_tid[i] for i in idxs]
        if TokensPrompt is not None:
            return llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], sp)
        return llm.generate(prompt_token_ids=ids, sampling_params=sp)

    two_stage = bool(args.two_stage) and args.zone_sigma > 0.0
    if two_stage:
        outputs = []
    else:
        print(f"generating {len(usable)} prompts x {args.m} rollouts ...")
        outputs = _generate(usable, args.m)

    # --- grade every completion + score per prompt -----------------------
    kept, stats, solve_new = [], {}, {}
    dist = Counter()
    if args.zone_sigma > 0.0:
        # EXACT-sigma gate: grade with the SAME validator-parity worker the MINER
        # uses (consensus.classify_by_grading -> deployed grader worker), so the
        # build's sigma8 == the sigma the miner/validator recomputes at mine time.
        # The old in-process exec() grader spuriously failed ~15% of runnable
        # completions; max-std-8 then turned that noise into sigma8~0.43 on NEARLY
        # EVERY prompt (false scatter that skipped at mine as sigma=0.000, 0.5% ship).
        # Worker grading closes that build/mine gap -> the pool only holds prompts
        # that ACTUALLY scatter for the validator.
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from consensus import WorkerPool, classify_by_grading
        gpool = WorkerPool(size=max(8, args.grade_workers),
                           timeout=float(GRADER_EVAL_TIMEOUT_SECONDS))
        print(f"grading {len(usable)} prompts via validator-parity "
              f"worker (consensus, {gpool.size} workers) ...")
        try:
            screen = usable
            if two_stage:
                print(f"stage 1: generating {len(usable)} prompts x 8 rollouts ...")
                out1 = _generate(usable, 8)
                screen = []
                for i, out in zip(usable, out1):
                    comps = [_extract_python(o.text or "") for o in out.outputs
                             if o.finish_reason != "length"]
                    n_term = len(comps)
                    if n_term < 8:
                        stats[i] = {"stage": 1, "n": n_term, "n_terminated": n_term,
                                    "frac_correct": 0.0, "mean_reward": 0.0,
                                    "n_pass": 0, "k_full": 0, "sigma8": 0.0}
                        solve_new[str(i)] = {"solve_rate": None, "stage": 1,
                                             "n_terminated": n_term, "snapshot": args.checkpoint,
                                             "snapshot_hash": snap_id, "ts": int(time.time())}
                        continue
                    res = classify_by_grading(list(enumerate(comps)), cmap[i]["cases"],
                                              gpool, timeout=float(GRADER_EVAL_TIMEOUT_SECONDS))
                    rs = list(res.get("rewards", {}).values())
                    kf = sum(1 for r in rs if r >= 1.0)
                    binary = [1 if r >= 0.5 else 0 for r in rs]
                    frac = sum(binary) / max(1, len(rs))
                    stats[i] = {"stage": 1, "n": len(rs), "n_terminated": n_term,
                                "frac_correct": round(frac, 3),
                                "mean_reward": round(sum(rs) / max(1, len(rs)), 3),
                                "n_pass": sum(binary), "n_static": int(res.get("n_static", 0)),
                                "k_full": kf, "sigma8": round(float(res.get("sigma", 0.0)), 3)}
                    solve_new[str(i)] = {"solve_rate": round(kf / max(1, len(rs)), 4), "stage": 1,
                                         "n_terminated": n_term, "snapshot": args.checkpoint,
                                         "snapshot_hash": snap_id, "ts": int(time.time())}
                    if 1 <= kf <= 7:
                        screen.append(i)
                    else:
                        dist[round(frac, 3)] += 1
                print(f"stage 1: {len(screen)}/{len(usable)} advance (full-pass k in [1,7] of 8)")
                if screen:
                    print(f"stage 2: generating {len(screen)} prompts x {args.m} rollouts ...")
                    outputs = _generate(screen, args.m)
            for i, out in zip(screen, outputs):
                comps = [_extract_python(o.text or "") for o in out.outputs
                         if o.finish_reason != "length"]
                k = len(comps)
                if k < 8:
                    stats[i] = {"stage": 2, "n": k, "n_terminated": k, "frac_correct": 0.0,
                                "mean_reward": 0.0, "n_pass": 0, "sigma8": 0.0}
                    solve_new[str(i)] = {"solve_rate": None, "stage": 2,
                                         "n_terminated": k, "snapshot": args.checkpoint,
                                         "snapshot_hash": snap_id, "ts": int(time.time())}
                    continue
                res = classify_by_grading(list(enumerate(comps)), cmap[i]["cases"],
                                          gpool, timeout=float(GRADER_EVAL_TIMEOUT_SECONDS))
                rs = list(res.get("rewards", {}).values())
                sigma8 = float(res.get("sigma", 0.0))
                kf = sum(1 for r in rs if r >= 1.0)
                binary = [1 if r >= 0.5 else 0 for r in rs]
                frac = sum(binary) / max(1, len(rs))
                dist[round(frac, 3)] += 1
                stats[i] = {"stage": 2, "n": len(rs), "n_terminated": k,
                            "frac_correct": round(frac, 3),
                            "mean_reward": round(sum(rs) / max(1, len(rs)), 3),
                            "n_pass": sum(binary), "n_static": int(res.get("n_static", 0)),
                            "sigma8": round(sigma8, 3)}
                solve_new[str(i)] = {"solve_rate": round(kf / max(1, len(rs)), 4), "stage": 2,
                                     "n_terminated": k, "snapshot": args.checkpoint,
                                     "snapshot_hash": snap_id, "ts": int(time.time())}
                if sigma8 >= args.zone_sigma:
                    kept.append(i)
        finally:
            gpool.shutdown()
    else:
        # legacy: in-process exec grader + binary frac band (NOT validator parity).
        # SPAWN (not fork): parent has CUDA initialized; fork-inherited CUDA deadlocks.
        grade_tasks, span = [], []
        for i, out in zip(usable, outputs):
            cases = cmap[i]["cases"]
            texts = [o.text for o in out.outputs if o.finish_reason != "length"]
            for c in texts:
                grade_tasks.append((_extract_python(c or ""), cases, float(GRADER_EVAL_TIMEOUT_SECONDS)))
            span.append((i, len(texts)))
        print(f"grading {len(grade_tasks)} completions on {args.grade_workers} workers (exec) ...")
        import multiprocessing as _mp
        with ProcessPoolExecutor(max_workers=args.grade_workers,
                                 mp_context=_mp.get_context("spawn")) as ex:
            rewards = list(ex.map(_grade_one, grade_tasks, chunksize=8))
        pos = 0
        for idx, k in span:
            rs = rewards[pos:pos + k]; pos += k
            binary = [1 if r >= 0.5 else 0 for r in rs]
            frac = sum(binary) / max(1, k)
            dist[round(frac, 3)] += 1
            stats[idx] = {"n": k, "n_terminated": k, "frac_correct": round(frac, 3),
                          "mean_reward": round(sum(rs) / max(1, k), 3), "n_pass": sum(binary)}
            if k and args.p_low <= frac <= args.p_high:
                kept.append(idx)

    kept.sort()
    os.makedirs(DATA, exist_ok=True)
    if args.append and os.path.exists(args.out):
        # incremental chunk: UNION into the existing pool (dedup) so the pool grows
        try:
            _prev = set(json.load(open(args.out)))
        except Exception:
            _prev = set()
        _merged = sorted(_prev | set(kept))
        _added = len(set(kept) - _prev)
        print(f"append: {len(_prev)} existing + {len(kept)} kept = {len(_merged)} total "
              f"(+{_added} new idxs)")
        json.dump(_merged, open(args.out, "w"))
    else:
        json.dump(kept, open(args.out, "w"))
    # record the candidates we screened this chunk so later chunks skip them
    if args.screened_file:
        try:
            _prev_scr = set(json.load(open(args.screened_file))) if os.path.exists(args.screened_file) else set()
        except Exception:
            _prev_scr = set()
        _all_scr = sorted(_prev_scr | set(cand))
        json.dump(_all_scr, open(args.screened_file, "w"))
        print(f"screened-file: {len(_all_scr)} idxs screened this checkpoint (+{len(_all_scr) - len(_prev_scr)})")
    if solve_new:
        try:
            _ss = json.load(open(solve_stats_path)) if os.path.exists(solve_stats_path) else {}
        except Exception:
            _ss = {}
        _ss.update(solve_new)
        json.dump(_ss, open(solve_stats_path, "w"))
        print(f"solve-stats: merged {len(solve_new)} idxs -> {solve_stats_path} ({len(_ss)} total)")
    os.makedirs(DIAG, exist_ok=True)
    json.dump(stats, open(os.path.join(DIAG, "opencode_pool_meta.json"), "w"), indent=0)

    # --- report ----------------------------------------------------------
    n_zero = sum(1 for s in stats.values() if s["frac_correct"] == 0)
    n_full = sum(1 for s in stats.values() if s["frac_correct"] == 1)
    print(f"\nscreened {len(stats)} prompts:")
    print(f"  pure 0/M (too hard / bimodal-low) : {n_zero} ({n_zero/len(stats):.0%})")
    print(f"  pure M/M (solved / bimodal-high)  : {n_full} ({n_full/len(stats):.0%})")
    if args.zone_sigma > 0.0:
        _sig8 = [s.get("sigma8", 0.0) for s in stats.values()]
        _band = sum(1 for s in stats.values() if args.p_low <= s["frac_correct"] <= args.p_high)
        print(f"  binary frac-band [{args.p_low},{args.p_high}] (OLD criterion) : {_band} ({_band/len(stats):.0%})")
        print(f"  EXACT sigma8>={args.zone_sigma} (ship-aligned) -> KEPT : {len(kept)} ({len(kept)/len(stats):.0%})")
        print(f"  sigma8 buckets: >=0.43:{sum(1 for x in _sig8 if x>=0.43)} "
              f">=0.38:{sum(1 for x in _sig8 if x>=0.38)} >=0.30:{sum(1 for x in _sig8 if x>=0.30)} "
              f"~0(<0.05):{sum(1 for x in _sig8 if x<0.05)}")
    else:
        print(f"  IN-BAND [{args.p_low},{args.p_high}] -> KEPT : {len(kept)} ({len(kept)/len(stats):.0%})")
    print(f"  frac-correct histogram: {dict(sorted(dist.items()))}")
    print(f"\nwrote {len(kept)} idxs -> {args.out}")
    print(f"      per-prompt stats   -> {os.path.join(DIAG, 'opencode_pool_meta.json')}")
    print("\nMINE IT:  RELIQUARY_ENVIRONMENTS=opencodeinstruct RELIQUARY_OCI_PROMPT_ONLY=1 \\")
    print(f"          ... mine --prompt-idx-file {args.out}   (sigma target: k in [2,6]/8, SIGMA_MIN={SIGMA_MIN})")


if __name__ == "__main__":
    main()
