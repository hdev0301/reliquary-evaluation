"""Case-INDEPENDENT in-zone curation for OpenCodeInstruct.

The validator recomputes every rollout's reward itself (batcher recompute ->
grader -> passed/total over HIDDEN cases) and the only zone gate is
``std(8 rewards) >= 0.43``. We therefore do NOT need the hidden cases. We need
to ship a group whose recomputed rewards have spread, and BOTH extremes are
case-independent:

  * LOSER  (reward == 0.0 on EVERY hidden case): a completion the grader
    sandbox cannot execute -> ``status != "ok"`` -> grader_client returns 0.0
    (it never raises). Detected STATICALLY with certainty:
      - syntax error                          (compile fails -> runtime_error)
      - a top-level import outside the tiny    (forbidden_import)
        stdlib allow-list (random/json/sys/os/numpy/...)
      - no top-level function/class            (no resolvable entry)
    plus, found by execution: forbidden_import / runtime_error / bad_output on
    every probe input.

  * WINNER (reward ~= 1.0): a genuinely-correct completion. Confirmed against
    the PUBLIC ``Sample Input/Output`` blocks parsed from the prompt (a literal
    subset of the hidden cases) and tightened by output-consensus across the
    oversample on fuzzed inputs.

Ship 4 winners + 4 losers -> validator recomputes -> rewards ~ {1,1,1,1,0,0,0,0}
-> std = 0.5 >= 0.43 -> IN ZONE BY CONSTRUCTION, no hidden cases required. Only
need >=2 of 4 winner picks to be truly correct (std at k=2 is 0.433), so winner
misclassification is well tolerated.

This module only RUNS the real validator worker (python -m
reliquary.environment.grader.worker) as a subprocess for parity; it never edits
validator/environment code.
"""

from __future__ import annotations

import ast
import json
import math
import queue
import re
import select
import subprocess
import sys
import threading
from collections import Counter
from typing import Any

# Mirror reliquary.environment.grader.worker._ALLOWED_IMPORT_ROOTS exactly.
ALLOWED_IMPORT_ROOTS = {
    "abc", "array", "bisect", "collections", "copy", "dataclasses", "decimal",
    "enum", "functools", "heapq", "itertools", "math", "operator", "re",
    "statistics", "string", "typing",
}

# Statuses that make the grader score the WHOLE completion 0.0 (fatal, returned
# by the server on first occurrence; grader_client maps !=ok -> 0.0).
FATAL_STATUSES = {
    "forbidden_import", "runtime_error", "timeout", "crash", "tampered",
    "bad_entry", "bad_request", "grader_error",
}

REPO = "/root/reliquary"

# Run the DEPLOYED (HEAD) worker for exact validator parity. The working-tree
# worker.py is stale (structure-resolution fallback removed), so importing it via
# `-m` would misclassify correctly-solving completions that use a non-canonical
# function name as losers. _grader_worker_head.py is `git show HEAD:...worker.py`
# (pure stdlib, standalone) — refresh it if the deployed worker changes.
import os as _os
HEAD_WORKER = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "_grader_worker_head.py")


# ──────────────────────────  comparison (mirror server)  ──────────────────────
def json_equal(left: Any, right: Any) -> bool:
    """Mirror grader.server._GraderServer._json_equal (compare == 'exact')."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) or isinstance(right, float):
            return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
        return left == right
    if left is None or right is None or isinstance(left, str) or isinstance(right, str):
        return type(left) is type(right) and left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(json_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left.keys()) == set(right.keys()) and all(
            json_equal(left[k], right[k]) for k in left
        )
    return False


# ──────────────────────────  static loser detection  ─────────────────────────
def static_loser_reason(code: str) -> str | None:
    """Return a reason string iff *code* scores 0.0 on EVERY hidden case with
    certainty, by static analysis alone; else None.

    Only TOP-LEVEL forbidden imports are flagged (those always execute at module
    exec -> forbidden_import). Imports nested in functions are left to the
    execution probe."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return "syntax"
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    return f"forbidden_import:{a.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                return "forbidden_import:relative"
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                return f"forbidden_import:{node.module}"
    has_entry = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for n in tree.body
    )
    if not has_entry:
        return "no_entry"
    return None


# ──────────────────────────  prompt sample parsing  ──────────────────────────
_FENCE = r"```[^\n`]*\n(.*?)```"
_SAMPLE_RE = re.compile(
    r"(?:\*\*|#+\s*)?\s*(?:Sample\s+|Example\s+)?(Input|Output)\s*\d*\s*[:\-]?\s*(?:\*\*)?\s*"
    + _FENCE,
    re.IGNORECASE | re.DOTALL,
)
# Only ACCEPT confident, unambiguous names (backticked identifiers, explicit
# `def name(`, or "function named/called X"). A wrong guess is harmless — the
# worker falls back to structure resolution — but a stopword like "that" is
# worse than nothing, so reject prose words.
_NAME_RES = [
    re.compile(r"\bdef\s+([A-Za-z_]\w*)\s*\("),
    re.compile(r"function\s+(?:named|called)\s+`?([A-Za-z_]\w*)`?", re.IGNORECASE),
    re.compile(r"`([A-Za-z_]\w*)`\s*(?:function|method|\()", re.IGNORECASE),
    re.compile(r"(?:implement|write|define)\s+(?:a\s+|the\s+)?`([A-Za-z_]\w*)`", re.IGNORECASE),
]
_NAME_STOPWORDS = {
    "that", "the", "a", "an", "to", "which", "this", "your", "you", "it",
    "function", "method", "returns", "return", "given", "is", "and", "of",
}


def parse_entry_name(prompt: str) -> str | None:
    for rx in _NAME_RES:
        m = rx.search(prompt)
        if m and m.group(1).lower() not in _NAME_STOPWORDS:
            return m.group(1)
    return None


def _lit(block: str) -> tuple[bool, Any]:
    """Best-effort literal-eval of a fenced block's content."""
    s = block.strip()
    if not s:
        return False, None
    # strip surrounding quotes already handled by literal_eval; try whole block
    for cand in (s, s.strip("`")):
        try:
            return True, ast.literal_eval(cand)
        except (ValueError, SyntaxError):
            pass
    # numeric / bool / none fallbacks
    low = s.lower()
    if low in ("true", "false"):
        return True, low == "true"
    if low in ("none", "null"):
        return True, None
    try:
        return True, int(s)
    except ValueError:
        pass
    try:
        return True, float(s)
    except ValueError:
        pass
    return True, s  # raw string (last resort)


def parse_samples(prompt: str) -> list[dict]:
    """Return [{'args': [...], 'expected': value}] parsed from public Sample I/O.

    Each parsed Input value is treated as a SINGLE positional arg (covers the
    common single-argument problems). Pairs an Input block with the next Output
    block in document order."""
    found = _SAMPLE_RE.findall(prompt)
    pending_in = None
    samples: list[dict] = []
    for kind, block in found:
        if kind.lower() == "input":
            ok, val = _lit(block)
            pending_in = ([val], ok)
        else:  # output
            if pending_in is None:
                continue
            args, in_ok = pending_in
            ok, exp = _lit(block)
            pending_in = None
            if in_ok and ok:
                samples.append({"args": args, "expected": exp})
    return samples


# ──────────────────────────  light fuzz inputs  ──────────────────────────────
def fuzz_inputs(samples: list[dict]) -> list[list]:
    """Type-driven mutations of sample inputs (single-arg). No known expected;
    used only for output-consensus clustering among winner candidates."""
    out: list[list] = []
    seen = set()

    def add(arg):
        try:
            key = json.dumps(arg, sort_keys=True)
        except TypeError:
            return
        if key not in seen:
            seen.add(key)
            out.append([arg])

    for s in samples:
        if not s["args"]:
            continue
        v = s["args"][0]
        if isinstance(v, list):
            add([])
            if v:
                add(v[:1])
                add(list(reversed(v)))
                add(v + v[:1])
            if all(isinstance(x, int) for x in v):
                add([1])
                add([5, 3, 5, 1, 3, 2])
                add([-2, -2, -1])
        elif isinstance(v, str):
            add("")
            add(v + v)
            add(v[:1])
            add(v[::-1])
        elif isinstance(v, bool):
            add(not v)
        elif isinstance(v, int):
            add(0)
            add(v + 1)
            add(-v)
    return out[:6]


# ──────────────────────────  worker pool  ────────────────────────────────────
class WorkerPool:
    """Thread-safe pool of REAL grader workers for output+status probing."""

    def __init__(self, size: int = 16, python: str | None = None,
                 cwd: str = REPO, timeout: float = 4.0) -> None:
        self.size = max(1, size)
        self.python = python or sys.executable
        self.cwd = cwd
        self.timeout = timeout
        self._idle: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._procs: list = []
        for _ in range(self.size):
            self._idle.put(self._spawn())

    def _spawn(self):
        argv = ([self.python, HEAD_WORKER] if _os.path.exists(HEAD_WORKER)
                else [self.python, "-m", "reliquary.environment.grader.worker"])
        p = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=self.cwd, text=True, bufsize=1,
        )
        with self._lock:
            self._procs.append(p)
        return p

    def _restart(self, p):
        try:
            p.kill()
        except Exception:
            pass
        with self._lock:
            if p in self._procs:
                self._procs.remove(p)
        return self._spawn()

    def run(self, code: str, entry: dict, args: list, kwargs: dict | None = None,
            timeout: float | None = None) -> tuple[Any, str]:
        timeout = timeout or self.timeout
        kwargs = kwargs or {}
        w = self._idle.get()
        try:
            req = json.dumps({
                "req_id": "x", "code": code, "entry": entry,
                "args": args, "kwargs": kwargs, "timeout_s": timeout,
            }) + "\n"
            try:
                w.stdin.write(req)
                w.stdin.flush()
            except (BrokenPipeError, OSError):
                w = self._restart(w)
                return None, "crash"
            r, _, _ = select.select([w.stdout], [], [], timeout + 1.5)
            if not r:
                w = self._restart(w)
                return None, "timeout"
            line = w.stdout.readline()
            if not line:
                w = self._restart(w)
                return None, "crash"
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                w = self._restart(w)
                return None, "crash"
            return resp.get("output"), resp.get("status", "crash")
        finally:
            self._idle.put(w)

    def shutdown(self):
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass


# ──────────────────────────  classification  ─────────────────────────────────
def classify_by_grading(items: list[tuple[int, str]], cases: list, pool: WorkerPool,
                        timeout: float = 4.0) -> dict:
    """EXACT classification: grade each completion against the validator's real
    reconstructed hidden cases (from nvidia unit_tests via the deployed
    structure_tests, matched by id). Mirrors validator scoring precisely:
    any fatal status on any case -> 0.0 total; else passed/total.

    WINNER = passes EVERY case (-> validator scores 1.0).
    LOSER  = fails EVERY case (crash/forbidden/all-wrong -> validator scores 0.0).
    MIDDLE (passes some, fails some -> partial) is DROPPED.
    Shipping k winners + (8-k) losers -> the validator recomputes the SAME rewards
    on the SAME cases -> [1..,0..] -> std>=0.43 -> in-zone, GUARANTEED (not estimated).
    """
    from concurrent.futures import ThreadPoolExecutor

    static_losers: list = []
    runnable: list[tuple[int, str]] = []
    for key, code in items:
        if static_loser_reason(code):
            static_losers.append(key)
        else:
            runnable.append((key, code))

    ncases = len(cases)

    def _grade(key_code):
        key, code = key_code
        passed = 0
        for cs in cases:
            out, st = pool.run(code, cs["entry"], cs.get("args", []),
                               cs.get("kwargs", {}), timeout)
            if st != "ok":
                return key, 0.0             # fatal on any case -> validator 0.0 TOTAL
            if json_equal(out, cs.get("expected")):
                passed += 1
        return key, (passed / ncases if ncases else 0.0)

    rewards: dict = {}     # key -> EXACT validator reward in [0,1]
    if runnable:
        with ThreadPoolExecutor(max_workers=max(1, pool.size)) as _ex:
            for key, r in _ex.map(_grade, runnable):
                rewards[key] = r
    for key in static_losers:
        rewards[key] = 0.0

    # Max-variance 8-subset: with EXACT rewards (= the validator's own recompute
    # on the same cases/worker), the std we measure is the std the validator will
    # gate on. Capture the partial-reward spread (NOT just 1.0/0.0 extremes) so
    # easy prompts that never fully-fail can still scatter. Variance of a fixed-
    # size subset is maximised by taking j smallest + (size-j) largest.
    best8, sigma = select_max_std(list(rewards.items()), size=8)
    return {
        "rewards": rewards,
        "best8": best8,
        "sigma": sigma,
        "est": rewards,
        "n_static": len(static_losers),
        "n_graded": len(rewards),
        "samples": -1,
        "graded": True,
        "reason": None,
    }


def select_max_std(reward_items: list, size: int = 8):
    """Return (list-of-`size`-keys maximising population std, that std).

    reward_items: list of (key, reward). The max-variance size-subset is some
    j smallest + (size-j) largest values (extremes only)."""
    import statistics
    if len(reward_items) < size:
        return None, 0.0
    s = sorted(reward_items, key=lambda kv: kv[1])
    n = len(s)
    best_keys, best_sigma = None, -1.0
    for j in range(0, size + 1):
        hi = size - j
        idxs = list(range(j)) + list(range(n - hi, n)) if hi > 0 else list(range(j))
        if len(set(idxs)) != size:
            continue
        vals = [s[i][1] for i in idxs]
        sig = statistics.pstdev(vals)
        if sig > best_sigma:
            best_sigma = sig
            best_keys = [s[i][0] for i in idxs]
    return best_keys, best_sigma


def classify(items: list[tuple[int, str]], prompt: str, pool: WorkerPool,
             timeout: float = 4.0, winner_min: float = 0.999,
             loser_max: float = 0.001) -> dict:
    """Bucket completions by ESTIMATED reward, keeping only the two extremes.

    For each completion we estimate the validator's passed/total by probing the
    prompt's public Sample I/O (KNOWN truth) plus fuzz inputs scored against the
    sample-passers' output consensus (derived truth on edge cases):

      WINNER (est >= winner_min ~ true 1.0): correct on EVERY probed input
        (samples + edge-case fuzz) -> passes ~all hidden cases.
      LOSER  (est <= loser_max  ~ true 0.0): wrong-or-unrunnable on EVERY probed
        input -> fails ~all hidden cases. Two sources, both reliably ~0:
          * static / exec failure (forbidden import, syntax, no entry, crash):
            case-independent 0 on every hidden case;
          * wrong-algorithm: a clean completion that mismatches truth on every
            input (incl. the basic sample) -> wrong broadly, not just on edges.

    The uncertain MIDDLE (passes some, fails some -> partial credit ~0.3..0.9)
    is DROPPED: a 0.8 'loser' or a 0.9 'winner' would pull the recomputed std
    below 0.43. Shipping k winners + (8-k) losers -> validator-recomputed
    rewards ~ {1..,0..} -> std ~ 0.5 >= 0.43, in-zone by construction.

    items: list of (token_key, extracted_code).
    """
    from concurrent.futures import ThreadPoolExecutor

    samples = parse_samples(prompt)
    entry = {"kind": "function", "name": parse_entry_name(prompt) or "__solve__"}

    static_losers: list = []
    runnable: list[tuple[int, str]] = []
    for key, code in items:
        if static_loser_reason(code):
            static_losers.append(key)
        else:
            runnable.append((key, code))

    # No public samples -> cannot confirm winners -> skip (a group needs both).
    if not samples:
        return {"winners": [], "losers": list(static_losers), "est": {},
                "n_winner": 0, "n_loser": len(static_losers),
                "n_static": len(static_losers), "n_dropped": len(runnable),
                "samples": 0, "reason": "no_samples"}

    sample_args = [s["args"] for s in samples]
    fuzz = fuzz_inputs(samples)
    nsamp = len(samples)
    inputs = sample_args + fuzz

    def _probe(key_code):
        key, code = key_code
        return key, [pool.run(code, entry, a, {}, timeout) for a in inputs]

    probed: dict = {}
    if runnable:
        with ThreadPoolExecutor(max_workers=max(1, pool.size)) as _ex:
            for key, res in _ex.map(_probe, runnable):
                probed[key] = res

    # The sample-passers (ok + match on EVERY sample) define the fuzz oracle.
    def _passes_samples(res):
        return all(res[i][1] == "ok" and json_equal(res[i][0], samples[i]["expected"])
                   for i in range(nsamp))
    sample_pass = [k for k in probed if _passes_samples(probed[k])]

    # fuzz truth = majority output among sample-passers (need >=2 agreeing,
    # else that fuzz input has no reliable oracle and is not counted).
    fuzz_truth: list[str | None] = []
    for j in range(len(fuzz)):
        c: Counter = Counter()
        for k in sample_pass:
            out, st = probed[k][nsamp + j]
            if st == "ok":
                c[json.dumps(out, sort_keys=True)] += 1
        if c:
            val, n = c.most_common(1)[0]
            fuzz_truth.append(val if n >= 2 else None)
        else:
            fuzz_truth.append(None)

    # Reliability comes from the PUBLIC SAMPLES only: each sample IS a hidden
    # case, so a completion that FAILS a sample is reliably penalised by the
    # validator. Fuzz edges are NOT guaranteed to be hidden cases (proven live:
    # a group whose losers only crashed on fuzz was recomputed all-1.0 ->
    # out_of_zone), so fuzz is used ONLY to tighten winners, never to confirm
    # losers.
    #
    #   WINNER: passes EVERY sample AND survives fuzz (no fatal, matches the
    #           sample-passers' fuzz consensus) -> robust, ~1.0 on hidden cases.
    #   LOSER : est_sample == 0 -> fails EVERY sample. Two flavours, both ~0.0:
    #           * fatal on a sample (crash/forbidden/timeout) -> 0.0 TOTAL
    #             (grader returns passed=0 the moment any case is fatal);
    #           * clean-WRONG on every sample -> wrong algorithm, ~0 on hidden.
    #           (static forbidden-import/syntax/no-entry are case-independent 0.)
    def _sample_fatal(res):
        # crash/forbidden/timeout on ANY sample -> grader returns passed=0 ->
        # reward 0.0 TOTAL, guaranteed (the sample IS a hidden case).
        return any(res[i][1] in FATAL_STATUSES for i in range(nsamp))

    def _passes_all_samples(res):
        return all(res[i][1] == "ok" and json_equal(res[i][0], samples[i]["expected"])
                   for i in range(nsamp))

    def _fuzz_robust(res):
        for j in range(len(fuzz)):
            out, st = res[nsamp + j]
            if st in FATAL_STATUSES:
                return False
            t = fuzz_truth[j]
            if t is not None and (st != "ok" or json.dumps(out, sort_keys=True) != t):
                return False
        return True

    def _n_sample_hits(res):
        return sum(1 for i in range(nsamp)
                   if res[i][1] == "ok" and json_equal(res[i][0], samples[i]["expected"]))

    est: dict = {}
    winners: list = []
    crash_losers: list = []   # tier-1: fatal on a sample -> 0.0 TOTAL (rock-solid)
    wrong_losers: list = []   # tier-2: clean-wrong on EVERY sample -> wrong algo, ~0
    for key, res in probed.items():
        if _sample_fatal(res):
            est[key] = 0.0
            crash_losers.append(key)
        elif _passes_all_samples(res) and _fuzz_robust(res):
            est[key] = 1.0
            winners.append(key)
        elif _n_sample_hits(res) == 0:
            est[key] = 0.05                       # wrong on every sample -> likely ~0
            wrong_losers.append(key)
        else:
            est[key] = 0.5                        # partial / fuzz-divergent -> dropped (uncertain)
    for key in static_losers:
        est[key] = 0.0

    # losers ordered most-reliable-first so _curate_group fills loser slots with
    # static (0.0) + sample-crashes (0.0) before clean-wrong (~0).
    sample_losers = list(static_losers) + crash_losers + wrong_losers

    losers = sample_losers  # already static + crash + wrong, reliable-first
    n_runnable_loser = len(crash_losers) + len(wrong_losers)
    return {
        "winners": winners,
        "losers": losers,
        "est": est,
        "n_winner": len(winners),
        "n_loser": len(losers),
        "n_static": len(static_losers),
        "n_crash": len(crash_losers),
        "n_wrong": len(wrong_losers),
        "n_dropped": len(probed) - len(winners) - n_runnable_loser,
        "samples": nsamp,
        "reason": None,
    }
