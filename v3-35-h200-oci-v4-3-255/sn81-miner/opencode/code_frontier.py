"""Code-aware online frontier for the opencode consensus path.

The stock reliquary frontier (reliquary/miner/frontier.py) extracts MATH-word-
problem features (gt_is_integer / gt_magnitude / arithmetic keywords) and learns
from an n_correct label that is ALWAYS 0 in consensus mode -> useless here. This
module is the opencode replacement, kept entirely under sn81-miner/ (no edits to
reliquary/miner/frontier.py):

  * extract_code_features  -- difficulty/topic features that actually vary across
    CODE prompts (length, #test-cases, algorithmic keywords, constraints).
  * CodeFrontierModel       -- online logistic regression, positive-upweighted.
  * CodeFrontierSampler     -- drop-in candidate_sampler. Its record() keeps the
    SAME (prompt_idx, n_correct, terminated) signature as frontier.FrontierSampler
    so pregen's existing _record_outcome(idx, None, n) calls fall through as
    negatives unchanged; the consensus path feeds the crisp POSITIVE signal by
    passing synthetic counts derived from the exact grading sigma.

Pure prompt SELECTION: the validator recomputes every reward/logprob, so biasing
which prompts we deep-mine cannot affect correctness — only our in-zone yield.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading

import numpy as np

logger = logging.getLogger("reliquary.miner.code_frontier")

_NUM_RE = re.compile(r"(?<![\w.])(-?\d+(?:,\d{3})*(?:\.\d+)?)")

# Topic / difficulty keyword groups. Each contributes one 0/1 feature: whether
# ANY alias in the group appears in the (lower-cased) prompt. Chosen because a
# code prompt's boundary-ness (model sometimes-solves / sometimes-fails) tracks
# algorithmic difficulty far more than the math features ever could.
_KW_GROUPS: dict[str, str] = {
    "recursion": r"recursi|backtrack|memoi",
    "dp": r"dynamic program|\bdp\b|subsequence|knapsack|subarray",
    "graph": r"\bgraph\b|\bnode\b|\bedge\b|\bvertex|adjacen|\bbfs\b|\bdfs\b|dijkstra|topolog",
    "tree": r"\btree\b|\bbinary tree\b|\bbst\b|\bleaf\b|\broot\b|traversal",
    "matrix": r"\bmatrix\b|\bgrid\b|2d array|\brow\b.*\bcolumn\b",
    "sort_search": r"\bsort\b|sorted|\bsearch\b|binary search|\bmerge\b|\bpivot\b",
    "string": r"\bstring\b|substring|palindrome|anagram|\bprefix\b|\bsuffix\b|charact",
    "hashmap": r"\bhash\b|dictionar|\bmap\b|key-value|frequen|\bcount\b",
    "stack_queue": r"\bstack\b|\bqueue\b|\bheap\b|priority queue|deque|monoton",
    "optimize": r"optim|efficien|time complex|space complex|\bo\(n|minimi|maximi|fewest|least number",
    "mathnum": r"\bprime\b|\bgcd\b|\blcm\b|modul|factorial|fibonacci|combinat|permutat|divisor|\bmodulo\b",
    "bitwise": r"bitwise|\bxor\b|\bbit\b|binary representation|two's complement",
    "parse": r"\bparse\b|\bregex\b|tokeni|\bjson\b|\bformat\b|delimiter",
    "oop": r"\bclass\b|\bobject\b|\bmethod\b|attribute|instanc|inherit",
    "edge": r"edge case|corner case|empty (?:list|array|string|input)|\bnull\b|\bnone\b|invalid input",
    "constraint": r"constraint|1 *<=|<= *10|10\s*\^|10\*\*|1e[0-9]|at most|at least|\bn\s*<=",
}

CODE_FEATURE_NAMES: list[str] = [
    "len_chars", "len_words", "n_lines", "n_cases", "n_numbers", "max_number",
    "n_constraints", "n_defs", "n_qmarks", "has_code_fence", "has_examples",
    "n_kw_total",
] + ["kw_" + k for k in _KW_GROUPS]

# Heavy-tailed features get a log1p before standardisation.
_LOG_FEATURES = {
    "len_chars", "len_words", "n_lines", "n_cases", "n_numbers", "max_number",
    "n_constraints", "n_defs", "n_qmarks", "n_kw_total",
}
_DIM = len(CODE_FEATURE_NAMES)
_LOG_MASK = np.array([name in _LOG_FEATURES for name in CODE_FEATURE_NAMES])


def _numbers_in(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return out


def extract_code_features(problem: dict, n_cases: int = 0) -> np.ndarray:
    """Fixed-order feature vector for an opencode prompt dict."""
    p = problem.get("prompt", "") or ""
    pl = p.lower()
    words = p.split()
    nums = _numbers_in(p)
    kw = {k: int(bool(re.search(pat, pl))) for k, pat in _KW_GROUPS.items()}
    f = {
        "len_chars": float(len(p)),
        "len_words": float(len(words)),
        "n_lines": float(p.count("\n") + 1),
        "n_cases": float(n_cases),
        "n_numbers": float(len(nums)),
        "max_number": float(max(nums)) if nums else 0.0,
        "n_constraints": float(len(re.findall(r"<=|>=|10\s*\^|10\*\*|1e[0-9]", p))),
        "n_defs": float(len(re.findall(r"\bdef \b|\bfunction\b|\bclass \b", pl))),
        "n_qmarks": float(p.count("?")),
        "has_code_fence": float("```" in p or "    return " in p or ">>>" in p),
        "has_examples": float(bool(re.search(r"example|sample input|sample output|for instance|e\.g\.", pl))),
        "n_kw_total": float(sum(kw.values())),
    }
    f.update({"kw_" + k: float(kw[k]) for k in _KW_GROUPS})
    return np.array([f[name] for name in CODE_FEATURE_NAMES], dtype=np.float64)


class CodeFrontierModel:
    """Online logistic regression: P(prompt is in-zone | code features)."""

    def __init__(self, lr: float = 0.05, l2: float = 1e-4):
        self.dim = _DIM
        self.w = np.zeros(self.dim)
        self.b = 0.0
        self.mu = np.zeros(self.dim)
        self.sd = np.ones(self.dim)
        self.lr = lr
        self.l2 = l2
        self.n_pos = 0
        self.n_neg = 0
        self._lock = threading.Lock()

    def set_scaler(self, X: np.ndarray) -> None:
        Xt = self._logt(X)
        mu = Xt.mean(0)
        sd = Xt.std(0)
        sd[sd == 0] = 1.0
        self.mu, self.sd = mu, sd

    @staticmethod
    def _logt(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float64).copy()
        X[:, _LOG_MASK] = np.log1p(np.maximum(X[:, _LOG_MASK], 0.0))
        return X

    def _z(self, feats: np.ndarray) -> np.ndarray:
        x = self._logt(feats)[0]
        return (x - self.mu) / self.sd

    def score(self, feats: np.ndarray) -> float:
        with self._lock:
            z = self._z(feats)
            return 1.0 / (1.0 + math.exp(-(float(z @ self.w) + self.b)))

    def update(self, feats: np.ndarray, label: int, weight: float = 1.0) -> None:
        with self._lock:
            z = self._z(feats)
            p = 1.0 / (1.0 + math.exp(-(float(z @ self.w) + self.b)))
            g = p - label
            self.w -= self.lr * (weight * g * z + self.l2 * self.w)
            self.b -= self.lr * weight * g
            if label:
                self.n_pos += 1
            else:
                self.n_neg += 1

    def stats(self) -> str:
        return f"pos={self.n_pos} neg={self.n_neg}"

    def save(self, path: str) -> None:
        try:
            np.savez(path, w=self.w, b=np.array([self.b]), mu=self.mu, sd=self.sd,
                     n=np.array([self.n_pos, self.n_neg]), dim=np.array([self.dim]))
        except Exception:
            logger.exception("code-frontier save failed")

    def load(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            d = np.load(path)
            if int(d["dim"][0]) != self.dim:
                logger.warning("code-frontier dim mismatch (%d vs %d) -> ignoring stale model",
                               int(d["dim"][0]), self.dim)
                return False
            self.w, self.b = d["w"], float(d["b"][0])
            self.mu, self.sd = d["mu"], d["sd"]
            self.n_pos, self.n_neg = int(d["n"][0]), int(d["n"][1])
            return True
        except Exception:
            logger.exception("code-frontier load failed")
            return False


class CodeFrontierSampler:
    """candidate_sampler over an explicit pool, scored by CodeFrontierModel.

    record() mirrors frontier.FrontierSampler's (prompt_idx, n_correct, terminated)
    signature so it is a drop-in for pregen's _record_outcome path.
    """

    def __init__(self, env, pool_idxs, model: CodeFrontierModel, n_cases_fn,
                 *, epsilon: float = 0.35, subset: int = 1024,
                 save_path: str | None = None, priority_provider=None):
        self.env = env
        self.pool = list(pool_idxs)
        self._pool_set = set(self.pool)
        self.model = model
        self._n_cases_fn = n_cases_fn
        self.epsilon = epsilon
        self.subset = subset
        self.save_path = save_path
        self.priority_provider = priority_provider
        self._feat_cache: dict[int, np.ndarray] = {}
        self._updates = 0
        self._pos_buffer: list[np.ndarray] = []
        self._pos_weight = 20.0
        self._pos_buffer_cap = 1000

    def set_priority_provider(self, fn) -> None:
        self.priority_provider = fn

    def _features(self, idx: int) -> np.ndarray:
        f = self._feat_cache.get(idx)
        if f is None:
            try:
                f = extract_code_features(self.env.get_problem(idx), self._n_cases_fn(idx))
            except Exception:
                f = np.zeros(_DIM)
            if len(self._feat_cache) < 200_000:
                self._feat_cache[idx] = f
        return f

    def __call__(self, n: int, exclude: set) -> list:
        import random as _random
        pool = self.pool
        out: list = []
        seen: set = set()
        if self.priority_provider is not None:
            try:
                for idx in self.priority_provider():
                    idx = int(idx)
                    if idx in exclude or idx in seen or idx not in self._pool_set:
                        continue
                    seen.add(idx)
                    out.append(idx)
                    if len(out) >= n:
                        return out[:n]
            except Exception:
                logger.exception("code-frontier priority provider failed")
        exclude = exclude | seen
        cand: list = []
        tries = 0
        while len(cand) < self.subset and tries < self.subset * 4:
            idx = pool[_random.randrange(len(pool))]
            tries += 1
            if idx in exclude or idx in seen:
                continue
            seen.add(idx)
            cand.append(idx)
        remaining = n - len(out)
        if not cand or remaining <= 0:
            return out[:n]
        n_explore = max(0, int(round(remaining * self.epsilon)))
        n_exploit = remaining - n_explore
        scored = sorted(cand, key=lambda i: self.model.score(self._features(i)), reverse=True)
        out += scored[:n_exploit]
        rest = scored[n_exploit:]
        _random.shuffle(rest)
        out += rest[:n_explore]
        return out[:n]

    def record(self, prompt_idx: int, n_correct, terminated: int) -> None:
        """Online update. label=1 iff usable(>=8 term) AND in-zone(2..6 correct).

        The consensus path feeds the crisp sigma-derived positive by passing
        n_correct=4, terminated=8 when sigma>=gate (see pregen). All other
        callers (skips/failures with n_correct=None) fall through as negatives.
        """
        feats = self._features(prompt_idx)
        in_zone = bool(terminated >= 8 and n_correct is not None and 2 <= n_correct <= 6)
        if in_zone:
            self._pos_buffer.append(feats)
            if len(self._pos_buffer) > self._pos_buffer_cap:
                self._pos_buffer.pop(0)
            self.model.update(feats, 1, weight=self._pos_weight)
        else:
            self.model.update(feats, 0, weight=1.0)
            if self._pos_buffer and (self._updates % 5 == 0):
                import random as _r
                self.model.update(self._pos_buffer[_r.randrange(len(self._pos_buffer))],
                                  1, weight=self._pos_weight)
        self._updates += 1
        if self.save_path and self._updates % 200 == 0:
            self.model.save(self.save_path)

    def record_negative(self, prompt_idx: int) -> None:
        self.model.update(self._features(prompt_idx), 0, weight=1.0)
        self._updates += 1
        if self.save_path and self._updates % 200 == 0:
            self.model.save(self.save_path)


def build_code_frontier_sampler(env, pool_idxs, *, cases: dict | None,
                                save_path: str | None = None,
                                seed_positive_idxs=None, seed_epochs: int = 25,
                                priority_provider=None) -> CodeFrontierSampler:
    """Construct + warm-start a CodeFrontierSampler.

    cases: {prompt_idx: [case,...]} so n_cases is a feature. Positives seed from
    seed_positive_idxs (the validator's live cooldown_prompts = recent winners on
    the CURRENT checkpoint). Negatives are a random pool sample.
    """
    import random as _random

    cases = cases or {}

    def _n_cases(idx: int) -> int:
        c = cases.get(idx)
        return len(c) if c else 0

    model = CodeFrontierModel()
    n_env = len(env)

    pos_feats: list[np.ndarray] = []
    if seed_positive_idxs:
        for i in seed_positive_idxs:
            i = int(i)
            if 0 <= i < n_env:
                try:
                    pos_feats.append(extract_code_features(env.get_problem(i), _n_cases(i)))
                except Exception:
                    pass

    n_neg = max(len(pos_feats), 800)
    neg_feats: list[np.ndarray] = []
    for _ in range(n_neg):
        idx = pool_idxs[_random.randrange(len(pool_idxs))]
        try:
            neg_feats.append(extract_code_features(env.get_problem(idx), _n_cases(idx)))
        except Exception:
            pass

    scaler_rows = (pos_feats + neg_feats) if pos_feats else neg_feats
    if scaler_rows:
        model.set_scaler(np.vstack(scaler_rows))

    if pos_feats:
        data = [(f, 1.0) for f in pos_feats] + [(f, 0.0) for f in neg_feats]
        for _ in range(seed_epochs):
            _random.shuffle(data)
            for f, y in data:
                model.update(f, y)
        model.n_pos = model.n_neg = 0
        logger.info("code-frontier PRE-TRAINED on %d positives vs %d negatives",
                    len(pos_feats), len(neg_feats))

    if save_path and model.load(save_path):
        logger.info("code-frontier loaded from %s (%s)", save_path, model.stats())

    return CodeFrontierSampler(env, pool_idxs, model, _n_cases,
                               save_path=save_path, priority_provider=priority_provider)
