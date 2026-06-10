"""Online frontier predictor for prompt selection.

The zone filter rewards groups with 2..6 of 8 correct (high reward variance).
On a converged checkpoint, easy prompts give 8/8 and hard prompts ramble (no
EOS) or give 0/8 — the in-zone frontier is narrow and DRIFTS as the policy
changes. ``winners.jsonl`` encodes an *earlier* checkpoint's frontier, so it is
only a weak prior; the dominant signal is ONLINE — every group the pregenerator
builds is a labelled observation (in-zone vs not) on the CURRENT checkpoint.

This module:
  * extracts cheap text features from a problem (mirrors feat_analysis.py),
  * holds an online logistic-regression ``FrontierModel`` (numpy, no sklearn),
  * exposes a ``FrontierSampler`` used as the pregenerator's candidate_sampler:
    score a random candidate subset, exploit the top (with ε-greedy
    exploration), and learn from each observed outcome via ``record()``.

Selection is on PREDICTED-FRONTIER-PROBABILITY only — it never inspects or
shapes the rewards of a submitted group, so it stays within the rules (prompt
selection is explicitly allowed).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading

import numpy as np

logger = logging.getLogger(__name__)

# --- feature extraction (kept byte-compatible with feat_analysis.py) ---
_NUM_RE = re.compile(r"(?<![\w.])(\d[\d,]*\.?\d*|\.\d+)(?![\w])")
_KW = {
    "twice": r"\btwice\b", "half": r"\bhalf|one[- ]half\b", "each": r"\beach\b",
    "total": r"\btotal\b", "remaining": r"\bremain", "percent": r"percent|%",
    "more_than": r"\bmore than\b", "less_than": r"\bless than\b", "fewer": r"\bfewer\b",
    "increase": r"\bincrease", "decrease": r"\bdecrease", "average": r"\baverage|\bmean\b",
    "ratio": r"\bratio\b", "per": r"\bper\b", "discount": r"\bdiscount", "tax": r"\btax\b", "tip": r"\btip\b",
}
_STEM = {
    "how_many": r"how many", "how_much": r"how much", "what_is": r"what is|what was|what would",
    "find": r"\bfind\b", "calculate": r"\bcalculate\b",
}
FEATURE_NAMES: list[str] = (
    ["len_chars", "len_words", "n_numeric_tokens", "max_number", "has_decimal",
     "has_percent", "has_fraction", "has_money", "gt_is_integer", "gt_is_decimal",
     "gt_is_text", "gt_magnitude", "gt_num_digits", "gt_has_dot",
     "kw_any_arith", "kw_count_arith"]
    + ["kw_" + k for k in _KW] + ["stem_" + k for k in _STEM]
)
_DIM = len(FEATURE_NAMES)


def _numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.finditer(text):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return out


def _gt_classify(g: str):
    g = str(g).strip()
    gn = g.replace(",", "").replace("$", "").replace("%", "").strip()
    hasdot = "." in gn
    try:
        v = float(gn)
        if not math.isfinite(v):
            # e.g. a googol-sized integer or literal "inf" -> not a usable number;
            # classify as text so round()/abs() never see infinity (OverflowError).
            return 0, 0, 1, 0.0, 0.0, int(hasdot)
        is_int = abs(v - round(v)) < 1e-9 and not hasdot
        return int(is_int), int(not is_int), 0, abs(v), len(re.sub(r"\D", "", gn)), int(hasdot)
    except (ValueError, OverflowError):
        return 0, 0, 1, 0.0, 0.0, int(hasdot)


def extract_features(problem: dict) -> np.ndarray:
    """Return a fixed-order feature vector for an env problem dict."""
    p = problem.get("prompt", "")
    pl = p.lower()
    words = p.split()
    nums = _numbers_in(p)
    kw = {k: int(bool(re.search(pat, pl))) for k, pat in _KW.items()}
    is_int, is_dec, is_text, mag, ndig, hasdot = _gt_classify(problem.get("ground_truth", ""))
    f = {
        "len_chars": len(p), "len_words": len(words), "n_numeric_tokens": len(nums),
        "max_number": max(nums) if nums else 0.0,
        "has_decimal": int(bool(re.search(r"\d+\.\d+", p))),
        "has_percent": int("%" in p or "percent" in pl),
        "has_fraction": int(bool(re.search(r"\b\d+/\d+\b|one[- ](third|fourth|fifth|half|quarter)|two[- ]thirds|three[- ]quarters", pl))),
        "has_money": int("$" in p),
        "gt_is_integer": is_int, "gt_is_decimal": is_dec, "gt_is_text": is_text,
        "gt_magnitude": mag, "gt_num_digits": ndig, "gt_has_dot": hasdot,
        "kw_any_arith": int(any(kw.values())), "kw_count_arith": sum(kw.values()),
    }
    f.update({"kw_" + k: kw[k] for k in _KW})
    f.update({"stem_" + k: int(bool(re.search(pat, pl))) for k, pat in _STEM.items()})
    vec = np.array([float(f[name]) for name in FEATURE_NAMES], dtype=np.float64)
    # A non-finite feature (e.g. max_number/gt_magnitude from a huge value) would
    # propagate NaN through the SGD weights; clamp inf/nan to 0.0.
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


class FrontierModel:
    """Online logistic regression: P(group is in-zone | prompt features).

    Standardisation stats come from a reference sample (winners+controls or a
    random env sample). Weights start at zero (uniform 0.5 score) and are
    updated by SGD from each observed (features, in_zone) pair, so the model
    converges to the CURRENT checkpoint's frontier regardless of the stale prior.
    """

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
        mu = X.mean(0)
        sd = X.std(0)
        sd[sd == 0] = 1.0
        self.mu, self.sd = mu, sd

    def _z(self, feats: np.ndarray) -> np.ndarray:
        # log1p the heavy-tailed magnitude/length features before standardising.
        x = feats.copy()
        for i, name in enumerate(FEATURE_NAMES):
            if name in ("len_chars", "len_words", "max_number", "gt_magnitude", "n_numeric_tokens"):
                x[i] = math.log1p(max(x[i], 0.0))
        return (x - self.mu) / self.sd

    def score(self, feats: np.ndarray) -> float:
        with self._lock:
            z = self._z(feats)
            return 1.0 / (1.0 + math.exp(-(float(z @ self.w) + self.b)))

    def update(self, feats: np.ndarray, label: int, weight: float = 1.0) -> None:
        # weight>1 upweights the minority (in-zone) class so the rare positives
        # are not drowned by the ~99% negative stream under single-pass SGD.
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
                     n=np.array([self.n_pos, self.n_neg]))
        except Exception:
            logger.exception("frontier model save failed")

    def load(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            d = np.load(path)
            self.w, self.b = d["w"], float(d["b"][0])
            self.mu, self.sd = d["mu"], d["sd"]
            self.n_pos, self.n_neg = int(d["n"][0]), int(d["n"][1])
            return True
        except Exception:
            logger.exception("frontier model load failed")
            return False


class FrontierSampler:
    """candidate_sampler that scores prompts with a FrontierModel and balances
    exploit (top-scoring) vs explore (random), learning online via record()."""

    def __init__(self, env, pool_idxs: list[int], model: FrontierModel,
                 *, epsilon: float = 0.35, subset: int = 1024, save_path: str | None = None,
                 priority_provider=None):
        self.env = env
        self.pool = pool_idxs
        # DECOOL-SNIPE: optional callable -> list[int] of freshly-decooled idxs.
        # Restricted to our in-zone pool so a snipe never drags us out of the
        # working numeric category; an empty intersection just falls through to
        # the frontier ranking. None disables the front-load entirely.
        self.priority_provider = priority_provider
        self._pool_set = set(pool_idxs) if pool_idxs else set()
        self.model = model
        self.epsilon = epsilon
        self.subset = subset
        self.save_path = save_path
        self._feat_cache: dict[int, np.ndarray] = {}
        self._updates = 0
        # Positive (in-zone) replay buffer + minority upweight to beat the
        # ~1% positive-class imbalance with single-pass online SGD.
        self._pos_buffer: list[np.ndarray] = []
        self._pos_weight = 20.0
        self._pos_buffer_cap = 1000

    def _features(self, idx: int) -> np.ndarray:
        f = self._feat_cache.get(idx)
        if f is None:
            f = extract_features(self.env.get_problem(idx))
            if len(self._feat_cache) < 200_000:
                self._feat_cache[idx] = f
        return f

    def __call__(self, n: int, exclude: set[int]) -> list[int]:
        import random as _random
        pool = self.pool
        out: list[int] = []
        seen: set[int] = set()
        # DECOOL-SNIPE FRONT-LOAD: freshly-decooled prompts that are in our
        # in-zone pool are recently-rewarded (in-zone) AND just-freed — other
        # miners haven't re-discovered them, so submitting one early claims a
        # DISTINCT seal slot that isn't being raced. Jump these to the front of
        # the frontier ranking; fall through to the learned ranking for the rest.
        if self.priority_provider is not None:
            try:
                for idx in self.priority_provider():
                    idx = int(idx)
                    if idx in exclude or idx in seen or idx not in self._pool_set:
                        continue
                    seen.add(idx)
                    out.append(idx)
                    if len(out) >= n:
                        logger.debug("decool-snipe front-loaded %d idxs (full)", len(out))
                        return out[:n]
            except Exception:
                logger.exception("decool priority provider failed; frontier-only this draw")
        if out:
            logger.debug("decool-snipe front-loaded %d/%d idxs", len(out), n)
        # draw a random candidate subset, score, then split exploit/explore.
        cand = []
        exclude = exclude | seen  # don't re-pick the sniped idxs in the ranking
        tries = 0
        while len(cand) < self.subset and tries < self.subset * 4:
            idx = pool[_random.randrange(len(pool))]
            tries += 1
            if idx in exclude or idx in seen:
                continue
            seen.add(idx)
            cand.append(idx)
        remaining = n - len(out)  # slots left after the decool-snipe front-load
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

    def record(self, prompt_idx: int, n_correct: int | None, terminated: int) -> None:
        """Update the model from a group's outcome on the current checkpoint.

        label=1 (good) iff the group was usable (>=8 terminated) AND in-zone
        (2..6 correct). Everything else — too-easy 8/8, all-wrong 0/8, or
        ramble (<8 terminated) — is label=0 (don't waste generation there).
        """
        feats = self._features(prompt_idx)
        # BINARY label: 1 iff usable (>=8 terminated) AND in-zone (2..6 correct);
        # everything else (8/8 too-easy, 0/8, or <8-terminated ramble) = 0. A
        # graded label dilutes the already-rare positive signal, so we keep it
        # crisp and instead fight the class imbalance via upweighting + replay.
        in_zone = bool(terminated >= 8 and n_correct is not None and 2 <= n_correct <= 6)
        if in_zone:
            self._pos_buffer.append(feats)
            if len(self._pos_buffer) > self._pos_buffer_cap:
                self._pos_buffer.pop(0)
            self.model.update(feats, 1, weight=self._pos_weight)
        else:
            self.model.update(feats, 0, weight=1.0)
            # Replay a stored positive every few negatives so the rare in-zone
            # signature is not forgotten under the negative-dominated stream.
            if self._pos_buffer and (self._updates % 5 == 0):
                import random as _r
                self.model.update(self._pos_buffer[_r.randrange(len(self._pos_buffer))],
                                  1, weight=self._pos_weight)
        self._updates += 1
        if self.save_path and self._updates % 200 == 0:
            self.model.save(self.save_path)

    def record_negative(self, prompt_idx: int) -> None:
        """Force a label-0 update from a CHEAP screen rejection.

        Unlike ``record()``, this NEVER derives an in-zone positive: the cheap
        two-stage screen has no reliable 8-sample in-zone estimate, so it must
        only ever teach the model a negative. The crisp label-1 comes solely
        from the full deep-mine outcome (Pregenerator.build_groups -> record()).
        Keeping the label explicit here (rather than re-deriving from counts)
        means screen knob choices can never accidentally train a positive.
        """
        self.model.update(self._features(prompt_idx), 0, weight=1.0)
        self._updates += 1
        if self.save_path and self._updates % 200 == 0:
            self.model.save(self.save_path)


def build_frontier_sampler(env, pool_idxs, *, winners_path: str | None,
                           controls_path: str | None, save_path: str | None,
                           seed_positive_idxs: list[int] | None = None,
                           seed_epochs: int = 25,
                           priority_provider=None) -> FrontierSampler:
    """Construct a FrontierSampler and give it a CURRENT-checkpoint prior.

    Positive examples (the frontier signature) come, in priority order, from:
      1. ``seed_positive_idxs`` — the validator's live ``cooldown_prompts``
         (recently-winning prompts on the CURRENT checkpoint). Best signal.
      2. ``winners.jsonl`` rows — an earlier checkpoint's frontier (weak prior).
    Negatives are a random sample of the candidate pool (mostly non-frontier).
    The model is pre-trained on these, then refined online from live outcomes.
    """
    import random as _random
    model = FrontierModel()
    n_env = len(env)

    pos_feats: list[np.ndarray] = []
    pos_src = "none"
    if seed_positive_idxs:
        for i in seed_positive_idxs:
            if 0 <= i < n_env:
                pos_feats.append(extract_features(env.get_problem(i)))
        pos_src = "live cooldown winners"
    if not pos_feats and winners_path and os.path.exists(winners_path):
        try:
            rows = [json.loads(l) for l in open(winners_path) if l.strip()]
            # Prefer env.get_problem(idx) so the seed positives carry the SAME
            # answer-format suffix the live candidates do. The winners-row 'prompt'
            # is the raw question WITHOUT the env's "\n\nPut your final answer
            # within \boxed{}." suffix, which otherwise injects a spurious
            # length bias (len_chars/len_words) on the cold-start prior.
            pos_feats = [
                extract_features(env.get_problem(int(r["idx"])))
                if isinstance(r, dict) and "idx" in r and 0 <= int(r["idx"]) < n_env
                else extract_features(r)
                for r in rows
            ]
            pos_src = "winners.jsonl (stale, idx-reconstructed)"
        except Exception:
            logger.exception("could not read %s", winners_path)

    n_neg = max(len(pos_feats), 800)
    neg_feats = [extract_features(env.get_problem(pool_idxs[_random.randrange(len(pool_idxs))]))
                 for _ in range(n_neg)]

    def _t(X):
        X = X.copy()
        for j, name in enumerate(FEATURE_NAMES):
            if name in ("len_chars", "len_words", "max_number", "gt_magnitude", "n_numeric_tokens"):
                X[:, j] = np.log1p(np.maximum(X[:, j], 0.0))
        return X

    scaler_rows = (pos_feats + neg_feats) if pos_feats else neg_feats
    model.set_scaler(_t(np.vstack(scaler_rows)))

    if pos_feats:
        data = [(f, 1.0) for f in pos_feats] + [(f, 0.0) for f in neg_feats]
        for _ in range(seed_epochs):
            _random.shuffle(data)
            for f, y in data:
                model.update(f, y)
        # reset the telemetry counters so online pos/neg reflect live learning
        model.n_pos = model.n_neg = 0
        logger.info("frontier PRE-TRAINED on %d positives (%s) vs %d negatives",
                    len(pos_feats), pos_src, len(neg_feats))

    if save_path and model.load(save_path):
        logger.info("frontier model loaded from %s (%s)", save_path, model.stats())
    return FrontierSampler(env, pool_idxs, model, save_path=save_path,
                           priority_provider=priority_provider)
