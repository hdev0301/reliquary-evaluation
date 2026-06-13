"""Frontier selection + σ screening for OpenMath (the ``EnvProducer``).

Same problem as OpenCode: for a well-trained policy the hard part is *finding*
the prompts at the model's learning frontier (σ ≥ 0.43), not generating. This
producer reuses the env-agnostic machinery from ``OpenCodeFrontier`` — the
σ-maximizing 8-subset selection, the p_stop-aware ``finalize_payloads`` (avoids
``BAD_TERMINATION``), the scoring and verdict feedback — and overrides only the
two env-specific pieces:

  * the difficulty proxy (math answer *shape*, not code test cases);
  * candidate proposal + screening, which grade by ANSWER EQUALITY against the
    dataset's public ``expected_answer`` (``MathOracle``). Because that's the
    same function the validator uses, the local σ ≈ the validator's σ, so
    ``out_of_zone`` should be far rarer than on OpenCode.

Plugs into ``mining.common.pregen.PregenPool`` exactly like the OpenCode
producer (``next_candidates`` + ``screen``; ``record_verdict`` feeds outcomes
back).
"""

from __future__ import annotations

import logging
import re

from reliquary.constants import M_ROLLOUTS, SIGMA_MIN

from mining.common.pregen import Candidate, ScreenResult
from mining.opencode.frontier import OpenCodeFrontier

logger = logging.getLogger(__name__)


class OpenMathFrontier(OpenCodeFrontier):
    """``EnvProducer`` for OpenMath pregen screening.

    Inherits ``__init__`` and the env-agnostic selection/finalize logic from
    ``OpenCodeFrontier``; ``oracle`` here is a ``MathOracle`` (answer matcher),
    and ``env`` is an ``OpenMathInstructEnvironment``.
    """

    _INT_RE = re.compile(r"-?\d+")
    _SMALL_INT_RE = re.compile(r"-?\d{1,3}")
    _RATIONAL_RE = re.compile(r"-?\d+/\d+|-?\d*\.\d+")

    _HARD_SOURCES = ("math", "augmented_math")

    def __init__(self, env, tokenizer, oracle, config, *, cooldown_getter, crowding=None) -> None:
        super().__init__(env, tokenizer, oracle, config, cooldown_getter=cooldown_getter, crowding=crowding)
        from mining.common.frontier_cache import FrontierCache

        self._source_weights = dict(getattr(config, "omi_source_weights", {}) or {})
        self._answer_weights = dict(getattr(config, "omi_answer_weights", {}) or {})
        self._hot_fraction = float(getattr(config, "hot_prompt_fraction", 0.0) or 0.0)
        self._checkpoint_n = 0
        self._cache = None
        cache_path = getattr(config, "frontier_cache_path", "")
        if cache_path:
            try:
                self._cache = FrontierCache(
                    cache_path,
                    hot_cap=int(getattr(config, "hot_prompt_cap", 6000)),
                    warm_decay=float(getattr(config, "frontier_warm_decay", 0.5)),
                )
            except Exception:
                logger.exception("frontier cache init failed; running without it")
                self._cache = None

    def set_checkpoint(self, checkpoint_n: int) -> None:
        self._checkpoint_n = int(checkpoint_n)

    def reset_for_checkpoint(self, checkpoint_n: int | None = None) -> None:
        """New checkpoint ⇒ snapshot the old bucket learning, then WARM-START
        the fresh learner from the (decayed) cache instead of cold-starting."""
        if checkpoint_n is not None:
            self._checkpoint_n = int(checkpoint_n)
        # Persist the OLD checkpoint's learning before clearing — but never let
        # an empty learner (e.g. at startup) overwrite cache loaded from disk.
        if self._cache is not None and self._buckets.n:
            try:
                self._cache.snapshot_buckets(self._buckets, self._checkpoint_n)
            except Exception:
                logger.exception("frontier cache snapshot failed")
        super().reset_for_checkpoint()
        if self._cache is not None:
            try:
                self._cache.seed_bucket_stats(self._buckets, self._checkpoint_n)
                self._cache.flush()
            except Exception:
                logger.exception("frontier cache warm-start failed")

    # ------------------------------------------------------------------
    # Difficulty proxy (overrides the code-test-count proxy)
    # ------------------------------------------------------------------
    def _answer_tier(self, ground_truth: str) -> int:
        """Answer-SHAPE tier 0..3 — the DOMINANT minability signal.

        Validated on the network's won-prompt distribution (/state cooldown set):
        latex/symbolic answers are 51% of wins vs 25% of the dataset (2.0x),
        decimals 2.6x, tuples/other 5x — while SMALL INTEGERS are 67% of the
        dataset but only 25% of wins (0.38x). A small integer has ONE canonical
        form (the model always boxes it identically → saturated, σ≈0); a
        fraction / decimal / symbolic answer has MANY equivalent surface forms
        the model mixes (200/7 vs 28.57, expanded vs factored) → σ-split →
        minable. Statement length is deliberately NOT used: won prompts are
        SHORT (median 155 chars) and the EOS-floor probe already drops long-CoT.
        """
        g = (ground_truth or "").strip()
        if not g or self._SMALL_INT_RE.fullmatch(g):
            return 0                        # small integer → one canonical form → saturated
        if self._INT_RE.fullmatch(g):
            return 1                        # large integer
        if self._RATIONAL_RE.fullmatch(g):
            return 2                        # fraction / decimal → several surface forms
        return 3                            # latex / symbolic / tuple → most surface forms

    def _difficulty(self, prompt: str, ground_truth: str) -> float:
        return float(self._answer_tier(ground_truth))

    def _answer_weight(self, ground_truth: str) -> float:
        """Per-answer-type sampling weight (rejection-sampling multiplier).

        Downsamples saturated small integers and upsamples the surface-form-rich
        answer types that actually win. Tunable via RELIQUARY_OMI_ANSWER_WEIGHTS.
        """
        return self._answer_weights.get(self._answer_tier(ground_truth), 1.0)

    @staticmethod
    def _is_numeric_gt(gt: str) -> bool:
        """True iff the ground truth is a CONCRETE NUMBER (int/fraction/decimal,
        or a numeric LaTeX value like \\frac{1}{2} / 2\\sqrt{3} with no free
        variables).

        Why this gate exists: ``_compute_omi_reward`` value-compares NUMBERS
        (so 1/2 == 0.5) but STRING-compares symbolic expressions (it cannot
        evaluate algebra with free variables). On a symbolic answer like
        ``4(x-a)^2+2(x-a)`` the model emits many algebraically-EQUIVALENT but
        differently-formatted boxed forms; the local string grader scores the
        reformatted ones "wrong" → a FALSE σ split → the group screens in-zone
        locally but the validator (which recognises the equivalence → all
        correct → σ=0) rejects it OUT_OF_ZONE. For numeric answers a wrong
        boxed answer differs in VALUE, which every grader version agrees on, so
        the local σ matches the validator. So mine numeric-answer prompts only.
        """
        from reliquary.environment.openmathinstruct import (
            _as_number, _expr_str_is_safe, _latex_to_pyexpr, _normalize_answer,
        )

        g = _normalize_answer(gt)
        if not g:
            return False
        if _as_number(g) is not None:
            return True
        eg = _latex_to_pyexpr(g)
        return eg is not None and _expr_str_is_safe(eg)

    def _bucket(self, prompt: str, ground_truth: str, source: str = "") -> int:
        """Difficulty bucket folding ANSWER SHAPE *and* dataset source.

        Hard sources (math/augmented_math) occupy buckets 6-11, easy sources
        (gsm8k/augmented_gsm8k) buckets 0-5. This separates the competition-hard
        math problems that happen to have small-integer answers — previously
        mis-bucketed as GSM8K-easy (b0) and starved by the σ sampler — into their
        own band so the learner can find them.
        """
        shape = min(int(self._difficulty(prompt, ground_truth)), 5)
        hard = 1 if source in self._HARD_SOURCES else 0
        return shape + 6 * hard

    # ------------------------------------------------------------------
    # EnvProducer.next_candidates
    # ------------------------------------------------------------------
    def next_candidates(self, n: int, *, exclude: set[int]) -> list[Candidate]:
        cooldown = self._cooldown_getter() or set()
        out: list[Candidate] = []
        seen: set[int] = set()

        # Hot-prompt EXPLOIT pool: prompt idxs that previously screened in-zone
        # for a nearby checkpoint. Re-tried first (the model's frontier drifts
        # slowly) so the miner does not cold-start re-discover it every reload.
        # Cooled hot prompts are won-and-gone → prune them from the cache.
        hot_pool: list[int] = []
        if self._cache is not None and self._hot_fraction > 0.0:
            hot = self._cache.hot_set()
            cooled = [i for i in hot if i in cooldown]
            if cooled:
                self._cache.drop(cooled)
            hot_pool = [i for i in hot if i not in exclude and i not in cooldown]
            self._rng.shuffle(hot_pool)

        def _make(idx: int) -> bool:
            problem = self.env.get_problem(idx)
            prompt = problem["prompt"]
            gt = str(problem.get("ground_truth", "") or "")
            if not gt:
                return False  # no public answer → can't predict σ locally
            src = str(problem.get("problem_source", "") or "")
            b = self._bucket(prompt, gt, src)
            try:
                prompt_tokens = self._encode(prompt)
            except Exception:
                return False
            out.append(Candidate(
                prompt_idx=idx, prompt_tokens=prompt_tokens,
                context={"ground_truth": gt, "bucket": b, "source": src},
            ))
            return True

        attempts = 0
        max_attempts = n * 80
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            # EXPLOIT: draw a known-hot prompt with prob hot_fraction.
            if hot_pool and self._rng.random() < self._hot_fraction:
                idx = hot_pool.pop()
                if idx in exclude or idx in cooldown or idx in seen:
                    continue
                seen.add(idx)
                _make(idx)
                continue
            # EXPLORE: source/bucket-weighted random sampling.
            idx = self._rng.randrange(self._n)
            if idx in exclude or idx in cooldown or idx in seen:
                continue
            problem = self.env.get_problem(idx)
            prompt = problem["prompt"]
            gt = str(problem.get("ground_truth", "") or "")
            if not gt:
                continue  # no public answer → can't predict σ locally
            src = str(problem.get("problem_source", "") or "")
            b = self._bucket(prompt, gt, src)
            # Acceptance ∝ bucket σ-weight × source weight × ANSWER-TYPE weight.
            # Answer type is the dominant minability signal (won-prompt analysis):
            # small integers are saturated (one canonical form) so they're heavily
            # downsampled, while fraction/decimal/symbolic answers (many surface
            # forms → σ-splits) are kept. Source weight skips saturated sources.
            sw = self._source_weights.get(src, 1.0) if self._source_weights else 1.0
            aw = self._answer_weight(gt)
            accept_p = min(1.0, self._buckets.weight(b) / 0.5) * sw * aw
            if self._rng.random() > accept_p:
                continue
            seen.add(idx)
            try:
                prompt_tokens = self._encode(prompt)
            except Exception:
                continue
            out.append(Candidate(
                prompt_idx=idx, prompt_tokens=prompt_tokens,
                context={"ground_truth": gt, "bucket": b, "source": src},
            ))
        return out

    # ------------------------------------------------------------------
    # EnvProducer.probe_keep — CONTENT-gated cheap probe (throughput)
    # ------------------------------------------------------------------
    # A well-trained policy is bimodal on OMI: it either solves a prompt (all-
    # EOS-pass) or fails it (all-wrong, mostly truncated). Only the rare ~4%
    # minable prompts split into a MIX of EOS-pass AND EOS-fail. The full
    # overgen×8192 budget is wasted on the ~96% saturated/too-hard prompts. This
    # probe grades the cheap probe and COMMITS the full budget only to prompts
    # whose probe already shows the usable mix, ~3x-ing prompts-screened/min.
    #
    # An earlier σ-probe was reverted for cold-start collapse — it gated on the
    # LEARNED σ-bucket weight, which starves to zero when nothing commits. This
    # version is immune: the keep decision is purely the probe's own measured
    # pass/fail content (independent of the learner), source-targeting already
    # restricts proposals to math/augmented_math, and an exploration floor still
    # commits a fraction of saturated-looking prompts so a small probe that
    # missed a borderline split is recovered by the full screen. Opt out with
    # RELIQUARY_OMI_SIGMA_PROBE=0 (falls back to the EOS-only probe).
    def probe_keep(self, candidate: Candidate, probe_rollouts, min_eos_frac: float) -> bool:
        if not getattr(self.config, "sigma_probe", True):
            # EOS-only fallback (the original robust path).
            if not probe_rollouts:
                return False
            eos_frac = sum(1 for r in probe_rollouts if r.finished_with_eos) / len(probe_rollouts)
            return eos_frac >= min_eos_frac
        if not probe_rollouts:
            return False
        eos_frac = sum(1 for r in probe_rollouts if r.finished_with_eos) / len(probe_rollouts)
        gt = candidate.context["ground_truth"]
        comps = [self.tokenizer.decode(r.tokens[r.prompt_length:]) for r in probe_rollouts]
        rewards = self.oracle.grade_completions(
            comps, gt, concurrency=getattr(self.config, "grade_concurrency", 8)
        )
        eos_pass = sum(1 for r, rw in zip(probe_rollouts, rewards) if r.finished_with_eos and rw >= 0.5)
        eos_fail = sum(1 for r, rw in zip(probe_rollouts, rewards) if r.finished_with_eos and rw < 0.5)
        # MINABLE signal: a clean σ≥0.43 group needs ≥6 EOS-passes AND ≥2 EOS-fails
        # with ≤1 truncated. A strong model that fails by TRUNCATING (rambling past
        # the cap, like gt=255) never yields a usable group — its fails are non-EOS.
        # So require the probe to show genuine EOS termination AND a usable EOS
        # pass+fail mix before spending the full overgen budget.
        if eos_frac >= min_eos_frac and eos_pass >= 1 and eos_fail >= 1:
            return True
        # Exploration floor: still commit a fraction so an 8-rollout probe that
        # missed a rare split is recovered downstream — but only if the prompt
        # actually terminates (else its group truncate-rejects anyway).
        if eos_frac >= min_eos_frac:
            return self._rng.random() < float(getattr(self.config, "probe_explore_floor", 0.08))
        return False

    # ------------------------------------------------------------------
    # EnvProducer.screen
    # ------------------------------------------------------------------
    def screen(self, candidate: Candidate, rollouts) -> ScreenResult:
        ctx = candidate.context
        ground_truth = ctx["ground_truth"]
        bucket = ctx["bucket"]
        completions = [
            self.tokenizer.decode(r.tokens[r.prompt_length:]) for r in rollouts
        ]
        # Grade ALL generated rollouts (N >> M_ROLLOUTS). Reward is binary
        # (1.0 correct answer / 0.0 wrong) and matches the validator exactly,
        # so the σ-maximizing 8-subset is in-zone for the validator too.
        all_rewards = self.oracle.grade_completions(
            completions, ground_truth,
            concurrency=getattr(self.config, "grade_concurrency", 8),
        )
        sel, reward_vec, sigma = self._select_best_eight(rollouts, all_rewards)
        k = sum(1 for x in reward_vec if x >= 0.5)
        n_truncated = sum(1 for i in sel if not rollouts[i].finished_with_eos)

        threshold = SIGMA_MIN + self.config.sigma_margin
        sigma_ok = len(reward_vec) == M_ROLLOUTS and sigma >= threshold
        trunc_ok = n_truncated <= 1  # validator MAX_TRUNCATED_PER_SUBMISSION=1
        shape_ok = not self._reward_shape_risk(reward_vec, [rollouts[i] for i in sel])
        in_zone = sigma_ok and trunc_ok and shape_ok
        if sigma >= 0.43:
            print(
                f"@@SCREEN sigma={sigma:.3f} len_rv={len(reward_vec)} thr={threshold:.3f} "
                f"margin={self.config.sigma_margin:.3f} sok={sigma_ok} ntrunc={n_truncated} "
                f"trunc_ok={trunc_ok} shape_ok={shape_ok} in_zone={in_zone} "
                f"rv={[round(x,2) for x in reward_vec]}", flush=True,
            )
            # DIAGNOSE out_of_zone: show exactly what the local grader saw for
            # each selected rollout (reward, extracted boxed answer, len, eos) vs
            # the ground truth — so a local-vs-validator grading gap is visible.
            if getattr(self.config, "debug_grade", False):
                from reliquary.environment.openmathinstruct import (
                    _last_boxed_only_string, _normalize_answer, _strip_boxed_wrapper,
                )
                ngt = _normalize_answer(ground_truth)
                src = candidate.context.get("source", "")
                print(f"@@GRADEDBG idx={candidate.prompt_idx} src={src} gt={ground_truth!r} norm_gt={ngt!r}", flush=True)
                for i in sel:
                    comp = self.tokenizer.decode(rollouts[i].tokens[rollouts[i].prompt_length:])
                    bx = _last_boxed_only_string(comp)
                    cand = _normalize_answer(_strip_boxed_wrapper(bx)) if bx else "<NOBOX>"
                    L = len(rollouts[i].tokens) - rollouts[i].prompt_length
                    print(
                        f"@@GRADEDBG r={all_rewards[i]:.0f} eos={int(rollouts[i].finished_with_eos)} "
                        f"len={L} cand={cand!r} boxed={(bx or '')[:80]!r}", flush=True,
                    )
        # Learn from the dense σ signal: steer future selection toward the
        # buckets at the model's frontier — but COST-AWARE on BOTH σ AND
        # termination. Weight each band's σ by its EOS fraction so the sampler
        # converges on SHORT-CoT, EOS-terminating frontier bands (the minable
        # sweet spot — top miners' winners are ~700 tok and EOS). A high-σ but
        # long-CoT band (low EOS → its in-zone groups truncate and get dropped)
        # is discounted toward 0 and sampled less; a saturated band (σ≈0) stays 0.
        eos_frac = sum(1 for r in rollouts if r.finished_with_eos) / max(1, len(rollouts))
        eff_sigma = sigma * eos_frac
        self._buckets.update(bucket, eff_sigma)
        # Remember σ-frontier prompts (cleared the zone floor locally) so the
        # next reload re-tries them first instead of cold-starting. Stores only
        # the idx + σ — re-screened against the live checkpoint before any fire.
        if self._cache is not None and sigma_ok:
            self._cache.record_hot(candidate.prompt_idx, sigma)
        self._screen_count += 1
        if self._screen_count % 64 == 0:
            tbl = " ".join(
                f"b{b}:{self._buckets.mean_sigma(b):.2f}({self._buckets.n.get(b,0)})"
                for b in sorted(self._buckets.n)
            )
            print(f"@@BUCKETS mean-σ by difficulty: {tbl}", flush=True)
            if self._cache is not None:
                self._cache.snapshot_buckets(self._buckets, self._checkpoint_n)
                self._cache.flush()

        reason = ""
        if sigma_ok and not in_zone:
            reason = (f"truncated={n_truncated}" if not trunc_ok else
                      "reward_shape" if not shape_ok else "?")

        # Candidate pools for the inherited p_stop-aware finalize (terminated
        # only). Binary rewards: pass = correct answer (1.0), fail = wrong (0.0).
        term = [i for i in range(len(rollouts)) if rollouts[i].finished_with_eos]
        pass_idx = sorted([i for i in term if all_rewards[i] >= 0.5],
                          key=lambda i: -all_rewards[i])
        fail_idx = sorted([i for i in term if all_rewards[i] < 0.5],
                          key=lambda i: all_rewards[i])

        score = self._score(sigma, k, n_truncated, candidate.prompt_idx)
        return ScreenResult(
            in_zone=in_zone, sigma=sigma, k_correct=float(k),
            reward_vec=reward_vec, score=score, reject_reason=reason,
            selected_indices=sel, pass_idx=pass_idx, fail_idx=fail_idx,
            all_rewards=all_rewards,
        )
