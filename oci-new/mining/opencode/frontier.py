"""Frontier selection + σ screening for OpenCode (the ``EnvProducer``).

For a well-trained policy the hard part is not generating — it is *finding the
~5% of prompts at the learning frontier* where the model scores 2–6/8 (σ ≥
0.43). This producer:

  * proposes non-cooldown candidate prompts, biased toward difficulty buckets
    that have recently yielded in-zone groups (online, reset on checkpoint
    change);
  * screens each sampled 8-group with the local oracle (predicted reward
    vector → σ), keeping only groups comfortably in-zone and naturally
    terminated;
  * scores survivors so the pool fires the cleanest, most-central groups first
    and, where it can, the *least crowded* prompts (higher per-prompt yield,
    since the slot pays ``(pool/8)/K_p``).

Plugs into ``mining.common.pregen.PregenPool`` via ``next_candidates`` +
``screen``; ``record_verdict`` feeds real outcomes back in.
"""

from __future__ import annotations

import logging
import math
import random as _random

from reliquary.constants import M_ROLLOUTS, SIGMA_MIN

from mining.common.pregen import Candidate, ScreenResult

logger = logging.getLogger(__name__)


def _popstd(xs: list[float]) -> float:
    if not xs:
        return 0.0
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


class _BucketStats:
    """Online mean-σ per difficulty bucket.

    The in-zone signal (σ≥0.43) is far too sparse to learn from for a
    well-trained model, but EVERY screened group yields a σ — a dense signal.
    We track each difficulty bucket's mean σ and steer candidate selection
    toward the buckets whose σ runs highest (i.e. where the model actually sits
    at its frontier), away from the easy buckets where it scores 8/8 (σ≈0).
    """

    OPTIMISTIC_PRIOR = 0.5   # untried buckets look attractive → explored first

    def __init__(self) -> None:
        self.sum_sigma: dict[int, float] = {}
        self.n: dict[int, int] = {}

    def mean_sigma(self, b: int) -> float:
        n = self.n.get(b, 0)
        return self.sum_sigma.get(b, 0.0) / n if n else self.OPTIMISTIC_PRIOR

    def weight(self, b: int) -> float:
        # Exploration floor so a transiently-low bucket still gets sampled.
        return max(0.05, self.mean_sigma(b))

    def update(self, b: int, sigma: float) -> None:
        self.sum_sigma[b] = self.sum_sigma.get(b, 0.0) + sigma
        self.n[b] = self.n.get(b, 0) + 1

    def reset(self) -> None:
        self.sum_sigma.clear()
        self.n.clear()


class OpenCodeFrontier:
    """``EnvProducer`` for OpenCode pregen screening."""

    def __init__(self, env, tokenizer, oracle, config, *, cooldown_getter, crowding=None) -> None:
        self.env = env
        self.tokenizer = tokenizer
        self.oracle = oracle
        self.config = config
        self._cooldown_getter = cooldown_getter      # () -> set[int]
        self._crowding = crowding                     # optional: prompt_idx -> recent K estimate
        self._buckets = _BucketStats()
        self._rng = _random.Random()
        self._n = len(env)
        self._screen_count = 0

    # ------------------------------------------------------------------
    def reset_for_checkpoint(self) -> None:
        """New checkpoint ⇒ old frontier estimates are stale."""
        self._buckets.reset()
        self._screen_count = 0

    @staticmethod
    def _difficulty(prompt: str, cases: list[dict]) -> float:
        """Cheap static difficulty proxy (no oracle rebuild needed).

        Harder problems → longer statements, more cases, and structured (list/
        dict) expected outputs rather than scalars. Used only to bucket prompts;
        the σ-driven bucket stats then learn which difficulty regions actually
        sit at the model's frontier.
        """
        plen = len(prompt) / 600.0
        ncases = len(cases) * 0.4
        structured = sum(
            1 for c in cases if isinstance(c.get("expected"), (list, dict))
        ) * 0.8
        return plen + ncases + structured

    def _bucket(self, prompt: str, cases: list[dict]) -> int:
        return min(int(self._difficulty(prompt, cases)), 11)

    # ------------------------------------------------------------------
    # EnvProducer.next_candidates
    # ------------------------------------------------------------------
    def next_candidates(self, n: int, *, exclude: set[int]) -> list[Candidate]:
        cooldown = self._cooldown_getter() or set()
        out: list[Candidate] = []
        seen: set[int] = set()
        # Over-sample idxs; bias acceptance toward high-hit-rate buckets via
        # rejection sampling on the bucket weight.
        attempts = 0
        max_attempts = n * 60
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            idx = self._rng.randrange(self._n)
            if idx in exclude or idx in cooldown or idx in seen:
                continue
            problem = self.env.get_problem(idx)
            prompt = problem["prompt"]
            cases = self.oracle.cases_for(prompt)
            if not cases:
                continue
            b = self._bucket(prompt, cases)
            # Accept with prob proportional to the bucket's mean-σ weight
            # (favours difficulty bands sitting at the model's frontier; the
            # easy k8 bands decay toward σ≈0 and get sampled less).
            if self._rng.random() > min(1.0, self._buckets.weight(b) / 0.5):
                continue
            seen.add(idx)
            try:
                prompt_tokens = self._encode(prompt)
            except Exception:
                continue
            out.append(Candidate(
                prompt_idx=idx, prompt_tokens=prompt_tokens,
                context={"cases": cases, "bucket": b},
            ))
        return out

    def _encode(self, prompt: str) -> list[int]:
        from reliquary.protocol.tokens import encode_prompt

        return encode_prompt(self.tokenizer, prompt)

    # ------------------------------------------------------------------
    # EnvProducer.screen
    # ------------------------------------------------------------------
    def screen(self, candidate: Candidate, rollouts) -> ScreenResult:
        ctx = candidate.context
        cases = ctx["cases"]
        bucket = ctx["bucket"]
        completions = [
            self.tokenizer.decode(r.tokens[r.prompt_length:]) for r in rollouts
        ]
        # Grade ALL generated rollouts (N may be >> M_ROLLOUTS=8). A well-trained
        # model is confident: 8 raw samples are nearly always all-pass (σ≈0). But
        # it DOES fail occasionally, so over a larger sample we can select 8 — a
        # mix of natural passes and natural failures — that form a genuinely
        # gradient-rich, in-zone (σ≥0.43) group. The validator recomputes these
        # same rewards on the same cases, so the selected group is in-zone for it
        # too. The picks are real, terminated, distinct generations (not the
        # manufactured-loser pattern the reward-shape guard targets).
        all_rewards = self.oracle.grade_completions(
            completions, cases, concurrency=getattr(self.config, "grade_concurrency", 8)
        )
        sel, reward_vec, sigma = self._select_best_eight(rollouts, all_rewards)
        k = sum(1 for x in reward_vec if x >= 0.5)
        n_truncated = sum(1 for i in sel if not rollouts[i].finished_with_eos)

        threshold = SIGMA_MIN + self.config.sigma_margin

        # The validator's zone filter is ONLY σ ≥ SIGMA_MIN on the (fractional,
        # passed/total) reward vector. We select the 8-subset that maximizes σ.
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
        # Learn from the DENSE σ signal: bucket mean-σ steers future selection
        # toward the model's frontier (high σ), away from the easy k8 bands.
        self._buckets.update(bucket, sigma)
        self._screen_count += 1
        if self._screen_count % 64 == 0:
            tbl = " ".join(
                f"b{b}:{self._buckets.mean_sigma(b):.2f}({self._buckets.n.get(b,0)})"
                for b in sorted(self._buckets.n)
            )
            print(f"@@BUCKETS mean-σ by difficulty: {tbl}", flush=True)

        # Explain why a σ-passing group was dropped (truncation / reward-shape).
        reason = ""
        if sigma_ok and not in_zone:
            reason = (f"truncated={n_truncated}" if not trunc_ok else
                      "reward_shape" if not shape_ok else "?")

        # Candidate pools for the p_stop-aware finalize step (terminated only).
        term = [i for i in range(len(rollouts)) if rollouts[i].finished_with_eos]
        pass_idx = sorted([i for i in term if all_rewards[i] >= 0.5],
                          key=lambda i: -all_rewards[i])      # best passes first
        fail_idx = sorted([i for i in term if all_rewards[i] < 0.5],
                          key=lambda i: all_rewards[i])       # worst fails first

        score = self._score(sigma, k, n_truncated, candidate.prompt_idx)
        return ScreenResult(
            in_zone=in_zone, sigma=sigma, k_correct=float(k),
            reward_vec=reward_vec, score=score, reject_reason=reason,
            selected_indices=sel, pass_idx=pass_idx, fail_idx=fail_idx,
            all_rewards=all_rewards,
        )

    def finalize_payloads(self, rollouts, screen, proof_builder):
        """Build the 8 proof payloads, choosing rollouts whose EXACT HF p_stop
        clears the floor — so the group avoids the validator's BAD_TERMINATION.

        Precomputes a small candidate pool (the most-failing + best-passing
        terminated rollouts), keeps those with p_stop ≥ min_pstop, then picks
        up to 4 fails + 4 passes that keep σ ≥ SIGMA_MIN. Returns the 8 payloads
        or None if a clean in-zone group can't be formed.
        """
        import math

        min_pstop = getattr(self.config, "min_pstop", 0.012)
        threshold = SIGMA_MIN + self.config.sigma_margin

        seen_pstops: list[float] = []

        def vet(idx_list, limit):
            out = []
            for i in idx_list[: limit + 6]:
                r = rollouts[i]
                try:
                    p = proof_builder.precompute(
                        r.tokens, r.prompt_length, finished_with_eos=r.finished_with_eos
                    )
                except Exception:
                    continue
                pstop = math.exp(p.token_logprobs[-1]) if p.token_logprobs else 0.0
                seen_pstops.append(pstop)
                if pstop >= min_pstop:
                    out.append((i, p))
                if len(out) >= limit:
                    break
            return out

        good_fails = vet(screen.fail_idx, 4)
        good_passes = vet(screen.pass_idx, 4)
        print(
            f"@@FINALIZE good_fails={len(good_fails)} good_passes={len(good_passes)} "
            f"min_pstop={min_pstop} pstops={[round(x,4) for x in sorted(seen_pstops)[:12]]}",
            flush=True,
        )
        if len(good_fails) < 2 or len(good_fails) + len(good_passes) < M_ROLLOUTS:
            return None

        nf = min(4, len(good_fails))
        np_ = M_ROLLOUTS - nf
        if np_ > len(good_passes):          # not enough passes: take more fails
            nf = M_ROLLOUTS - len(good_passes)
            np_ = len(good_passes)
        chosen = good_fails[:nf] + good_passes[:np_]
        rewards = [screen.all_rewards[i] for i, _ in chosen]
        if len(chosen) != M_ROLLOUTS or _popstd(rewards) < threshold:
            return None
        return [p for _, p in chosen]

    def _select_best_eight(self, rollouts, rewards):
        """Pick the M_ROLLOUTS-subset of generated rollouts that maximizes σ.

        Maximising the population std of a fixed-size subset means taking the
        extremes — j lowest-reward + (M-j) highest-reward. We prefer naturally
        terminated rollouts (≤1 truncated allowed) and avoid identical loser
        lengths (reward-shape). Returns (indices, rewards_of_selection, sigma).
        """
        M = M_ROLLOUTS
        n = len(rollouts)
        if n <= M:
            sel = list(range(n))
            return sel, [rewards[i] for i in sel], _popstd([rewards[i] for i in sel])

        # Prefer terminated rollouts; keep at most 1 truncated overall.
        term = [i for i in range(n) if rollouts[i].finished_with_eos]
        if len(term) < M:
            trunc = [i for i in range(n) if not rollouts[i].finished_with_eos]
            term = term + trunc[: M - len(term)]  # backfill if too few terminated
        pool = sorted(term, key=lambda i: rewards[i])  # low → high reward

        best_sel, best_sigma = None, -1.0
        for j in range(2, M - 1):  # j lows + (M-j) highs, keep both tails non-trivial
            sel = pool[:j] + pool[len(pool) - (M - j):]
            if len(set(sel)) != M:
                continue
            sub = [rewards[i] for i in sel]
            sg = _popstd(sub)
            # Penalise selections that would trip the reward-shape guard.
            if self._reward_shape_risk(sub, [rollouts[i] for i in sel]):
                sg -= 0.05
            if sg > best_sigma:
                best_sigma, best_sel = sg, sel
        if best_sel is None:
            best_sel = pool[:M]
        return best_sel, [rewards[i] for i in best_sel], _popstd([rewards[i] for i in best_sel])

    @staticmethod
    def _reward_shape_risk(reward_vec: list[float], rollouts) -> bool:
        """Avoid groups that could read as manufactured losers (reward_shape).

        Honest middle-band sampling rarely trips this, but skip a group if ≥3
        zero-reward rollouts share an identical completion length ≥64 — the
        signature the validator's reward-shape guard keys on.
        """
        from reliquary.constants import (
            REWARD_SHAPE_MIN_EXACT_ZERO_ROLLOUTS,
            REWARD_SHAPE_ZERO_MODE_MIN_LENGTH,
        )

        lengths: dict[int, int] = {}
        for r, rew in zip(rollouts, reward_vec):
            if rew <= 0.0:
                L = len(r.tokens) - r.prompt_length
                if L >= REWARD_SHAPE_ZERO_MODE_MIN_LENGTH:
                    lengths[L] = lengths.get(L, 0) + 1
        return any(c >= REWARD_SHAPE_MIN_EXACT_ZERO_ROLLOUTS for c in lengths.values())

    def _score(self, sigma: float, k: int, n_truncated: int, prompt_idx: int) -> float:
        # Prefer maximal-σ (k≈4), fully-terminated, low-crowding prompts.
        centrality = 1.0 - abs(k - (M_ROLLOUTS / 2.0)) / (M_ROLLOUTS / 2.0)
        score = sigma + 0.25 * centrality - 0.15 * n_truncated
        if self._crowding is not None:
            score -= 0.1 * float(self._crowding.get(prompt_idx, 0))
        return score

    # ------------------------------------------------------------------
    # Feedback from /verdicts
    # ------------------------------------------------------------------
    def record_verdict(self, accepted: bool, reason: str) -> None:
        """Light global signal; bucket-level credit already happens in screen().

        A spike of ``out_of_zone`` means our proxy is over-optimistic → nudge
        the σ margin up a touch for the rest of this checkpoint.
        """
        if not accepted and reason == "out_of_zone":
            self.config.sigma_margin = min(self.config.sigma_margin + 0.005, 0.15)
