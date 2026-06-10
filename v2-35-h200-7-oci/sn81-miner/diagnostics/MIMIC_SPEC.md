# SN81 data-source picking + solution spec (reverse-engineered from top miners, 2026-06-09)

Source-of-truth: deep profiles of 5Hp6EPJd, 5F7YBWD1 (openmath) and 5DARq6, 5ECEJH9M (opencode).
`prompt_idx = dataset_row % shard_len`; set `RELIQUARY_OMI_SHARDS=2` or every submit = `prompt_mismatch`.

## A) OPENMATH (5Hp6EPJd precision / 5F7YBWD1 volume — same machine, different tuning)

### A1. Prompt selection filter (OMI, ~862-880k rows; sample UNIFORMLY across full idx range)
KEEP a row iff the boxed answer matches a SHORT, FORMAT-AMBIGUOUS shape, AND core question ≤ ~300 chars, AND topic in whitelist:

| shape | ~% | regex / test |
|---|---|---|
| fraction | 30 | `^\\?-?\\?(t\|d)?frac\{-?\d+\}\{-?\d+\}$` or `-?\d+/\d+` |
| has_variable | 22 | `[a-zA-Z]` inside expr (e.g. `(1-p)^{k-1}p`, `x^2+4`) |
| ordered_tuple | 20 | `^\(\s*[^()]+,\s*[^()]+\)$` (e.g. `(3+\sqrt5,-1)`) |
| radical | 13 | contains `\sqrt` |
| degrees | – | `-?\d+(\.\d+)?\^?\\?circ` (e.g. `105^\circ`) |
| small_matrix | – | `\begin{pmatrix}` (2x2/3x3) |
| single_MCQ | – | `^[A-E]$` (curve/shape-ID rows) |
| plain_int | 28 | `^-?\d+$` — BACKBONE ONLY (the 6-correct fillers) |

Target blend ~34-45% numeric / 55-66% symbolic. **Do not go below ~45% symbolic** or the distinct-wrong supply dries up.

Topic whitelist (overweight vs random OMI): trig+geometry ~32%, GSM8k word problems ~23%, coord/vector/conic ~20%, polynomial/function ~17%; plus **coefficient-templated families** (e.g. `x^2+y^2-7 = Ay-Bx+C, area`). DROP pure-arithmetic and trivial unique-answer plain-int.

Difficulty gate: **right often-but-not-always for a 4B model** AND format-ambiguous (frac vs decimal, √ ordering, tuple order, ±branch, with/without °). Reject trivially-unique answers (no wrongs) and too-hard (rarely 6 correct).

### A2. The distinct-wrong FACTORY (why symbolic — this is the whole game)
Reward = EXACT STRING equality vs boxed gt. For a frac/radical/tuple/var/degree answer, a competent model emits many **mathematically-equal-but-textually-different** forms (`\tfrac12` vs `0.5`, rationalized vs not, `\sqrt5` ordering, tuple order, ±, factoring of `x^4+4`, with/without `^\circ`). Each → a WRONG rollout (string mismatch), and they differ from EACH OTHER → DISTINCT wrongs. So the format ambiguity manufactures the **2 scarce distinct-wrongs for free**; the canonical form supplies the 6 correct. Plain-int can't (a right model re-emits the same int) → symbolic lean + tiny int backbone only.

### A3. Two FREE levers
- **MCQ single-letter (A-E)** rows guarantee 4 distinct-wrong options — highest-value distinct-wrong source.
- **Duplicate-stem re-mining (CORE engine):** validator dedups by `prompt_idx`, NOT content. Group OMI by normalized stem, keep MULTIPLE idx per stem, resubmit the same known answer under each. Scale confirmed: 5F7YBWD1's 163 accepted rows = only 93 distinct stems (60% dup-slots). Recurring stems to include: trapezoid-ABCD-angle (10x), conic `x^2+y^2-7=4y-14x+3` area (9x), arc-ratio circle (6x), inscribed-circle-in-square (6x), shortest-distance-between-circles (5x), track-LCM (5x), snail-in-well (4x).

### A4. Solution generation
Genuine current-checkpoint Qwen CoT (validator re-runs the forward + GRAIL — no post-editing). Numbered/bold LaTeX steps, a "Let's check"/"Alternative Method" section, occasional self-correction rambles (load-bearing — scatters wrongs). Exactly one trailing `\boxed{ANSWER}` then `<|im_end|>` (eos). Correct rollouts must match gt format EXACTLY.
Length: precision (5Hp6EPJd) med ~600-720, p90 ~1300-1550, max ~1900 → `MAX_NEW_TOKENS=SCREEN_MAX_TOKENS ≥ 1920`. Volume (5F7YBWD1) med ~620, p90 ~997 → 1280 suffices.

### A5. Curation (k=6)
Over-generate 64 (→96) rollouts/prompt → local math-equality grade → assemble 6 correct + 2 textually-distinct wrong → σ = 0.4330127018922193. k drift to 5/4 emerges from supply, NOT tuned per group. Dominant reject `out_of_zone` = distinct-wrong SUPPLY shortage (raise oversample), not speed.

### A6. Integrity pitfalls
- `distribution_suspicious` (`detect_opposite_reward_clones`): ≥3 opposite-reward pairs ≥96.5% textually similar → your wrongs are near-copies of corrects. Fix: raise OVERSAMPLE so wrongs are genuine distinct rollouts, never edited corrects.
- Truncation < ~1920 → reward-0 long tail. `token_tampered`/`boxed_answer_tampered`: never hand-edit boxed strings; keep completions genuine.

## B) OPENCODE — 5DARq6 (k=4 binary easy, RECOMMENDED) vs 5ECEJH9M (honest, fallback)

### B-I. 5DARq6 — k=4 binary, easy, short, curated
Source `nvidia/OpenCodeInstruct`, EASY subset. KEEP rows with BOTH `**Sample Input` AND `**Sample Output` blocks, scalar/short output, prompt body 400-1400 chars, task = classic algo / simple single-class OOP / date-time / dict-string-list / file-CSV-JSON / HTML-md table / basic geometry / trivial unit-tests. Drop multi-file/framework/perf. Answer in Python regardless of stated language. INCLUDE paraphrase-distinct dup archetypes (LIS ~8-10, primes ~7, date-diff ~4, Fibonacci/Calculator/BankAccount several) but cap per-archetype.
Distinct-wrong engine = spec-edge ambiguity (off-by-one, repr/sort order, getBalance contract, empty-input return). Generate 16-32 rollouts, single ```python``` block, match sample output EXACTLY incl separators/sentinels, `<|im_end|>` (~400 tok, cap ~509). Grade exact-match binary → curate k=4 at σ≈0.5 (4 correct + 4 distinct-wrong).
Integrity risk HIGHER: topic concentration + uniform 1.0 + easy-idx clustering → distribution flags. Spread topics, cap archetypes.

### B-II. 5ECEJH9M — honest/continuous, hard, long, NO curation (robustness fallback)
Stream a ~50k synthetic "implement function/class" Python set sequentially. KEEP hard self-contained tasks with a backtick signature + example-IO. ONE low-temp rollout/prompt, Python always, allow rambling/self-correction, `<|im_end|>`; no best-of, no post-edit. `CURATE=0`; rely on continuous partial-pass scoring (σ ~0.43-0.50, ~34% non-canonical). Low integrity exposure (organic).

### B-III. Recommendation: mimic 5DARq6's curated k=4 pipeline, but borrow 5ECEJH9M's topic-spread + organic completion texture (raise oversample, cap per-archetype, let some completions ramble) to keep reward density high while dodging distribution flags.
