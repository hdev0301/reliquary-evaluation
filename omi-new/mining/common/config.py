"""Runtime configuration for the optimized miner.

All knobs are local *performance/strategy* choices — none of them touch the
consensus constants in ``reliquary.constants`` (those are fixed by the
protocol and must agree network-wide). Everything is overridable from the
environment so the launch scripts stay declarative.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _parse_answer_weights(spec: str) -> dict:
    """Parse ``"small_int:0.15,symbolic:1.0"`` → ``{tier_int: weight}``.

    Names map to answer-shape tiers used by OpenMathFrontier._answer_tier:
    small_int=0, big_int=1, rational/decimal/fraction=2, symbolic/latex/other=3.
    Bare tier ints (``"0:0.15"``) are also accepted.
    """
    name_to_tier = {
        "small_int": 0, "big_int": 1,
        "rational": 2, "decimal": 2, "fraction": 2,
        "symbolic": 3, "latex": 3, "other": 3,
    }
    out: dict = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, _, val = part.partition(":")
        key = key.strip().lower()
        try:
            w = float(val)
        except ValueError:
            continue
        if key in name_to_tier:
            out[name_to_tier[key]] = w
        elif key.isdigit():
            out[int(key)] = w
    return out


def _parse_source_weights(spec: str) -> dict:
    """Parse ``"math:1.0,gsm8k:0.03"`` → ``{"math": 1.0, "gsm8k": 0.03}``.

    Unparseable entries are skipped; an empty/garbage spec yields ``{}`` (the
    caller then treats every source as weight 1.0).
    """
    out: dict = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, _, val = part.partition(":")
        try:
            out[key.strip()] = float(val)
        except ValueError:
            continue
    return out


@dataclass
class MinerConfig:
    """Operator-tunable miner configuration."""

    # --- identity / connectivity ---
    wallet_name: str = field(default_factory=lambda: os.environ.get("BT_WALLET_NAME", "default"))
    hotkey: str = field(default_factory=lambda: os.environ.get("BT_HOTKEY", "default"))
    wallet_path: str = field(default_factory=lambda: os.environ.get("BT_WALLET_PATH", ""))
    network: str = field(default_factory=lambda: os.environ.get("BT_NETWORK", "finney"))
    netuid: int = field(default_factory=lambda: _int("NETUID", 81))
    validator_url: str = field(default_factory=lambda: os.environ.get("RELIQUARY_VALIDATOR_URL", ""))
    use_drand: bool = field(default_factory=lambda: _flag("RELIQUARY_USE_DRAND", "1"))

    # --- model / checkpoint ---
    checkpoint: str = field(default_factory=lambda: os.environ.get("RELIQUARY_CHECKPOINT", "Qwen/Qwen3.5-4B"))
    gen_gpu: int = field(default_factory=lambda: _int("RELIQUARY_GEN_GPU", 0))
    proof_gpu: int = field(default_factory=lambda: _int("RELIQUARY_PROOF_GPU", 1))
    gpu_mem_util: float = field(default_factory=lambda: _float("RELIQUARY_VLLM_GPU_MEM_UTIL", 0.55))
    vllm_max_model_len: int = field(default_factory=lambda: _int("RELIQUARY_VLLM_MAX_MODEL_LEN", 12288))
    # Path to the SEPARATE vLLM venv's python (built by mining/setup.sh). When set
    # and present, generation runs in a worker process there (keeping the main
    # venv on the validator-matched torch 2.7.0 / transformers 5.9.0 / flash-attn
    # 2.8.3 stack). Empty/missing → in-process vLLM (single-venv test box only).
    vllm_python: str = field(
        default_factory=lambda: os.environ.get("RELIQUARY_VLLM_PYTHON", ".venv-vllm/bin/python")
    )
    # Generation speed knobs. eager=False enables torch.compile + CUDA graphs
    # (3-5x faster, much higher GPU util; ~1-2 min one-time compile per checkpoint).
    # max_num_seqs caps vLLM concurrent sequences (0 = vLLM default); raise it to
    # keep the GPU saturated with a big screening batch.
    vllm_enforce_eager: bool = field(default_factory=lambda: _flag("RELIQUARY_VLLM_ENFORCE_EAGER", "0"))
    vllm_max_num_seqs: int = field(default_factory=lambda: _int("RELIQUARY_VLLM_MAX_NUM_SEQS", 0))

    # --- pregen pool sizing ---
    # How many *ready, in-zone, fully-proven* prompt groups to keep queued per
    # environment. A window needs at most MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
    # (8) distinct prompts, so a depth a few multiples above 8 lets us fire a
    # full slate the instant a window opens and immediately start refilling.
    pool_target_depth: int = field(default_factory=lambda: _int("RELIQUARY_POOL_TARGET_DEPTH", 24))
    pool_max_depth: int = field(default_factory=lambda: _int("RELIQUARY_POOL_MAX_DEPTH", 48))

    # How many candidate prompts to screen per pregen batch before keeping the
    # in-zone ones. A well-trained policy rejects most prompts as σ≈0, so we
    # over-sample candidates and keep the survivors.
    screen_batch_prompts: int = field(default_factory=lambda: _int("RELIQUARY_SCREEN_BATCH_PROMPTS", 8))
    # Rollouts to GENERATE per candidate before selecting the 8 to submit. A
    # confident model rarely splits 8 raw samples into an in-zone group, but over
    # N≫8 it fails enough times that we can pick ~4 natural passes + 4 natural
    # fails → σ≥0.43. Higher N → more prompts become in-zone, at more gen cost.
    overgen_rollouts: int = field(default_factory=lambda: _int("RELIQUARY_OVERGEN_ROLLOUTS", 24))
    # Max new tokens during generation. The protocol cap is 8192, but code
    # solutions terminate far shorter; capping lower makes screening fast and
    # biases toward short, naturally-terminating solutions (ramblers that ride
    # the cap are BAD_TERMINATION risks we drop anyway). <=0 uses the full cap.
    gen_max_tokens: int = field(default_factory=lambda: _int("RELIQUARY_GEN_MAX_TOKENS", 4096))
    # Probe→commit screening. When >0, generate this many rollouts per candidate
    # FIRST and drop prompts whose probe EOS-fraction is below probe_min_eos_frac
    # (long-CoT prompts whose in-zone group would truncate-reject anyway) BEFORE
    # spending the full overgen_rollouts budget. Survivors are topped up to
    # overgen_rollouts (probe rollouts reused). 0 = single-stage (generate all).
    probe_rollouts: int = field(default_factory=lambda: _int("RELIQUARY_PROBE_ROLLOUTS", 0))
    probe_min_eos_frac: float = field(
        default_factory=lambda: _float("RELIQUARY_PROBE_MIN_EOS_FRAC", 0.5)
    )
    # Content-gated probe (OpenMath): grade the cheap probe and commit the full
    # overgen budget ONLY to prompts whose probe already shows a usable EOS
    # pass+fail mix (the ~4% minable frontier), skipping the ~96% saturated/too-
    # hard prompts at probe cost. ~3x prompts-screened/min. Off → EOS-only probe.
    sigma_probe: bool = field(default_factory=lambda: _flag("RELIQUARY_OMI_SIGMA_PROBE", "1"))
    # Diagnostic: log per-rollout local grade + extracted boxed answer for in-zone
    # groups, to debug a local-vs-validator OUT_OF_ZONE grading gap.
    debug_grade: bool = field(default_factory=lambda: _flag("RELIQUARY_OMI_DEBUG_GRADE", "0"))
    # Fraction of saturated-looking prompts still committed (recovers a borderline
    # split an 8-rollout probe missed; keeps the learner from over-narrowing).
    probe_explore_floor: float = field(
        default_factory=lambda: _float("RELIQUARY_OMI_PROBE_EXPLORE_FLOOR", 0.08)
    )
    # Concurrent local grader calls (the grader server has GRADER_POOL_SIZE=8
    # warm workers, so 8 keeps them busy without overcommit).
    grade_concurrency: int = field(default_factory=lambda: _int("RELIQUARY_GRADE_CONCURRENCY", 8))

    # --- frontier / zone targeting ---
    # Submit only groups whose locally-predicted σ clears SIGMA_MIN by this
    # margin (proxy grade ≠ validator grade, so leave headroom).
    sigma_margin: float = field(default_factory=lambda: _float("RELIQUARY_SIGMA_MARGIN", 0.0))
    # Per-source candidate-sampling weights for OpenMath. The OMI dataset labels
    # every row with ``problem_source`` ∈ {gsm8k, augmented_gsm8k, math,
    # augmented_math}; a well-trained policy saturates gsm8k (σ≈0) while the
    # competition-hard ``math``/``augmented_math`` rows sit at its frontier. The
    # weight multiplies the bucket-weight acceptance probability in
    # ``next_candidates`` (rejection sampling), so a near-zero weight all but
    # skips a saturated source without wasting GPU on it. Empty → uniform (1.0).
    omi_source_weights: dict = field(default_factory=lambda: _parse_source_weights(
        os.environ.get(
            "RELIQUARY_OMI_SOURCE_WEIGHTS",
            # gsm8k/augmented_gsm8k measured ~0% minable (saturated); the frontier
            # lives in math/augmented_math (~4% minable each).
            "math:1.0,augmented_math:1.0,augmented_gsm8k:0.05,gsm8k:0.02",
        )
    ))
    # Per-answer-TYPE candidate-sampling weights (the dominant minability signal,
    # validated on the network's won-prompt distribution): small integers have one
    # canonical form (saturated, win 0.38x their dataset share) so downsample them;
    # fraction/decimal/symbolic answers have many surface forms (win 2-5x) so keep.
    # tiers: small_int=0, big_int=1, rational(decimal/fraction)=2, symbolic/other=3.
    omi_answer_weights: dict = field(default_factory=lambda: _parse_answer_weights(
        os.environ.get(
            "RELIQUARY_OMI_ANSWER_WEIGHTS",
            "small_int:0.15,big_int:0.6,rational:1.0,symbolic:1.0",
        )
    ))
    # Persistent frontier cache (the "pregen" of selection): remembers, across
    # checkpoint reloads + restarts, which difficulty buckets and which concrete
    # prompt idxs yielded in-zone (σ≥SIGMA_MIN) groups, so the miner re-tries the
    # rare minable frontier FIRST instead of cold-starting after every reload.
    frontier_cache_path: str = field(
        default_factory=lambda: os.environ.get(
            "RELIQUARY_OMI_FRONTIER_CACHE", "mining/state/openmath_frontier.json.gz"
        )
    )
    # Fraction of candidates drawn directly from the cached hot-prompt set (the
    # rest explore via source/bucket-weighted random sampling). 0 disables.
    hot_prompt_fraction: float = field(
        default_factory=lambda: _float("RELIQUARY_OMI_HOT_PROMPT_FRACTION", 0.34)
    )
    # Cap on cached hot prompts (LRU by last-seen σ) and warm-start decay applied
    # to carried-over bucket σ stats on a checkpoint change (model drifts slowly).
    hot_prompt_cap: int = field(default_factory=lambda: _int("RELIQUARY_OMI_HOT_PROMPT_CAP", 6000))
    frontier_warm_decay: float = field(
        default_factory=lambda: _float("RELIQUARY_OMI_FRONTIER_WARM_DECAY", 0.5)
    )
    # Prefer prompts the proxy scores in this *count-correct* band (out of 8).
    # k in [target_k_lo, target_k_hi] maximises distance from the σ floor.
    target_k_lo: int = field(default_factory=lambda: _int("RELIQUARY_TARGET_K_LO", 3))
    target_k_hi: int = field(default_factory=lambda: _int("RELIQUARY_TARGET_K_HI", 5))

    # --- fire / race ---
    # Safety margin (seconds) to stay clear of a drand round boundary so an
    # in-flight POST does not cross into the next 3 s bucket → STALE/FUTURE.
    drand_boundary_margin_s: float = field(
        default_factory=lambda: _float("RELIQUARY_DRAND_BOUNDARY_MARGIN_S", 0.4)
    )
    # Max distinct prompts to fire per window (the protocol per-hotkey cap is 8;
    # never set this above it or the surplus is RATE_LIMITED).
    max_fire_per_window: int = field(
        default_factory=lambda: _int("RELIQUARY_MAX_FIRE_PER_WINDOW", 8)
    )
    # Only fire within this many seconds of a window OPENING, so the submission
    # lands in the earliest drand round (the batch fills round-by-round and seals
    # at B distinct prompts; a later-round fire is BATCH_FILLED). A group that
    # becomes ready mid-window waits in the pool for the next window's open.
    # ~1 drand round (3s) keeps us in the first round without over-tightening.
    fire_window_s: float = field(default_factory=lambda: _float("RELIQUARY_FIRE_WINDOW_S", 3.0))
    # Min T=1.0 EOS probability for a naturally-terminated rollout. The validator
    # computes p_stop at T=1.0 and rejects BAD_TERMINATION below
    # MIN_EOS_PROBABILITY=0.01; we sampled at T=0.9, so re-check with a margin and
    # drop groups that would risk that reject. 0 disables the gate.
    min_eos_prob: float = field(default_factory=lambda: _float("RELIQUARY_MIN_EOS_PROB", 0.0))
    # p_stop floor for SELECTION: pick the 8 rollouts whose exact HF EOS prob
    # clears this (just above the validator's MIN_EOS_PROBABILITY=0.01), so the
    # group avoids BAD_TERMINATION without dropping winnable prompts.
    min_pstop: float = field(default_factory=lambda: _float("RELIQUARY_MIN_PSTOP", 0.01))

    # --- frontier oracle (opencode) ---
    oracle_path: str = field(
        default_factory=lambda: os.environ.get(
            "RELIQUARY_OCI_ORACLE_PATH", "mining/state/opencode_oracle.json.gz"
        )
    )
    allow_unsandboxed_local_grader: bool = field(
        default_factory=lambda: _flag("RELIQUARY_ALLOW_UNSANDBOXED_GRADER", "0")
    )

    # --- polling ---
    state_poll_interval_s: float = field(
        default_factory=lambda: _float("RELIQUARY_STATE_POLL_INTERVAL_S", 0.25)
    )
    verdict_poll_interval_s: float = field(
        default_factory=lambda: _float("RELIQUARY_VERDICT_POLL_INTERVAL_S", 5.0)
    )

    def validate(self) -> None:
        if not (0 <= self.target_k_lo <= self.target_k_hi <= 8):
            raise ValueError("target_k band must satisfy 0 <= lo <= hi <= 8")
        if self.max_fire_per_window > 8:
            raise ValueError(
                "max_fire_per_window must be <= MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW (8)"
            )
