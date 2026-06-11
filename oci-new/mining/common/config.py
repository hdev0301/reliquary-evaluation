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
    # Concurrent local grader calls (the grader server has GRADER_POOL_SIZE=8
    # warm workers, so 8 keeps them busy without overcommit).
    grade_concurrency: int = field(default_factory=lambda: _int("RELIQUARY_GRADE_CONCURRENCY", 8))

    # --- frontier / zone targeting ---
    # Submit only groups whose locally-predicted σ clears SIGMA_MIN by this
    # margin (proxy grade ≠ validator grade, so leave headroom).
    sigma_margin: float = field(default_factory=lambda: _float("RELIQUARY_SIGMA_MARGIN", 0.0))
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
