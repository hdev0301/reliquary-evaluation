"""vLLM-backed rollout generation at the protocol-fixed sampling params.

Two interchangeable backends with the same surface (``load`` /
``generate_groups`` / ``generate_group``):

  * ``VLLMWorkerClient`` (default, validator-compatible) — drives a vLLM worker
    in a SEPARATE venv (``mining.common.vllm_worker``). The main miner process
    keeps the validator-matched proof stack (torch 2.7.0 / transformers 5.9.0 /
    flash-attn 2.8.3 / flash-linear-attention 0.5.0); vLLM keeps its own newer
    torch. Only token ids cross the boundary, and the GRAIL proof is rebuilt by
    the validator-matched HF model, so generation's torch never touches
    consensus.
  * ``VLLMGenerator`` (in-process) — for a single-venv test box where vLLM and
    the proof model happen to be import-compatible.

``make_generator(config, eos)`` picks the backend: worker mode when a vLLM venv
python is configured and present, else in-process.

In both backends vLLM is used ONLY for sampling. Prompt tokens come from the
shared ``encode_prompt`` (fed as ``prompt_token_ids``) so we never re-template
(→ ``PROMPT_MISMATCH``); each completion is truncated at (and includes) the
first EOS; a cap-without-EOS rollout is flagged for the truncation guard.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass

from reliquary.constants import (
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
)

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    tokens: list[int]           # prompt tokens + completion (incl. first EOS)
    prompt_length: int
    finished_with_eos: bool


def _assemble_rollout(
    prompt_tokens: list[int], gen_ids: list[int], finished_with_eos: bool,
    stop_id, eos_token_ids: list[int],
) -> Rollout:
    """Build a Rollout ending in exactly ONE terminating EOS.

    Depending on the vLLM version/config, the stop token may already be present
    at the end of ``gen_ids`` OR excluded. Appending unconditionally produced a
    DOUBLE EOS ([..., im_end, im_end]) → HF gives ~0 prob to "im_end after
    im_end" (p_stop≈0) and the validator flags the repeated stop tail as
    BAD_TERMINATION. So only append when the sequence does not already end in an
    EOS, and never produce two.
    """
    gen = list(gen_ids)
    eos_set = set(eos_token_ids)
    if isinstance(stop_id, int):
        eos_set.add(stop_id)
    # Trim any trailing EOS padding down to none, then add exactly one if the
    # rollout terminated naturally.
    while len(gen) > 1 and gen[-1] in eos_set and gen[-2] in eos_set:
        gen.pop()
    if finished_with_eos and not (gen and gen[-1] in eos_set):
        gen.append(stop_id if isinstance(stop_id, int) else eos_token_ids[0])
    return Rollout(
        tokens=list(prompt_tokens) + gen,
        prompt_length=len(prompt_tokens),
        finished_with_eos=finished_with_eos,
    )


# ======================================================================
# Subprocess backend (default, validator-compatible)
# ======================================================================
class VLLMWorkerClient:
    """Drives ``mining.common.vllm_worker`` in a separate (vLLM) venv."""

    def __init__(
        self, model_path: str, *, vllm_python: str, repo_dir: str,
        eos_token_ids: list[int], gpu: int = 0,
        gpu_memory_utilization: float = 0.55, max_model_len: int = 12288,
        enforce_eager: bool = False, max_num_seqs: int = 0,
    ) -> None:
        self.model_path = model_path
        self.vllm_python = vllm_python
        self.repo_dir = repo_dir
        self.eos_token_ids = sorted(set(eos_token_ids))
        self.gpu = gpu
        self._gpu_mem = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._enforce_eager = enforce_eager
        self._max_num_seqs = max_num_seqs
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _ensure_proc(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        env = dict(os.environ)
        env["PYTHONPATH"] = self.repo_dir + os.pathsep + env.get("PYTHONPATH", "")
        logger.info("spawning vLLM worker: %s -m mining.common.vllm_worker", self.vllm_python)
        self._proc = subprocess.Popen(
            [self.vllm_python, "-m", "mining.common.vllm_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None,  # vLLM logs to our stderr (visible in the miner log)
            text=True, bufsize=1, cwd=self.repo_dir, env=env,
        )

    def _rpc(self, obj: dict) -> dict:
        with self._lock:
            self._ensure_proc()
            assert self._proc and self._proc.stdin and self._proc.stdout
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if line == "":
                    raise RuntimeError("vLLM worker exited unexpectedly")
                if line.startswith("RESP\t"):
                    return json.loads(line[5:])
                # else: vLLM's own stdout noise — ignore.

    def load(self, model_path: str | None = None) -> None:
        if model_path is not None:
            self.model_path = model_path
        r = self._rpc({
            "cmd": "load", "model": self.model_path, "gpu": self.gpu,
            "gpu_mem": self._gpu_mem, "max_model_len": self._max_model_len,
            "eos": self.eos_token_ids, "dtype": "bfloat16",
            "enforce_eager": self._enforce_eager, "max_num_seqs": self._max_num_seqs,
        })
        if not r.get("ok"):
            raise RuntimeError(f"vLLM worker load failed: {r.get('error')}")
        logger.info("vLLM worker ready (model=%s, gpu=%d)", self.model_path, self.gpu)

    def generate_group(self, prompt_tokens: list[int], n: int = M_ROLLOUTS, max_tokens: int = 0) -> list[Rollout]:
        return self.generate_groups([prompt_tokens], n=n, max_tokens=max_tokens)[0]

    def generate_groups(self, prompts_tokens, n: int = M_ROLLOUTS, max_tokens: int = 0) -> list[list[Rollout]]:
        r = self._rpc({
            "cmd": "generate", "prompts": [list(p) for p in prompts_tokens],
            "n": n, "max_tokens": max_tokens,
        })
        if not r.get("ok"):
            raise RuntimeError(f"vLLM worker generate failed: {r.get('error')}")
        groups: list[list[Rollout]] = []
        for prompt_tokens, g in zip(prompts_tokens, r["groups"]):
            groups.append([
                _assemble_rollout(prompt_tokens, c["t"], c["eos"], c.get("stop"), self.eos_token_ids)
                for c in g
            ])
        return groups


# ======================================================================
# In-process backend (single-venv test box)
# ======================================================================
class VLLMGenerator:
    """In-process ``vllm.LLM`` pinned to the protocol sampling params."""

    def __init__(
        self, model_path: str, *, eos_token_ids: list[int], gpu: int = 0,
        gpu_memory_utilization: float = 0.55, max_model_len: int = 12288,
        dtype: str = "bfloat16",
    ) -> None:
        self.model_path = model_path
        self.eos_token_ids = sorted(set(eos_token_ids))
        self.gpu = gpu
        self._gpu_mem = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._llm = None

    def load(self, model_path: str | None = None) -> None:
        from vllm import LLM

        if model_path is not None:
            self.model_path = model_path
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            self._llm = None
            self._llm = LLM(
                model=self.model_path, dtype=self._dtype,
                gpu_memory_utilization=self._gpu_mem, max_model_len=self._max_model_len,
                enforce_eager=False, disable_log_stats=True,
            )
        finally:
            if prev is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev
        logger.info("in-process vLLM ready (model=%s, gpu=%d)", self.model_path, self.gpu)

    def _sampling_params(self, n: int, max_tokens: int = 0):
        from vllm import SamplingParams

        top_k = -1 if TOP_K_PROTO in (0, -1) else TOP_K_PROTO
        mx = min(max_tokens or MAX_NEW_TOKENS_PROTOCOL_CAP, MAX_NEW_TOKENS_PROTOCOL_CAP)
        return SamplingParams(
            n=n, temperature=T_PROTO, top_p=TOP_P_PROTO, top_k=top_k,
            max_tokens=mx, stop_token_ids=self.eos_token_ids,
            logprobs=None,
        )

    def generate_group(self, prompt_tokens: list[int], n: int = M_ROLLOUTS, max_tokens: int = 0) -> list[Rollout]:
        return self.generate_groups([prompt_tokens], n=n, max_tokens=max_tokens)[0]

    def generate_groups(self, prompts_tokens, n: int = M_ROLLOUTS, max_tokens: int = 0) -> list[list[Rollout]]:
        if self._llm is None:
            self.load()
        from vllm import TokensPrompt

        sp = self._sampling_params(n, max_tokens)
        requests = [TokensPrompt(prompt_token_ids=list(pt)) for pt in prompts_tokens]
        outputs = self._llm.generate(requests, sp, use_tqdm=False)
        groups: list[list[Rollout]] = []
        for prompt_tokens, req_out in zip(prompts_tokens, outputs):
            groups.append([
                _assemble_rollout(
                    prompt_tokens, comp.token_ids, comp.finish_reason == "stop",
                    comp.stop_reason, self.eos_token_ids,
                )
                for comp in req_out.outputs
            ])
        return groups


def make_generator(config, eos_token_ids: list[int], repo_dir: str):
    """Pick the worker (validator-compatible) backend, else in-process."""
    vllm_python = getattr(config, "vllm_python", "") or ""
    if vllm_python and os.path.exists(vllm_python):
        logger.info("vLLM backend: separate worker venv (%s)", vllm_python)
        return VLLMWorkerClient(
            config.checkpoint, vllm_python=vllm_python, repo_dir=repo_dir,
            eos_token_ids=eos_token_ids, gpu=config.gen_gpu,
            gpu_memory_utilization=config.gpu_mem_util, max_model_len=config.vllm_max_model_len,
            enforce_eager=getattr(config, "vllm_enforce_eager", False),
            max_num_seqs=getattr(config, "vllm_max_num_seqs", 0),
        )
    logger.warning(
        "vLLM backend: IN-PROCESS (no vllm_python at %r). This only works on a "
        "single-venv box where vLLM and transformers==5.9.0 are import-compatible; "
        "for validator compatibility install the split venvs via mining/setup.sh.",
        vllm_python,
    )
    return VLLMGenerator(
        config.checkpoint, eos_token_ids=eos_token_ids, gpu=config.gen_gpu,
        gpu_memory_utilization=config.gpu_mem_util, max_model_len=config.vllm_max_model_len,
    )
