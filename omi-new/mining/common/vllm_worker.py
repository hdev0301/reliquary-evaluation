"""Standalone vLLM generation worker (runs in its OWN venv).

Why a separate process: the validator-compatible proof stack pins
``transformers==5.9.0`` + ``torch==2.7.0`` (see the Dockerfile), but a vLLM new
enough to accept transformers 5.9.0 requires torch 2.11. They cannot share one
venv. They don't need to: the validator never runs vLLM, and only the *token
ids* cross from generation into the GRAIL proof — which the validator-matched HF
model recomputes itself. So vLLM lives here, in ``.venv-vllm`` (its own torch),
and speaks a tiny JSON-lines protocol over stdin/stdout to the miner process.

This module imports **only vllm + stdlib** — never ``reliquary`` or torch from
the main venv — so it loads cleanly under the vLLM environment. The protocol
sampling constants are duplicated here (they are protocol-fixed and must mirror
``reliquary.constants``); they are not strategy knobs.

Protocol (one JSON object per line on stdin; each reply is one line on stdout
prefixed with ``RESP\\t`` so vLLM's own stdout chatter is unambiguous):

    {"cmd":"ping"}                                  -> RESP {"ok":true}
    {"cmd":"load","model":path,"gpu":0,             -> RESP {"ok":true}
     "gpu_mem":0.6,"max_model_len":12288,"eos":[...]}
    {"cmd":"generate","prompts":[[ids]...],"n":8}   -> RESP {"ok":true,"groups":[[
                                                          {"t":[ids],"eos":bool,"stop":int|null}]]}
"""

from __future__ import annotations

import json
import os
import sys

# Steer vLLM to Triton-compiled kernels (triton is installed) instead of
# FlashInfer's nvcc-JIT path, which fails on boxes without full CUDA dev headers
# (curand.h) or with a torch CUDA version newer than the system toolkit. This
# only affects HOW tokens are sampled, never the token ids, so the GRAIL proof
# (rebuilt by the validator-matched HF model) is unaffected. Must be set before
# vllm is imported.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# Protocol-fixed sampling params — MUST mirror reliquary.constants
# (T_PROTO / TOP_P_PROTO / TOP_K_PROTO / MAX_NEW_TOKENS_PROTOCOL_CAP / M_ROLLOUTS).
T_PROTO = 0.9
TOP_P_PROTO = 1.0
TOP_K_VLLM = -1          # vLLM disables top-k with -1 (protocol TOP_K_PROTO=0 == off)
MAX_NEW_TOKENS = 8192
M_ROLLOUTS = 8


def _respond(obj: dict) -> None:
    sys.stdout.write("RESP\t" + json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    from vllm import LLM, SamplingParams, TokensPrompt

    llm = None
    eos: list[int] = []

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "ping":
                _respond({"ok": True, "loaded": llm is not None})

            elif cmd == "load":
                eos = sorted(set(int(x) for x in req.get("eos", [])))
                os.environ["CUDA_VISIBLE_DEVICES"] = str(req.get("gpu", 0))
                # Drop the old engine before building the new one (checkpoint swap).
                llm = None
                kwargs = dict(
                    model=req["model"],
                    dtype=req.get("dtype", "bfloat16"),
                    gpu_memory_utilization=float(req.get("gpu_mem", 0.6)),
                    max_model_len=int(req.get("max_model_len", 12288)),
                    enforce_eager=bool(req.get("enforce_eager", False)),
                    disable_log_stats=True,
                    # Qwen3.5 is a VL arch used TEXT-ONLY here: tell vLLM no image/
                    # video inputs so it skips the vision-encoder profiling + cache
                    # (the slow/hang path) and runs the LM far faster.
                    limit_mm_per_prompt={"image": 0, "video": 0},
                )
                mns = int(req.get("max_num_seqs", 0))
                if mns > 0:
                    kwargs["max_num_seqs"] = mns
                # Qwen3.5 GatedDeltaNet: use the triton prefill backend so vLLM
                # doesn't nvcc-JIT the FlashInfer GDN kernel.
                gdn = req.get("gdn_prefill_backend", "triton")
                try:
                    llm = LLM(gdn_prefill_backend=gdn, **kwargs)
                except TypeError:
                    # Older/newer vLLM without that arg — fall back without it.
                    llm = LLM(**kwargs)
                _respond({"ok": True})

            elif cmd == "generate":
                if llm is None:
                    _respond({"ok": False, "error": "model not loaded"})
                    continue
                n = int(req.get("n", M_ROLLOUTS))
                mx = int(req.get("max_tokens", 0)) or MAX_NEW_TOKENS
                mx = min(mx, MAX_NEW_TOKENS)  # never exceed the protocol cap
                sp = SamplingParams(
                    n=n, temperature=T_PROTO, top_p=TOP_P_PROTO, top_k=TOP_K_VLLM,
                    max_tokens=mx, stop_token_ids=eos, logprobs=None,
                )
                prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in req["prompts"]]
                outs = llm.generate(prompts, sp, use_tqdm=False)
                groups = []
                for ro in outs:
                    g = []
                    for comp in ro.outputs:
                        stop = comp.stop_reason
                        g.append({
                            "t": list(comp.token_ids),
                            "eos": comp.finish_reason == "stop",
                            "stop": stop if isinstance(stop, int) else None,
                        })
                    groups.append(g)
                _respond({"ok": True, "groups": groups})

            else:
                _respond({"ok": False, "error": f"unknown cmd: {cmd}"})
        except Exception as e:  # never die on a single bad request
            _respond({"ok": False, "error": repr(e)})


if __name__ == "__main__":
    main()
