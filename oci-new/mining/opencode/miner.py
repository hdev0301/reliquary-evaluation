"""OpenCode miner orchestrator — vLLM + pregen pool + frontier oracle + fast fire.

Async task layout (single event loop; GPU work offloaded to a worker thread so
the fire path stays sub-millisecond):

  * **state**   — poll ``/state`` every ~0.25 s; track randomness/window/
    cooldown; on a ``checkpoint_n`` bump, flush the pool and request a model
    reload.
  * **pregen**  — keep the ready pool full: reload models when asked, else run
    one screen→prove cycle (``PregenPool.produce_once``) in a thread.
  * **fire**    — the instant a window is OPEN with randomness, fire up to the
    per-hotkey cap of distinct in-zone groups, boundary-safe, in the earliest
    drand round; top up as the pool refills.
  * **verdict** — poll ``/verdicts`` and feed real outcomes back to the
    frontier model.

Only ``fire`` is latency-critical, and it is pure CPU (cached buckets · r_vec +
ed25519 sign), so it never contends with the GPU producer thread.
"""

from __future__ import annotations

import asyncio
import logging

from reliquary.constants import (
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
)

from mining.common.config import MinerConfig
from mining.common.fire import fire_group, wait_for_safe_drand_window
from mining.common.grail_proof import ProofBuilder
from mining.common.pregen import PregenPool
from mining.common.state import StateView, fetch_verdicts, refresh_state
from mining.common.vllm_generator import make_generator
from mining.opencode.frontier import OpenCodeFrontier
from mining.opencode.oracle import LocalOracle

logger = logging.getLogger(__name__)

ENV_NAME = "opencodeinstruct"


def _setup_mining_logging() -> None:
    """Configure the ``mining`` logger — call AFTER importing bittensor.

    bittensor, on import, walks every existing logger and clears its handlers +
    raises its level to CRITICAL (its logging takeover). So the mining logger
    must be (re)configured afterwards. ``propagate=False`` + our own handlers
    make mining output fully independent of bittensor's root config.
    """
    import os as _os
    import sys as _sys

    # Silence the per-poll httpx/urllib3 request spam (set after bittensor, which
    # configures these loggers on import/Wallet init).
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    mlog = logging.getLogger("mining")
    mlog.setLevel(logging.INFO)
    mlog.propagate = False
    if any(getattr(h, "_reliquary_mining", False) for h in mlog.handlers):
        return
    fmt = logging.Formatter(
        "%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for h in (
        logging.FileHandler(_os.environ.get("RELIQUARY_PREGEN_LOG", "/root/opencode_pregen.log")),
        logging.StreamHandler(_sys.stdout),
    ):
        h.setFormatter(fmt)
        h._reliquary_mining = True
        mlog.addHandler(h)


class OpenCodeMiner:
    def __init__(self, config: MinerConfig) -> None:
        config.validate()
        self.config = config
        self.view = StateView()
        self._wallet = None
        self._url: str | None = None
        self._client = None
        self._tokenizer = None
        self._hf_model = None
        self._generator: VLLMGenerator | None = None
        self._proof_builder: ProofBuilder | None = None
        self._pool: PregenPool | None = None
        self._frontier: OpenCodeFrontier | None = None
        self._env = None
        self._oracle: LocalOracle | None = None
        self._local_n = 0
        self._local_path: str | None = None
        self._reload_requested = False
        self._fired_in_window: dict[int, set[int]] = {}
        self._last_verdict_ts = 0.0

    # ==================================================================
    # Boot
    # ==================================================================
    async def setup(self) -> None:
        import os

        import bittensor as bt
        import httpx
        import torch

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
        from reliquary.infrastructure.chain import NETUID, get_metagraph, get_subtensor
        from reliquary.miner.engine import _hf_download
        from reliquary.miner.submitter import discover_validator_url, get_window_state_v2
        from reliquary.shared.modeling import (
            load_text_generation_model, load_tokenizer, resolve_eos_token_ids,
        )

        os.environ.setdefault("RELIQUARY_OCI_PROMPT_ONLY", "1")

        cfg = self.config
        wallet_kwargs = {"name": cfg.wallet_name, "hotkey": cfg.hotkey}
        if cfg.wallet_path:
            wallet_kwargs["path"] = cfg.wallet_path
        self._wallet = bt.Wallet(**wallet_kwargs)

        # Configure mining logging AFTER the last bittensor call — bittensor
        # clears every logger's handlers + raises levels both at import AND on
        # Wallet/subtensor init, so an earlier call gets wiped.
        _setup_mining_logging()

        self._client = httpx.AsyncClient(timeout=30)

        # Resolve validator URL.
        if cfg.validator_url:
            self._url = cfg.validator_url
        else:
            subtensor = await get_subtensor()
            metagraph = await get_metagraph(subtensor, NETUID)
            self._url = discover_validator_url(metagraph)
        logger.info("validator: %s", self._url)

        # Seed checkpoint from the validator (lands directly on the live model).
        state = await get_window_state_v2(self._url, client=self._client)
        if state.checkpoint_repo_id and state.checkpoint_revision:
            self._local_path = await _hf_download(
                state.checkpoint_repo_id, state.checkpoint_revision
            )
            self._local_n = state.checkpoint_n
            logger.info("seeded checkpoint %d from %s", state.checkpoint_n, self._local_path)
        else:
            self._local_path = cfg.checkpoint
            logger.info("no published checkpoint; using %s", cfg.checkpoint)

        self._tokenizer = load_tokenizer(self._local_path)

        # Validator-matched HF proof model first (small, ~8GB on H200), resolve
        # the canonical EOS set from it, THEN start the vLLM worker (its own venv/
        # process) which reserves the bulk of the GPU. On the single H200 the
        # proof model's few GB leave ample headroom for vLLM's KV cache.
        self._hf_model = load_text_generation_model(
            self._local_path, torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to(f"cuda:{cfg.proof_gpu}").eval()
        eos_ids = sorted(set(resolve_eos_token_ids(self._hf_model, self._tokenizer)))
        self._proof_builder = ProofBuilder(self._hf_model, proof_gpu=cfg.proof_gpu)

        self._generator = make_generator(cfg, eos_ids, repo_dir=os.getcwd())
        self._stage_vllm_processor_files(self._local_path)
        self._generator.load(self._local_path)

        # Env + oracle + frontier + pool.
        self._env = OpenCodeInstructEnvironment()
        self._oracle = LocalOracle(cfg.oracle_path).load()
        self._oracle.ensure_grader(allow_unsandboxed=cfg.allow_unsandboxed_local_grader)
        self._frontier = OpenCodeFrontier(
            self._env, self._tokenizer, self._oracle, cfg,
            cooldown_getter=lambda: self.view.cooldown_per_env.get(ENV_NAME, set()),
        )
        self._pool = PregenPool(self._generator, self._proof_builder, cfg)
        self._pool.set_checkpoint(self._local_n)
        logger.info("OpenCode miner ready (env_len=%d, oracle=%d).", len(self._env), len(self._oracle))

    def _stage_vllm_processor_files(self, model_path: str) -> None:
        """Copy the static processor configs the vLLM loader needs.

        Qwen3.5-4B is a ``Qwen3_5ForConditionalGeneration`` (VL) arch used
        text-only; vLLM detects the arch and demands ``preprocessor_config.json``
        / ``video_preprocessor_config.json``, which the published text-only
        checkpoint omits. They are static arch metadata, so copy them from the
        base model into the checkpoint dir. The HF proof model never needs them.
        """
        import os as _os
        import shutil

        if not _os.path.isdir(model_path):
            return  # a bare repo id (base model) already ships these files
        from huggingface_hub import hf_hub_download

        base = self.config.checkpoint
        for fn in ("preprocessor_config.json", "video_preprocessor_config.json"):
            dst = _os.path.join(model_path, fn)
            if _os.path.exists(dst):
                continue
            try:
                src = hf_hub_download(repo_id=base, filename=fn)
                shutil.copy(src, dst)
                logger.info("staged %s from base %s for vLLM", fn, base)
            except Exception as e:
                logger.warning("could not stage %s for vLLM: %s", fn, e)

    # ==================================================================
    # Tasks
    # ==================================================================
    async def _state_task(self) -> None:
        while True:
            try:
                view = await refresh_state(self._url, self._client, [ENV_NAME])
            except Exception:
                await asyncio.sleep(self.config.state_poll_interval_s)
                continue
            self.view = view
            # Cooldown → drop now-ineligible ready groups.
            self._pool.drop_cooled(ENV_NAME, view.cooldown_per_env.get(ENV_NAME, set()))
            # Checkpoint change → flush + request reload.
            if view.checkpoint_n > self._local_n and view.checkpoint_repo_id and view.checkpoint_revision:
                logger.info("checkpoint %d -> %d detected", self._local_n, view.checkpoint_n)
                self._pool.set_checkpoint(view.checkpoint_n)
                self._frontier.reset_for_checkpoint()
                self._reload_requested = True
            await asyncio.sleep(self.config.state_poll_interval_s)

    async def _pregen_task(self) -> None:
        import traceback as _tb

        print("@@PREGEN task started", flush=True)
        while True:
            if self._reload_requested:
                await self._reload_models()
                continue
            try:
                if self._pool.depth(ENV_NAME) >= self.config.pool_target_depth:
                    await asyncio.sleep(0.2)
                    continue
                added = await asyncio.to_thread(
                    self._pool.produce_once, ENV_NAME, self._frontier
                )
                if added == 0:
                    await asyncio.sleep(0.05)
            except Exception as e:
                print(f"@@PREGEN ERROR: {e!r}\n{_tb.format_exc()}", flush=True)
                logger.exception("pregen cycle failed")
                await asyncio.sleep(1.0)

    async def _reload_models(self) -> None:
        import torch

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.miner.engine import _hf_download
        from reliquary.shared.modeling import load_text_generation_model

        cfg = self.config
        v = self.view
        try:
            path = await _hf_download(v.checkpoint_repo_id, v.checkpoint_revision)
            # Free the old proof model and rebuild vLLM FIRST (same single-GPU
            # memory-profiling reason as setup), then load the new proof copy.
            self._hf_model = None
            self._proof_builder = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._stage_vllm_processor_files(path)
            await asyncio.to_thread(self._generator.load, path)
            new_hf = (
                load_text_generation_model(
                    path, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION
                ).to(f"cuda:{cfg.proof_gpu}").eval()
            )
            self._hf_model = new_hf
            self._proof_builder = ProofBuilder(new_hf, proof_gpu=cfg.proof_gpu)
            self._pool.proof_builder = self._proof_builder
            self._local_n = v.checkpoint_n
            self._local_path = path
            self._reload_requested = False
            logger.info("models reloaded for checkpoint %d", v.checkpoint_n)
        except Exception:
            logger.exception("model reload failed; will retry")
            await asyncio.sleep(2.0)

    async def _fire_task(self) -> None:
        from reliquary.protocol.submission import RejectReason

        cap = min(self.config.max_fire_per_window, MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW)
        stop_reasons = {
            RejectReason.BATCH_FILLED, RejectReason.RATE_LIMITED,
            RejectReason.WINDOW_NOT_ACTIVE, RejectReason.PROMPT_FULL,
        }
        while True:
            v = self.view
            if not (v.is_open and v.randomness and v.checkpoint_n == self._local_n
                    and self._proof_builder is not None):
                await asyncio.sleep(self.config.state_poll_interval_s)
                continue
            window_n = v.window_n
            fired = self._fired_in_window.setdefault(window_n, set())
            if len(fired) >= cap:
                await asyncio.sleep(self.config.state_poll_interval_s)
                continue

            cooldown = v.cooldown_per_env.get(ENV_NAME, set())
            group = self._pool.pop_best(
                ENV_NAME, checkpoint_n=self._local_n, exclude=fired | cooldown
            )
            if group is None:
                await asyncio.sleep(0.02)  # pool may refill within the window
                continue

            await wait_for_safe_drand_window(self.config.drand_boundary_margin_s)
            resp = await fire_group(
                url=self._url, client=self._client, wallet=self._wallet,
                proof_builder=self._proof_builder, group=group,
                randomness=v.randomness, window_n=window_n,
                checkpoint_hash=v.checkpoint_revision or "",
            )
            fired.add(group.prompt_idx)
            # Keep the window map small.
            for w in list(self._fired_in_window):
                if w < window_n - 4:
                    self._fired_in_window.pop(w, None)
            if resp is not None and not resp.accepted and resp.reason in stop_reasons:
                logger.info("window %d closed for us (%s)", window_n,
                            getattr(resp.reason, "value", resp.reason))
                # Mark window saturated so we stop hammering it.
                fired.update(range(-cap, 0))

    async def _verdict_task(self) -> None:
        hotkey = self._wallet.hotkey.ss58_address
        while True:
            verdicts = await fetch_verdicts(self._url, self._client, hotkey, self._last_verdict_ts)
            for vd in verdicts:
                self._last_verdict_ts = max(self._last_verdict_ts, vd.get("ts", 0.0))
                self._frontier.record_verdict(
                    bool(vd.get("accepted")), str(vd.get("reason", ""))
                )
                # print() (not logger) — the mining logger gets suppressed by
                # bittensor; prints reliably reach the miner log.
                tag = "ACCEPTED ✅" if vd.get("accepted") else f"REJECTED {vd.get('reason')}"
                print(f"@@VERDICT win={vd.get('window_n')} {tag}", flush=True)
            await asyncio.sleep(self.config.verdict_poll_interval_s)

    # ==================================================================
    async def run(self) -> None:
        await self.setup()
        _setup_mining_logging()  # re-assert in case any late bittensor call wiped it
        await asyncio.gather(
            self._state_task(),
            self._pregen_task(),
            self._fire_task(),
            self._verdict_task(),
        )


async def _amain() -> None:
    import logging as _logging

    import os as _os

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    # The "mining" logger gets its own FileHandler so its output is captured
    # regardless of stdout buffering or whatever bittensor does to the root
    # logger. propagate stays on so it ALSO reaches the main log when possible.
    _mlog = _logging.getLogger("mining")
    _mlog.setLevel(_logging.INFO)
    _pregen_log = _os.environ.get("RELIQUARY_PREGEN_LOG", "/root/opencode_pregen.log")
    if not any(isinstance(h, _logging.FileHandler) for h in _mlog.handlers):
        _fh = _logging.FileHandler(_pregen_log)
        _fh.setFormatter(_logging.Formatter(
            "%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        _mlog.addHandler(_fh)
    miner = OpenCodeMiner(MinerConfig())
    try:
        await miner.run()
    except KeyboardInterrupt:
        logger.info("interrupted")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
