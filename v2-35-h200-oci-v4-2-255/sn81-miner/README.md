# SN81 (Reliquary) Qwen3.5 miner — consolidated workspace

Everything for preparing data and running the curation miner. The reliquary repo
itself stays at `/root/reliquary` (editable install); this dir holds the standalone
launcher, data-prep tools, diagnostics, and runtime data.

## Layout
```
bin/
  run_miner.sh     # launch the miner (Qwen3.5 numeric pool, k=2 curation). Reads data/ below.
  setup.sh         # fresh-box environment installer (torch cu130 + vllm + ninja + ...)
data/
  inzone_pool_qwen35.json  # ACTIVE pool: 150,970 plain-numeric prompts (gsm8k/aug_gsm8k)
  hot_pool.json            # runtime cache of curated prompts (self-built)
  submitted_idx.json       # persistent anti-hash_duplicate blocklist
  reference/               # legacy/harvested data (old non-numeric pool, top-miner captures, etc.)
dataprep/
  build_qwen35_pool.py     # build the numeric pool (pyarrow). THE current pool builder.
  format_analysis.py       # old non-numeric pool builder (for the pre-Qwen3.5 checkpoint)
  harvest_inzone.py        # harvest top-miner accepts/vectors from reliqua.ai
  verify_template.py, brittle_pool.py, analysis.py, ...  # analysis tools
diagnostics/
  qwen35_dist.py / qwen35_peek.py / qwen35_src.py  # measure correctness/box-rate/format on a checkpoint
  mon_breadth.sh           # throughput monitor
logs/
  miner.log                # current run log
backups/                   # env tarball + old hot-pool snapshots
miner.pid                  # current miner PID
```

## Run
```
bash /root/sn81-miner/bin/run_miner.sh      # kills any running miner, relaunches
tail -f /root/sn81-miner/logs/miner.log
```

## Qwen3.5 notes (why this config)
- Live checkpoint = Qwen3.5 (R0mAI/reliquary-sn-v23, multimodal/linear-attn). Loaded text-only
  via enforce_eager + limit_mm + ninja (NO hf_overrides — those mis-load to garbage).
- The reward needs `\boxed{}` for non-numeric answers, which Qwen3.5 never emits; its plain
  prose answers only extract when the answer is a bare number → pool is **numeric-only**.
- Curation target inverted to **k=2** (correct is the scarce side now; wrong is abundant).
- Prompt is locked to the validator's canonical (chat-template, enable_thinking) — do NOT customize.

## Rebuild the pool
```
cd /root/reliquary && .venv/bin/python /root/sn81-miner/dataprep/build_qwen35_pool.py
# writes /root/inzone_pool_qwen35.json — copy to data/ and relaunch
```
