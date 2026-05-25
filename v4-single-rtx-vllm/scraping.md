# Scraping & dataset preparation

The live miner spends most of its GPU time discovering which prompts are
in-zone for the current checkpoint. Two side-channels short-circuit that
work by populating a shared Supabase cache the miner hydrates from on
every checkpoint advance:

| Tool | Source of data | GPU | Throughput | What it adds |
|---|---|---|---|---|
| [`scripts/scrape_intel.py`](../scripts/scrape_intel.py) | Public R2 archive (`/api/r2/window/N`) | No | ~1 s/window | `(prompt_idx, k, sigma, status='good')` rows |
| [`scripts/prep_dataset.py`](../scripts/prep_dataset.py) | Local model.generate() | Yes (24+ GB VRAM) | ~160 s/batch (8 rollouts × 2 prompts) | Outcomes + full pregen token batches |

Both write into the same two tables defined in
[`sql/supabase_schema.sql`](../sql/supabase_schema.sql):

- **`prompt_outcomes`** — keyed on `(prompt_idx, checkpoint_hash)`. The
  miner hydrates `status='dud'`/`'oof'` rows into `_prescreen_dud_set`
  (skipped entirely by the picker) and `status='good'` rows into
  `_known_good_prompts` (picker still chooses them but bypasses the
  ~22 s prescreen and goes straight to full gen).
- **`pregen_batches`** — keyed on `(prompt_idx, checkpoint_hash)`. Full
  M=8 token rollouts. Miner restarts re-hydrate these into
  `_pregen_queue` so a crash or restart doesn't lose ~16 min of gen
  work.

`checkpoint_hash` is always the validator's published HF revision string
(e.g. `1e922bd484ff34457732f853b94258eb422c2f06`). Cache rows are
ckpt-scoped — when the validator publishes a new revision, prior rows
silently become inert (different key) rather than poisoning the new
policy.

## Prerequisites

In `scripts/.env` (or your shell):

```bash
export RELIQUARY_VALIDATOR_URL=http://86.38.238.30:8080
export RELIQUARY_SUPABASE_URL=https://<your-project>.supabase.co
export RELIQUARY_SUPABASE_KEY=<service_role JWT>
```

Apply the schema once per Supabase project — paste
[`sql/supabase_schema.sql`](../sql/supabase_schema.sql) into the SQL
editor at `https://supabase.com/dashboard/project/<ref>/sql/new` and
click Run. It's idempotent.

## `scripts/scrape_intel.py` — fast, no-GPU, intel only

Reads accepted submissions from other miners and persists the
`(prompt_idx, k, sigma)` triplets. Doesn't store rollout tokens —
re-submitting another miner's tokens would trip the validator's
`HASH_DUPLICATE` gate, so the live miner still generates locally on
these prompts. The value is **skipping the ~22 s prescreen** on prompts
already proven in-zone for this ckpt.

```bash
cd /root/reliquary && source scripts/.env

# Default: last 10 windows
python scripts/scrape_intel.py

# Override the window range (useful when recent prompts are all in
# cooldown — see "Cooldown gotcha" below)
python scripts/scrape_intel.py --since-window 6650 --until-window 6810

# Keep cache fresh as new windows close
python scripts/scrape_intel.py --watch 60

# Also persist OOZ/reward_distribution rejects as 'oof' duds
python scripts/scrape_intel.py --lookback 50 --include-rejected

# Override the ckpt_hash (skips validator query, useful if validator down)
python scripts/scrape_intel.py --ckpt-hash 1e922bd484ff34457732f853b94258eb422c2f06
```

**All flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--lookback N` | 10 | Number of past windows to scrape (ignored if `--since-window` set) |
| `--since-window N` | — | Explicit start window |
| `--until-window N` | validator current - 1 | Explicit end window |
| `--ckpt-hash HASH` | from validator `/state` | Override the cache key |
| `--validator-url URL` | `$RELIQUARY_VALIDATOR_URL` | Where to fetch current state |
| `--hotkey HK` | — | ss58 to stamp on rows (informational only) |
| `--watch SEC` | 0 | After one pass, sleep SEC and re-scrape forever |
| `--include-rejected` | off | Also persist OOZ / reward_distribution rejects as 'oof' |

### Cooldown gotcha

The validator publishes `cooldown_prompts` in `/state` — a list of
~4000-8000 prompt_idx values that are temporarily off-limits because
they were used recently. Scraping the last 10 windows produces intel
that is **100% in cooldown** (these prompts WERE the recent accepts —
they ARE the cooldown list).

To get useful intel, scrape a wider historical range. Cooldown expiry
varies but is typically 50-150 windows:

```bash
python scripts/scrape_intel.py --since-window 6650 --until-window 6810
# ~160 windows × 8 accepts ≈ 1300 rows; most should be out of cooldown
```

You can check how much of your intel is eligible:

```python
import json, urllib.request
from reliquary.miner.persistence import cache_from_env
st = json.loads(urllib.request.urlopen("http://86.38.238.30:8080/state").read())
cd = set(st["cooldown_prompts"])
c = cache_from_env()
res = c._client.table("prompt_outcomes").select("prompt_idx") \
    .eq("checkpoint_hash", st["checkpoint_revision"]) \
    .eq("status", "good").execute()
good = {int(r["prompt_idx"]) for r in res.data}
print(f"known_good={len(good)} eligible_now={len(good - cd)}")
```

## `scripts/prep_dataset.py` — slow, GPU, produces submittable rollouts

Runs the same model the validator scores against, on fresh prompts.
Writes both outcomes AND full M=8 token batches to Supabase. Miner
restarts hydrate these directly into `_pregen_queue` and submit them
without re-gen.

**Run on a separate GPU machine** — same-GPU concurrent runs with the
live miner halve everyone's throughput.

```bash
# On the prep machine:
cd reliquary && source scripts/.env

# Auto-detects validator's current ckpt and loads it
python scripts/prep_dataset.py --cuda 0 --total 500

# Forever
python scripts/prep_dataset.py --cuda 0

# Override ckpt (validator unreachable / specific revision)
python scripts/prep_dataset.py \
    --repo-id R0mAI/reliquary-sn-v23 \
    --revision 1e922bd484ff34457732f853b94258eb422c2f06 \
    --total 500
```

**All flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--total N` | forever | Stop after N prompts considered |
| `--cuda IDX` | 0 | CUDA device index |
| `--hotkey HK` | — | ss58 to stamp on rows |
| `--max-new-tokens N` | 8192 | Full-gen cap (== protocol cap) |
| `--prescreen-rollouts N` | 8 | Rollouts per prescreen mini-batch |
| `--prescreen-max-tokens N` | 1024 | Prescreen cap |
| `--batch-prompts N` | 2 | Prompts per `.generate()` call |
| `--environment` | openmathinstruct | env name |
| `--repo-id ID` | from validator | HF repo override |
| `--revision REV` | from validator | HF revision override |
| `--validator-url URL` | `$RELIQUARY_VALIDATOR_URL` | Where to fetch current ckpt |

### Sizing the prep machine

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | ~16 GB (drop `--batch-prompts` to 1, `--max-new-tokens` to 4096) | 24+ GB |
| Disk | ~30 GB (model + HF cache) | 100 GB |
| Network | dial-up | dial-up |

A Blackwell-class card produces a usable in-zone batch every ~5-15 min
on average (10% good-rate × 80-160 s/batch).

### Checkpoint advance

The script captures the validator's published checkpoint at startup. If
the validator publishes a new revision (every 10 windows per
`CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`), the script keeps writing under
the **old** hash — those rows become inert for the live miner. Kill and
restart the prep script after each ckpt advance.

A persistent ckpt-aware wrapper is left as an exercise — for now,
`while true; do python scripts/prep_dataset.py --total 50; done` in a
tmux session is a serviceable workaround that re-queries the validator
between rounds.

## What the live miner does with the cache

On every checkpoint advance the engine calls `_hydrate_from_cache()`,
which:

1. Loads `prompt_outcomes` WHERE `checkpoint_hash = <new ckpt>`:
   - `status IN ('dud', 'oof')` → added to `_prescreen_dud_set` (picker
     skips the prompt entirely).
   - `status = 'good'` → added to `_known_good_prompts` (picker still
     selects it, but the engine bypasses the prescreen).
2. Loads `pregen_batches` WHERE `checkpoint_hash = <new ckpt>` AND
   `consumed_at IS NULL` — rebuilds each into a `PregenBatch` and
   appends to `_pregen_queue`. The submit worker picks these up
   transparently.

After every classification or pregen-ready event during the run, the
engine writes the corresponding row back to Supabase (fire-and-forget
via `asyncio.to_thread` so the event loop never blocks on network).

After every `/submit` attempt the engine marks the batch consumed —
even on rejection — so a re-hydration after restart doesn't replay a
batch the validator has already seen.

## Validator-rule safety

Scraping itself is read-only and never touches the validator. The
downstream effect on the live miner:

- **No HASH_DUPLICATE risk**: we borrow only the prompt_idx, not the
  tokens. The miner still samples locally → unique rollout hashes.
- **No envelope/signature impact**: the miner signs with its own
  hotkey for its own window's randomness.
- **OOZ risk is unchanged**: scraped intel says "another miner found
  this in-zone for this ckpt"; our local sampling may still drift
  OOZ, but the prior is much better than blind.
- **Cooldown still gates submission**: the picker filters
  `state.cooldown_prompts` before our `_known_good_prompts` check, so
  we never violate the validator's rate-window quota.

See [mining.md](mining.md) for the broader miner architecture and
[validating.md](validating.md) for the validator-side rules these
caches are designed to live within.
