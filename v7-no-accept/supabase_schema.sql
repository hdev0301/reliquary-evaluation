-- Reliquary miner — Supabase prepared-prompt schema.
-- Paste this into your Supabase project's SQL editor once before first launch.
--
-- The `prompt_outcomes` table tracks per-(ckpt_n, prompt_idx, hotkey, window_n)
-- σ outcomes from the local zone pre-check. The picker
-- (reliquary/miner/prompt_picker.py) hydrates a "known-bad" set on startup
-- and after every checkpoint advance, then skips those prompts in addition
-- to the validator's own cooldown_prompts set.
--
-- Insert-only — we keep full history so we can later resurrect prompts whose
-- σ improved on a newer fine-tune. A nightly job (not included) can purge
-- rows older than 30 days or older than N checkpoints if storage matters.

create table if not exists prompt_outcomes (
    prompt_idx  integer       not null,
    ckpt_n      integer       not null,
    k_correct   smallint      not null,
    sigma       real          not null,
    -- 'submitted_accepted', 'submitted_rejected:<reason>', or 'zone_skip'
    outcome     text          not null,
    hotkey      text          not null,
    window_n    integer       not null,
    inserted_at timestamptz   not null default now(),
    primary key (prompt_idx, ckpt_n, hotkey, window_n)
);

-- Hot path: hydrate known-bad set for current checkpoint.
create index if not exists prompt_outcomes_ckpt_sigma_idx
    on prompt_outcomes (ckpt_n, sigma);

-- For potential future cross-hotkey aggregation queries.
create index if not exists prompt_outcomes_ckpt_idx
    on prompt_outcomes (ckpt_n, prompt_idx);
