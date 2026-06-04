-- Reliquary SN81 honest pregen->submit pipeline: dedicated table.
-- Run ONCE in the Supabase SQL editor (the service_role REST key cannot run DDL).
-- A "row" = one submit-ready, honest (CURATE=0) prepared group for ONE hotkey,
-- carrying the full GRAIL artifacts (tokens + per-token buckets + logprobs) so a
-- GPU-free consumer can bind it to the live window randomness, sign, and POST.

create table if not exists public.pregen (
  id              bigint generated always as identity primary key,
  prompt_idx      bigint            not null,
  checkpoint_hash text              not null,   -- validator-published revision; submit only when it matches
  model_name      text              not null,   -- HF name used in the commit binding
  hidden_dim      integer,                       -- for GRAILVerifier (r_vec gen ignores it, kept for correctness)
  sigma           double precision  not null,   -- population std of the group's rewards (in-zone band)
  n_correct       integer,                       -- descriptive: #rollouts with reward>=1 (NOT a selection knob)
  rollouts        jsonb             not null,   -- [{tokens, prompt_length, completion_length, reward,
                                                 --   token_logprobs, p_stop, buckets_b64, buckets_shape, buckets_dtype}]
  miner_hotkey    text              not null,   -- target hotkey ss58 (stardev)
  tier            text              not null default 'honest_first8',
  status          text              not null default 'ready',   -- ready | claimed | consumed
  created_at      timestamptz       not null default now(),
  consumed_at     timestamptz,
  -- one prepared group per (checkpoint, prompt, hotkey): prevents HASH_DUPLICATE resubmits
  unique (checkpoint_hash, prompt_idx, miner_hotkey)
);

-- fast pull of the next ready groups for a checkpoint
create index if not exists pregen_pull
  on public.pregen (checkpoint_hash, miner_hotkey, created_at)
  where status = 'ready';
