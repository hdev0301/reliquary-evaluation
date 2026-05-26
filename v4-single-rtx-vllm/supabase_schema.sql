-- Reliquary miner Supabase schema. Paste into:
--   https://supabase.com/dashboard/project/<your-ref>/sql/new
-- and click Run. Idempotent — safe to re-apply.

CREATE TABLE IF NOT EXISTS prompt_outcomes (
    prompt_idx          BIGINT NOT NULL,
    checkpoint_hash     TEXT NOT NULL,
    k                   INTEGER NOT NULL,
    sigma               DOUBLE PRECISION NOT NULL,
    status              TEXT NOT NULL,
    avg_completion_len  INTEGER,
    truncated_count     INTEGER,
    miner_hotkey        TEXT,
    last_seen           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (prompt_idx, checkpoint_hash)
);
CREATE INDEX IF NOT EXISTS idx_outcomes_ckpt_status
    ON prompt_outcomes (checkpoint_hash, status);

CREATE TABLE IF NOT EXISTS pregen_batches (
    id                  BIGSERIAL PRIMARY KEY,
    prompt_idx          BIGINT NOT NULL,
    checkpoint_hash     TEXT NOT NULL,
    local_n             INTEGER NOT NULL,
    sigma               DOUBLE PRECISION NOT NULL,
    k                   INTEGER NOT NULL,
    rollouts            JSONB NOT NULL,
    miner_hotkey        TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consumed_at         TIMESTAMPTZ,
    UNIQUE (prompt_idx, checkpoint_hash)
);
CREATE INDEX IF NOT EXISTS idx_pregen_unconsumed_ckpt
    ON pregen_batches (checkpoint_hash)
    WHERE consumed_at IS NULL;
