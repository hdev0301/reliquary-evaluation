# SN81 honest pregen → Supabase → submit pipeline

Decouples the **GPU-heavy pregen** from the **latency-critical submit**, so the
submitter can run as a tiny GPU-free process (ideally on a low-latency box near the
validator) and pull prepared groups from Supabase. This is the legitimate fix for the
`proof_admission` slot race — submission latency, not generation, is the bottleneck.

Honest by construction: the producer runs `RELIQUARY_CURATE=0` (genuine first-8
natural in-zone groups). It only changes *where prepared groups are stored*, not how
they're formed. Single hotkey (`stardev`); no multi-identity fan-out.

## Pieces
- `schema.sql`     — the `pregen` table (run once in the Supabase SQL editor).
- `sb.py`          — Supabase REST client + (de)serialization (incl. int8 GRAIL `buckets`).
- `producer.py`    — GPU: same stack as `cli.main mine`, but writes each prepared group
                      to Supabase and never submits. `run_producer.sh` launches it.
- `consumer.py`    — GPU-FREE: reuses `MiningEngine.mine_window` UNCHANGED, fed by a
                      Supabase pregen shim; binds to live window randomness, signs, POSTs.
                      `run_consumer.sh` launches it.
- `.env`           — Supabase URL/key/table (mode 600). Sourced by both run scripts.

## Data model (`pregen`)
One row = one submit-ready group for one hotkey + checkpoint. `rollouts` jsonb carries
the FULL GRAIL artifacts: `tokens, prompt_length, completion_length, reward,
token_logprobs, p_stop, buckets_b64/shape/dtype`. Tagged `tier='honest_first8'`,
`miner_hotkey=<stardev ss58>`, `status` ready→consumed. Unique
`(checkpoint_hash, prompt_idx, miner_hotkey)` prevents HASH_DUPLICATE resubmits.

## Setup
1. Create the table — paste `schema.sql` into the Supabase SQL editor and run it
   (the service_role REST key cannot run DDL).
2. Producer (GPU box):  `bash run_producer.sh`   → tail `logs/sb_producer.log`
   (look for `supabase <- group prompt=… sigma=…`).
3. Consumer (GPU-free): `bash run_consumer.sh`   → tail `logs/sb_consumer.log`
   (look for `submitted window=… accepted=…` / `verdict ACCEPTED`).

The consumer only submits rows whose `checkpoint_hash` matches the validator's CURRENT
published checkpoint, so keep the producer on the same checkpoint.

## Notes / limits
- The consumer claims (marks `consumed`) a group when it pops it; `batch_filled` groups
  are not re-queued (the producer replenishes). Fine for a single submitter.
- `model_name` in the commit is the published repo id; the per-window randomness binding
  and signing happen fresh at submit, so stored groups stay valid while the checkpoint matches.
- Producer and consumer can run on different machines sharing the same Supabase project
  and the same `stardev` wallet (the consumer needs the hotkey to sign).
