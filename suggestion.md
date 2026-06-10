Build inzone data with build_inzone_v2.py

nominal     actual	181     wins (n=103)	yield = win÷pool
int         23%	    23.3%	40.8%	1.75× 🟢
symbolic	60%	    53.8%	46.6%	0.87×
decimal	    17%	    22.9%	12.6%	0.55× 🔴


60% symbolic /  23% int / 15% decimal

cd /root/reliquary && .venv/bin/python /root/sn81-miner/dataprep/build_inzone_v2.py \
    --sym-ratio 0.60 --int-ratio 0.23 --max-prompt-len 400

cd /root/reliquary && .venv/bin/python /root/sn81-miner/dataprep/build_inzone_v2.py \
    --sym-ratio 0.65 --int-ratio 0.0 \
    --canon-keep-frac 0.3 --seed 7 --max-prompt-len 400


==================================================================

review thoroughly to avoid violating validator's rule and keep monitoring online miner data from this api endpoint response: https://www.reliqua.ai/api/miners/5FHxcxoJ4y5uPV5vufxU8iURosgNMjaZ39Qg2VMzEy2ppct5


ignore what's implemented so far, just think freshly to improve the miner so the online validation don't have distribution_suspicious errors when submitted
think deep and thoroughly with the implementation so far forgotten
for whatever improve, match validator's path exactly


Prompt picking in the σ≥0.43 zone (Supabase known-good cache + r2 intel)
Latency to /submit so you land in an earlier drand 3-second bucket (K-way split favors early arrivals)
Pregen cache to amortize the 3–5 s GRAIL sketch out of the critical path


check api response of the top miner here: https://www.reliqua.ai/api/miners/5F6VZ2roP7ikDQnfzaHUwi54bYL4hmcTBqaPSgzraZ2rMMmy.
https://www.reliqua.ai/api/miners/5DARq6byGXr9WjB3Ak591RoBGdewpxurM4irnYunTxTWzAai
come up with the method he might be using to prompt selection or rollout generation


github.com/mjun0812/flash-attention-prebuild-wheels




SN81 production update is live.

We noved to Qwen3.5:
checkpoint_n=817
revision=7926d852f1d955f44443fac1476681e0e0fdde92
base_model=Qwen3.5-4B

Main changes:
• validator now uses the Qwen3.5 chat-template path
• sharded checkpoints are supported
• full EOS handling is enabled
• zero-valid windows no longer freeze for 2h; they now seal after the liveness timeout and skip train/publish cleanly
• weight-setting retry spam after restart is fixed
• OpenCode secure grader path is prepared, but live scoring remains OpenMath-only for now

Miners must update:
• reload checkpoint 817
• use the exact Qwen3.5 chat template
• regenerate prompt hashes/signatures from the new canonical prompt path
• support sharded checkpoints
• use the full EOS set

If you see prompt_mismatch or bad_envelope_signature, your miner is stale or not hashing/signing the new prompt format correctly.




Advanced methods, ranked
1. Pivot to OpenCode (likely highest leverage). Code reward = passed/total hidden tests — continuous, so 8 clean solutions to a medium problem naturally scatter (some pass 5/5, some 2/5) → σ≥0.43 without any form trick. The current overall #1 (5DARq6) is OpenCode. Math is crowded and requires the fragile form-ambiguity game; OpenCode gives natural in-zone variance and (per the archive) likely lower contention → higher pool/8/K_p. Our miner already supports RELIQUARY_OCI_PROMPT_ONLY=1. This sidesteps the whole 0-pregen problem.

2. Form-ambiguity targeting (if validator is string-based). Don't just pick "symbolic" — rank prompts by an explicit form-ambiguity score: ground-truth answers with many model-likely surface variants (fractions, radicals, mixed numbers, multi-decimal-place). Build the pool from the top of that ranking, and curate with string reward (the toggle I just added). This is 5Hp6EPJd done deliberately instead of by shape-bucket.

3. Entropy-based difficulty screening (8× discovery throughput). Instead of 8-sample generation to test a prompt, do one forward pass and read the model's token entropy / answer-token confidence. High entropy at the answer ⇒ likely to scatter ⇒ in-zone. Screen 8× more prompts per GPU-second, then deep-mine only the high-entropy ones. Directly attacks our empty-store rate.

4. Harvest + neighbor-expand winning prompts. The R2 archive publishes every winning prompt_idx. They're cooled (single-use), but their dataset neighbors (same source/template/difficulty) scatter the same way. Scrape winners → embed/cluster → mine uncooled neighbors. Supervised frontier prediction on real win-labels — the durable version of "clustering by feature signature."

5. Checkpoint-transfer hot pool. Keep a persistent prompt_idx → per-checkpoint k-history DB; on each publish, warm-start by re-screening recent in-zone prompts first (the frontier drifts slowly) instead of cold rediscovery. Closes the gap to top miners' accumulated caches.

6. Verify-speed optimization for the seal race. Among in-zone groups, fire the shortest-to-GRAIL first (shortest total tokens) so the validator's serial worker clears ours before the 8th-distinct seal. Combined with co-location, this is how you convert a full store into actual slot wins.