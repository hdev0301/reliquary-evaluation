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