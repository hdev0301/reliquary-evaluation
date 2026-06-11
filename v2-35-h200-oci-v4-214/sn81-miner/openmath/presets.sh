#!/usr/bin/env bash
# OpenMath per-MODE presets (symbolic | numeric) for the shared launcher.
#
# Sourced by ../bin/run_miner.sh AFTER it resolves $MODE and $SN81; this file
# sets ENVIRONMENT / POOL / token budget / curation-K for the OpenMathInstruct
# strategies. Every value stays ${VAR:-default} so each knob remains overridable
# from the environment (e.g. POOL=... TARGET_K=5 MODE=symbolic bash openmath/run.sh).
#
# Math pool builders live in ../dataprep/ (build_inzone_v2.py, build_qwen35_pool.py,
# build_mathmix_pool.py, harvest_inzone.py, ...).

case "$MODE" in
  symbolic)   # 5Hp6EPJd (uid15) play — #1 openmath, cumTAO 7.97, trend up. OMI symbolic-heavy
              # pool (33/67 num/sym). Wins via k=6 reward-oracle curation: symbolic string-equality
              # answers (frac/radical/var) throw off many DISTINCT-WRONG rollouts = the scarce side a
              # 6-correct+2-wrong group needs. Verified 2026-06-08 (sigma EXACTLY 0.4330127 => k=6 89%).
    ENVIRONMENT="${ENVIRONMENT:-openmathinstruct}"
    POOL="${POOL:-$SN81/data/inzone_pool_v2.json}"          # CLONE 5Hp6EPJd (33/67): dataprep/build_inzone_v2.py --sym-ratio 0.67 --int-ratio 0.23 (rest=decimal).
                                                            # Steadier ~50/50 risers (5F7YBWD1/5CX7gQ4f): --sym-ratio 0.50 --int-ratio 0.35. REBUILD after changing (hot pool auto-clears).
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1280}"                # FIX (was 512 -> truncated ~75%): 5Hp6EPJd accepted len med 605, p90 914, max 1672. Cut before \boxed{} = reward 0. Do NOT lower.
    SCREEN_MAX_TOKENS="${SCREEN_MAX_TOKENS:-1280}"          # MUST cover the completion tail (p90 914) or the screen drops long-but-valid prompts as ramblers
    TARGET_K="${TARGET_K:-6}"                               # VERIFIED: 5Hp6EPJd sigma=0.4330127 EXACT in 89% of groups = k=6 (6 correct + 2 wrong). CORRECT abundant,
                                                            # distinct-WRONG scarce -> high k needs FEWER wrongs; the symbolic pool manufactures them.
    GPU_MEM="${GPU_MEM:-0.82}"; POOL_SIZE="${POOL_SIZE:-48}"; GEN_BATCH="${GEN_BATCH:-12}"   # GEN_BATCH=12 suits the longer 1280-tok completions. KV: 384x1280=491k < 1.33M cache. Dial down (32/8) if OOM.
    OVERSAMPLE="${OVERSAMPLE:-64}"                          # deeper to cut out_of_zone (5Hp6EPJd's top reject = 293): need 2 distinct-WRONG per k=6 group; more depth = higher in-zone assembly rate
    OCI_PROMPT_ONLY=0 ;;
  numeric)    # numeric-leaning blend (gsm8k plain-int/decimal). LEGACY pool; current meta is the symbolic
              # mode above (5Hp6EPJd #1). Use this for a lower-variance, numeric-heavy variant.
    ENVIRONMENT="${ENVIRONMENT:-openmathinstruct}"
    POOL="${POOL:-$SN81/data/inzone_pool_topmatch.json}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1280}"                # FIX (was 600): blend winners (5F7YBWD1/5CX7gQ4f) accepted len p90 ~1012-1022; even numeric word-problems run long (CoT)
    SCREEN_MAX_TOKENS="${SCREEN_MAX_TOKENS:-1280}"          # cover the tail so the screen doesn't drop long valid prompts
    TARGET_K="${TARGET_K:-6}"                               # live blend winners run k=6-dominant (5CX7gQ4f k2/6=68%, some k3/5/k4 spread); k=6 needs only 2 scarce wrongs. Override TARGET_K=5 for more spread.
    GPU_MEM="${GPU_MEM:-0.65}"; POOL_SIZE="${POOL_SIZE:-48}"; GEN_BATCH="${GEN_BATCH:-24}"
    OCI_PROMPT_ONLY=0 ;;
  *) echo "FATAL: openmath/presets.sh got non-openmath MODE='$MODE'"; exit 1 ;;
esac
