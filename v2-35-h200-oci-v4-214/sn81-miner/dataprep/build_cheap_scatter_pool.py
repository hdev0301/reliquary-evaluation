import pyarrow.compute as pc, pyarrow as pa, numpy as np, json, random
from reliquary.environment import load_environment
tbl  = load_environment("openmathinstruct")._dataset.data.table
ans  = pc.utf8_trim_whitespace(tbl["expected_answer"].cast(pa.string()))
src  = tbl["problem_source"].cast(pa.string())
prob = tbl["problem"].cast(pa.string())
plen = pc.utf8_length(prob)
nnum = pc.count_substring_regex(prob, r"[0-9]+")

# GOAL: CHEAP-TO-VERIFY (short completions) + SCATTER (format-ambiguous answers).
# Verify cost ~ prompt+completion tokens. Short prompt + few reasoning steps -> the
# model reaches \boxed{} fast -> short completion -> ~2-4s GRAIL verify (vs ~10-25s for
# the multi-step mathmix). Scatter still comes from format ambiguity (0.375 vs 0.38 vs 3/8),
# NOT from problem difficulty -- so we keep the k=correct/wrong split for curation while
# slashing verify cost (the drain-reachability lever).

# CHEAP-NUMERIC (~60%): SHORT, FEW-step problem with a NON-INTEGER DECIMAL answer.
numeric = pc.and_(pc.is_in(src, value_set=pa.array(["gsm8k","augmented_gsm8k","math","augmented_math"])),
          pc.and_(pc.match_substring_regex(ans, r"^[\-\+]?[0-9]+\.[0-9]+$"),
          pc.and_(pc.less_equal(nnum, 4),
          pc.and_(pc.greater_equal(plen,40), pc.less_equal(plen,180)))))

# CHEAP-SYMBOLIC (~40%): SHORT problem with a format-ambiguous fraction/radical answer
# (1/2 vs 0.5 vs \frac{1}{2} ; \sqrt scatter). Single-term, short prompt -> short CoT.
symbolic = pc.and_(pc.is_in(src, value_set=pa.array(["math","augmented_math","gsm8k","augmented_gsm8k"])),
           pc.and_(pc.match_substring_regex(ans, r"\\frac|\\sqrt|^[0-9]+/[0-9]+$"),
           pc.and_(pc.less_equal(pc.utf8_length(ans), 14),
           pc.and_(pc.greater_equal(plen,40), pc.less_equal(plen,180)))))

num = np.nonzero(numeric.combine_chunks().to_numpy(zero_copy_only=False))[0].tolist()
sym = np.nonzero(symbolic.combine_chunks().to_numpy(zero_copy_only=False))[0].tolist()
random.seed(0)
sym = random.sample(sym, min(len(sym), int(0.67*len(num))))   # ~40% of the blend
pool = sorted(set(num + sym))
json.dump(pool, open("data/inzone_pool_cheapscatter.json","w"))
print("pool=%d  numeric=%d symbolic=%d  (N:S = %d:%d)" %
      (len(pool), len(num), len(sym),
       round(100*len(num)/max(len(pool),1)), round(100*len(sym)/max(len(pool),1))))
