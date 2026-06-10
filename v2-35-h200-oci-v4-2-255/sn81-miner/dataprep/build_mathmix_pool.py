import pyarrow.compute as pc, pyarrow as pa, numpy as np, json, random
from reliquary.environment import load_environment
tbl  = load_environment("openmathinstruct")._dataset.data.table
ans  = pc.utf8_trim_whitespace(tbl["expected_answer"].cast(pa.string()))
src  = tbl["problem_source"].cast(pa.string())
prob = tbl["problem"].cast(pa.string())
plen = pc.utf8_length(prob)

# NUMERIC (~65%): gsm8k family, NON-INTEGER DECIMAL answer, multi-step, medium prompt
# (decimals scatter — model disagrees with itself on format/rounding 0.5 vs 1/2 vs 0.50 vs 50%,
#  producing the 6-correct/2-wrong split k=6 curation needs. Plain ints are bimodal 8/8 or 0/8.)
numeric = pc.and_(pc.is_in(src, value_set=pa.array(["gsm8k","augmented_gsm8k"])),
          pc.and_(pc.match_substring_regex(ans, r"^[\-\+]?[0-9]+\.[0-9]+$"),
          pc.and_(pc.greater_equal(pc.count_substring_regex(prob, r"[0-9]+"), 5),
          pc.and_(pc.greater_equal(plen,150), pc.less_equal(plen,400)))))

# SYMBOLIC (~35%): math family, SIMPLE format-ambiguous answer, SHORT prompt
# (≤18-char single-term answer + ≤300-char prompt keeps completions short — avoids
#  the 5HEAK6 long-CoT ramble that would force a 2048 cap)
symbolic = pc.and_(pc.is_in(src, value_set=pa.array(["math","augmented_math"])),
           pc.and_(pc.match_substring_regex(ans, r"\\frac|\\sqrt|^[A-Za-z]"),
           pc.and_(pc.less_equal(pc.utf8_length(ans), 18),
           pc.and_(pc.greater_equal(plen,80), pc.less_equal(plen,300)))))

num = np.nonzero(numeric.combine_chunks().to_numpy(zero_copy_only=False))[0].tolist()
sym = np.nonzero(symbolic.combine_chunks().to_numpy(zero_copy_only=False))[0].tolist()
random.seed(0)
sym = random.sample(sym, min(len(sym), int(0.54*len(num))))   # 0.54 -> ~35% of the blend
pool = sorted(num + sym)
json.dump(pool, open("data/inzone_pool_mathmix.json","w"))
print("pool=%d  numeric=%d symbolic=%d  (N:S = %d:%d)" %
      (len(pool), len(num), len(sym), round(100*len(num)/len(pool)), round(100*len(sym)/len(pool))))
