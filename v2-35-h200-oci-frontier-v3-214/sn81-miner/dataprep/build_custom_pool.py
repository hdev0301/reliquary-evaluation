import pyarrow.compute as pc, pyarrow as pa, numpy as np, json
from reliquary.environment import load_environment
tbl  = load_environment("openmathinstruct")._dataset.data.table
ans  = pc.utf8_trim_whitespace(tbl["expected_answer"].cast(pa.string()))
src  = tbl["problem_source"].cast(pa.string())
prob = tbl["problem"].cast(pa.string())

plain   = pc.match_substring_regex(ans, r"^[\-\+]?[0-9]+(\.[0-9]+)?$")        # int OR decimal (94% of top miner)
gsm     = pc.is_in(src, value_set=pa.array(["gsm8k", "augmented_gsm8k"]))
multi   = pc.greater_equal(pc.count_substring_regex(prob, r"[0-9]+"), 4)      # >=4 numbers => multi-step
length  = pc.and_(pc.greater_equal(pc.utf8_length(prob), 120),               # not trivial one-liners
                  pc.less_equal(pc.utf8_length(prob), 400))                   # but still short => clean EOS
mask = pc.and_(pc.and_(plain, gsm), pc.and_(multi, length))
idxs = np.nonzero(mask.combine_chunks().to_numpy(zero_copy_only=False))[0].astype(int).tolist()
json.dump(idxs, open("data/inzone_pool_custom.json", "w"))
