"""Build a Qwen3.5-minable pool (vectorized pyarrow). Keep prompts whose
expected_answer is a PLAIN number/fraction (the reward fallback can extract these
WITHOUT \\boxed{}, which 7926 never emits) from the numeric sources gsm8k /
augmented_gsm8k. Excludes augmented_math (non-numeric -> needs \\boxed -> reward 0)."""
import json, traceback
try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    from reliquary.environment import load_environment
    env = load_environment("openmathinstruct")
    tbl = env._dataset.data.table
    ans = tbl["expected_answer"].cast(pa.string())
    src = tbl["problem_source"].cast(pa.string())
    prob = tbl["problem"].cast(pa.string())
    ans_t = pc.utf8_trim_whitespace(ans)
    num_mask = pc.match_substring_regex(ans_t, r"^[\-\+]?[0-9]+(\.[0-9]+)?(/[0-9]+)?$")
    src_mask = pc.is_in(src, value_set=pa.array(["gsm8k", "augmented_gsm8k"]))
    len_mask = pc.less_equal(pc.utf8_length(prob), 400)
    mask = pc.and_(pc.and_(num_mask, src_mask), len_mask)
    m = mask.combine_chunks().to_numpy(zero_copy_only=False)
    idxs = np.nonzero(m)[0].astype(int).tolist()
    json.dump(idxs, open("/root/inzone_pool_qwen35.json", "w"))
    print("BUILT /root/inzone_pool_qwen35.json size:", len(idxs))
    # breakdown + samples
    src_np = src.combine_chunks().to_pylist()
    ans_np = ans.combine_chunks().to_pylist()
    from collections import Counter
    c = Counter(src_np[i] for i in idxs[:50000])
    print("source breakdown (first 50k):", dict(c))
    print("sample gts:", [ans_np[i] for i in idxs[:10]])
except Exception:
    traceback.print_exc()
