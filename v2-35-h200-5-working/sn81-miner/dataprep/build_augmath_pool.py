"""Build the augmented_math pool for the NO-THINKING + \\boxed{} regime (post #76/#78).
The model now boxes LaTeX answers, so non-numeric augmented_math is minable again and
(being harder) yields more distinct-WRONG completions -> more curatable groups (the
binding constraint for k=5/6). 'augmented_math only' per user. Writes data/inzone_pool_augmath.json."""
import json, traceback
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import numpy as np
    from reliquary.environment import load_environment
    env = load_environment("openmathinstruct")
    tbl = env._dataset.data.table
    src = tbl["problem_source"].cast(pa.string())
    prob = tbl["problem"].cast(pa.string())
    src_mask = pc.equal(src, "augmented_math")
    len_mask = pc.less_equal(pc.utf8_length(prob), 400)   # short prompts -> cheaper screen, faster discovery
    mask = pc.and_(src_mask, len_mask)
    m = mask.combine_chunks().to_numpy(zero_copy_only=False)
    idxs = np.nonzero(m)[0].astype(int).tolist()
    out = "/root/sn81-miner/data/inzone_pool_augmath.json"
    json.dump(idxs, open(out, "w"))
    print("BUILT", out, "size:", len(idxs))
    ans = tbl["expected_answer"].cast(pa.string()).combine_chunks().to_pylist()
    print("sample expected_answers:", [ans[i] for i in idxs[:12]])
except Exception:
    traceback.print_exc()
