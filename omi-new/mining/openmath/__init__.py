"""OpenMathInstruct optimized miner (vLLM + pregen + answer-match oracle).

Sibling of ``mining.opencode``: same env-agnostic machinery in ``mining.common``
(pregen pool, GRAIL proof, vLLM worker, fast fire), with a math-specific producer
that grades by ANSWER EQUALITY against the dataset's public ``expected_answer``
instead of running code in a sandbox.
"""
