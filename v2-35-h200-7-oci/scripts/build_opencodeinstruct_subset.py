"""Build the deterministic subset of nvidia/OpenCodeInstruct.

Run once offline (typically on a beefy box with disk + network).
Filters in order:
  1. Drop rows whose reference solution did not pass all its own
     tests (average_test_score < 1.0).
  2. Parse the unit_tests column (string-encoded list) — drop on
     parse failure.
  3. Convert simple deterministic asserts into structured black-box
     cases. Hidden assertion source is never stored in the output.
  4. Run a double-execution check on the reference solution using the
     structured cases (twice with different PYTHONHASHSEED) — drop on
     mismatch.
  5. Push the resulting private subset to HF Hub as
     R0mAI/opencodeinstruct-structured-subset.
  6. Optionally publish a public prompt-only mirror with the same row order
     for miners. The mirror contains `input` and `id` only.

Designed so the per-row filter functions are pure-Python and
testable without HuggingFace, network, or subprocess.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

DETERMINISM_TIMEOUT_SECONDS = float(os.environ.get("RELIQUARY_OCI_BUILD_TIMEOUT_SECONDS", "5"))

_FENCE_RE = re.compile(
    r"(```|~~~)(?:python3?|py)?\s*\n(.*?)\n\1",
    re.DOTALL,
)


# Conservative regex: any of these tokens anywhere in the test code
# disqualifies the row. False positives are fine — we have 5M rows
# and only need ~2-3M deterministic ones.
_NONDET_PATTERNS = re.compile(
    r"\b(?:import\s+(?:random|time|datetime|socket|urllib|requests|os|"
    r"subprocess|threading|multiprocessing|asyncio|signal|select)\b"
    r"|from\s+(?:random|time|datetime|socket|urllib|requests|os|"
    r"subprocess|threading|multiprocessing|asyncio|signal|select)\s+import"
    r"|\brandom\.|\btime\.|\bdatetime\.|\bsocket\.|\burllib\.|\brequests\."
    r"|\bos\.environ|\bsubprocess\.|\bthreading\.|\bmultiprocessing\.)"
)


def parse_unit_tests(raw: str) -> Optional[list[str]]:
    """Parse the string-encoded list of tests. Return None on failure."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    if not all(isinstance(t, str) for t in parsed):
        return None
    return parsed


def has_nondeterministic_pattern(test_src: str) -> bool:
    return _NONDET_PATTERNS.search(test_src) is not None


def filter_tests(tests: list[str]) -> list[str]:
    """Keep only tests free of non-deterministic patterns."""
    return [t for t in tests if not has_nondeterministic_pattern(t)]


def extract_reference_code(output: str) -> str:
    """Return executable Python from a reference-solution field.

    OpenCodeInstruct rows often wrap solutions in Markdown fences. The
    structured-case builder must execute the code itself, not the fence text,
    when checking that kept rows are deterministic and self-consistent.
    """
    if not output:
        return ""
    matches = _FENCE_RE.findall(output)
    if matches:
        return matches[-1][1]
    return output


def keep_row(row: dict) -> bool:
    """Stage-1 filter: reference solution must pass all its own tests."""
    return float(row.get("average_test_score", 0.0)) >= 1.0


def _json_safe_literal(node: ast.AST) -> Any:
    value = ast.literal_eval(node)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [_json_safe_value(v) for v in value]
    return _json_safe_value(value)


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("dict keys must be strings")
            out[k] = _json_safe_value(v)
        return out
    raise ValueError(f"unsupported literal type: {type(value).__name__}")


def _call_to_case(call: ast.Call, expected: Any) -> Optional[dict]:
    entry: dict[str, str]
    if isinstance(call.func, ast.Name):
        entry = {"kind": "function", "name": call.func.id}
    elif (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and not call.func.value.args
        and not call.func.value.keywords
    ):
        entry = {
            "kind": "method",
            "class_name": call.func.value.func.id,
            "method": call.func.attr,
        }
    else:
        return None

    try:
        args = [_json_safe_literal(a) for a in call.args]
        kwargs = {
            kw.arg: _json_safe_literal(kw.value)
            for kw in call.keywords
            if kw.arg is not None
        }
    except (ValueError, TypeError):
        return None
    if len(kwargs) != len(call.keywords):
        return None
    return {
        "entry": entry,
        "args": args,
        "kwargs": kwargs,
        "expected": expected,
        "compare": "exact",
    }


def structure_test(test_src: str) -> Optional[dict]:
    """Convert one simple assert into a structured case.

    Supported forms:
      * assert fn(args...) == literal
      * assert literal == fn(args...)
      * assert fn(args...)
      * assert not fn(args...)
      * same call shapes for Solution().method(args...)
    """
    try:
        mod = ast.parse(test_src, mode="exec")
    except SyntaxError:
        return None
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(mod)):
        return None
    if len(mod.body) != 1 or not isinstance(mod.body[0], ast.Assert):
        return None
    expr = mod.body[0].test

    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and isinstance(expr.ops[0], ast.Eq):
        left = expr.left
        right = expr.comparators[0]
        if isinstance(left, ast.Call):
            try:
                expected = _json_safe_literal(right)
            except (ValueError, TypeError):
                return None
            return _call_to_case(left, expected)
        if isinstance(right, ast.Call):
            try:
                expected = _json_safe_literal(left)
            except (ValueError, TypeError):
                return None
            return _call_to_case(right, expected)
        return None

    if isinstance(expr, ast.Call):
        return _call_to_case(expr, True)

    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not) and isinstance(expr.operand, ast.Call):
        return _call_to_case(expr.operand, False)

    return None


def structure_tests(tests: list[str]) -> list[dict]:
    cases = []
    for test in filter_tests(tests):
        case = structure_test(test)
        if case is not None:
            cases.append(case)
    return cases


def double_execute(code: str, cases: list[dict]) -> bool:
    """Run (code, structured cases) twice with different PYTHONHASHSEEDs.

    Returns True iff both runs pass every structured case.
    """
    runner = (
        "import json,math,sys\n"
        "data=json.loads(sys.stdin.read())\n"
        "ns={}\n"
        "try: exec(data['code'], ns)\n"
        "except: pass\n"
        "def eq(a,b):\n"
        "    if isinstance(a,bool) or isinstance(b,bool): return type(a) is type(b) and a==b\n"
        "    if isinstance(a,(int,float)) and isinstance(b,(int,float)):\n"
        "        return math.isclose(float(a),float(b),rel_tol=1e-6,abs_tol=1e-9) if (isinstance(a,float) or isinstance(b,float)) else a==b\n"
        "    if isinstance(a,list) and isinstance(b,list): return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))\n"
        "    if isinstance(a,dict) and isinstance(b,dict): return set(a)==set(b) and all(eq(a[k],b[k]) for k in a)\n"
        "    return type(a) is type(b) and a==b\n"
        "def call(c):\n"
        "    e=c['entry']\n"
        "    if e['kind']=='function': fn=ns[e['name']]\n"
        "    else: fn=getattr(ns[e['class_name']](), e['method'])\n"
        "    return fn(*c.get('args',[]), **c.get('kwargs',{}))\n"
        "p=0\n"
        "for c in data['cases']:\n"
        "    try:\n"
        "        p += 1 if eq(call(c), c.get('expected')) else 0\n"
        "    except: pass\n"
        "print(p)\n"
    )
    payload = json.dumps({"code": code, "cases": cases})
    try:
        out_seed0 = subprocess.run(
            [sys.executable, "-c", runner], input=payload, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": "0"}, timeout=DETERMINISM_TIMEOUT_SECONDS,
        )
        out_seed1 = subprocess.run(
            [sys.executable, "-c", runner], input=payload, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": "1"}, timeout=DETERMINISM_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    expected = str(len(cases))
    return out_seed0.stdout.strip() == expected and out_seed1.stdout.strip() == expected


def process_row(row: dict, *, include_reference_output: bool = False) -> Optional[dict]:
    """Apply all filters to one row. Return the kept row (with
    `structured_cases` added) or None to drop."""
    if not keep_row(row):
        return None
    tests = parse_unit_tests(row.get("unit_tests", ""))
    if tests is None:
        return None
    cases = structure_tests(tests)
    if not cases:
        return None
    code = extract_reference_code(row.get("output", ""))
    if not double_execute(code, cases):
        return None
    out = {
        "input": row["input"],
        # Store as JSON text to avoid Arrow union-type problems: expected
        # values can be ints, bools, strings, lists, dicts, or null across
        # rows. The runtime environment parses this field back into cases.
        "structured_cases": json.dumps(cases, sort_keys=True, separators=(",", ":")),
        "id": row.get("id", ""),
    }
    if include_reference_output:
        out["output"] = code
    return out


def prompt_only_rows(rows: list[dict]) -> list[dict]:
    """Return the public miner-facing mirror rows.

    The prompt-only dataset must preserve row order with the private
    structured dataset so prompt_idx maps to the same text on miners and
    validators, while omitting hidden cases and reference solutions.
    """
    return [{"input": row["input"], "id": row.get("id", "")} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="nvidia/OpenCodeInstruct")
    parser.add_argument("--target-repo", default="R0mAI/opencodeinstruct-structured-subset")
    parser.add_argument("--prompt-target-repo", default="R0mAI/opencodeinstruct-prompts",
                        help="Public prompt-only mirror repo for miners.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap on rows to process — for dry-runs.")
    parser.add_argument("--target-kept", type=int, default=None,
                        help="Stop once this many filtered rows have been kept.")
    parser.add_argument("--output-dir", default="./opencodeinstruct-subset",
                        help="Local save_to_disk path when --push is not used.")
    parser.add_argument("--prompt-output-dir", default=None,
                        help="Optional local save_to_disk path for the prompt-only mirror.")
    parser.add_argument("--push", action="store_true",
                        help="Push to HF Hub (requires HF_TOKEN).")
    parser.add_argument("--push-prompt-repo", action="store_true",
                        help="Push the prompt-only mirror to HF Hub (requires HF_TOKEN).")
    parser.add_argument("--public", action="store_true",
                        help="Make the pushed dataset public. Default is private.")
    parser.add_argument("--private-prompt-repo", action="store_true",
                        help="Make the prompt-only mirror private. Default is public.")
    parser.add_argument("--include-reference-output", action="store_true",
                        help="Include reference solutions. Do not use for production.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    import datasets as hf
    ds = hf.load_dataset(args.source, split="train", streaming=True)

    kept = []
    i = -1
    for i, row in enumerate(ds):
        if args.max_rows is not None and i >= args.max_rows:
            break
        out = process_row(row, include_reference_output=args.include_reference_output)
        if out:
            kept.append(out)
            if args.target_kept is not None and len(kept) >= args.target_kept:
                break
        if i % 1000 == 0:
            logger.info("processed=%d kept=%d", i, len(kept))

    logger.info("final: processed=%d kept=%d", i + 1, len(kept))
    out_ds = hf.Dataset.from_list(kept)

    if args.output_dir:
        out_ds.save_to_disk(args.output_dir)
        logger.info("saved locally to %s", args.output_dir)

    prompt_ds = hf.Dataset.from_list(prompt_only_rows(kept))
    if args.prompt_output_dir:
        prompt_ds.save_to_disk(args.prompt_output_dir)
        logger.info("saved prompt-only mirror locally to %s", args.prompt_output_dir)

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN env var is required to push.")
        out_ds.push_to_hub(args.target_repo, token=token, private=not args.public)
        logger.info(
            "pushed %d rows to %s private=%s",
            len(kept), args.target_repo, not args.public,
        )

    if args.push_prompt_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN env var is required to push.")
        prompt_ds.push_to_hub(
            args.prompt_target_repo,
            token=token,
            private=args.private_prompt_repo,
        )
        logger.info(
            "pushed %d prompt rows to %s private=%s",
            len(kept), args.prompt_target_repo, args.private_prompt_repo,
        )


if __name__ == "__main__":
    main()
