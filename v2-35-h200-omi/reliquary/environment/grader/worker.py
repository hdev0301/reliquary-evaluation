"""Sandboxed OpenCode worker.

The worker runs untrusted miner code inside gVisor. It does not receive
hidden assertions or expected values. Each request contains only the code,
an entrypoint, and public call arguments; the trusted grader server compares
the returned primitive value against the hidden expected value.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import math
import sys
from typing import Any


_CRITICAL_BUILTINS = {
    name: getattr(builtins, name)
    for name in ("__import__", "compile", "eval", "exec", "open", "input")
}

_ALLOWED_IMPORT_ROOTS = {
    "abc", "array", "bisect", "collections", "copy", "dataclasses", "decimal",
    "enum", "functools", "heapq", "itertools", "math", "operator", "re",
    "statistics", "string", "typing",
}

_DENIED_BUILTINS = {
    "breakpoint", "compile", "dir", "eval", "exec", "globals", "help", "input",
    "locals", "open", "vars",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    if level != 0 or root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"module {name!r} is not available in the grader sandbox")
    return _CRITICAL_BUILTINS["__import__"](name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    safe = {
        name: value
        for name, value in builtins.__dict__.items()
        if name not in _DENIED_BUILTINS
    }
    safe["__import__"] = _safe_import
    return safe


def _critical_builtins_intact() -> bool:
    return all(
        getattr(builtins, name) is original
        for name, original in _CRITICAL_BUILTINS.items()
    )


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe primitive, or raise TypeError.

    This intentionally rejects arbitrary objects so custom ``__eq__`` /
    comparator tricks never reach trusted scoring.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError("dict key is not a string")
            out[k] = _json_safe(v)
        return out
    raise TypeError(f"unsupported output type: {type(value).__name__}")


def evaluate_call(
    code: str,
    entry: dict[str, Any],
    args: list[Any],
    kwargs: dict[str, Any],
    timeout_s: float,
) -> tuple[Any | None, str]:
    """Execute miner code and call the requested entrypoint.

    Returns ``(output, status)``. The server enforces wall-clock timeouts;
    ``timeout_s`` is accepted for protocol symmetry.
    """
    del timeout_s
    if not code or not code.strip():
        return None, "runtime_error"
    if not isinstance(entry, dict):
        return None, "bad_entry"
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        return None, "bad_request"

    ns: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "__name__": "<miner_code>",
    }
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            exec(compile(code, "<miner_code>", "exec"), ns)
        except ImportError as e:
            if "not available in the grader sandbox" in str(e):
                return None, "forbidden_import"
            return None, "runtime_error"
        except BaseException:
            return None, "runtime_error"
        if not _critical_builtins_intact():
            return None, "tampered"

        try:
            kind = entry.get("kind")
            if kind == "function":
                fn = ns[entry["name"]]
            elif kind == "method":
                cls = ns[entry["class_name"]]
                fn = getattr(cls(), entry["method"])
            else:
                return None, "bad_entry"
            output = fn(*args, **kwargs)
            if not _critical_builtins_intact():
                return None, "tampered"
            return _json_safe(output), "ok"
        except ImportError as e:
            if "not available in the grader sandbox" in str(e):
                return None, "forbidden_import"
            return None, "runtime_error"
        except TypeError as e:
            if "unsupported output type" in str(e) or "dict key" in str(e) or "non-finite" in str(e):
                return None, "bad_output"
            return None, "runtime_error"
        except BaseException:
            return None, "runtime_error"


def _serve_stdin() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            output, status = evaluate_call(
                req.get("code", ""),
                req.get("entry", {}),
                req.get("args", []),
                req.get("kwargs", {}),
                float(req.get("timeout_s", 5.0)),
            )
            resp = {
                "req_id": req.get("req_id", ""),
                "output": output,
                "status": status,
            }
        except BaseException as e:
            resp = {
                "req_id": "",
                "output": None,
                "status": "crash",
                "error": str(e),
            }
        sys.__stdout__.write(json.dumps(resp) + "\n")
        sys.__stdout__.flush()


if __name__ == "__main__":
    _serve_stdin()
