#!/usr/bin/env python3

import math
from typing import Any, Dict, List, Optional, Sequence


_STATE: Dict[str, Any] = {
    "enabled": False,
    "public_calls": 0,
    "private_calls": 0,
    "public_time_s": 0.0,
    "private_time_s": 0.0,
    "public_mul_est": 0,
    "private_mul_est": 0,
    "preview": [],
}


def reset(enabled: bool = False):
    global _STATE
    _STATE = {
        "enabled": bool(enabled),
        "public_calls": 0,
        "private_calls": 0,
        "public_time_s": 0.0,
        "private_time_s": 0.0,
        "public_mul_est": 0,
        "private_mul_est": 0,
        "preview": [],
    }


def is_enabled() -> bool:
    return bool(_STATE.get("enabled", False))


def _shape_list(obj: Any) -> Optional[List[int]]:
    try:
        size = obj.size() if hasattr(obj, "size") else None
        if size is None:
            return None
        return [int(v) for v in tuple(size)]
    except Exception:
        return None


def _broadcast_shape(lhs: Sequence[int], rhs: Sequence[int]) -> Optional[List[int]]:
    out: List[int] = []
    li = list(lhs)[::-1]
    ri = list(rhs)[::-1]
    for idx in range(max(len(li), len(ri))):
        a = li[idx] if idx < len(li) else 1
        b = ri[idx] if idx < len(ri) else 1
        if a == 1:
            out.append(int(b))
        elif b == 1 or a == b:
            out.append(int(a))
        else:
            return None
    return out[::-1]


def _prod(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def estimate_mul_count(lhs: Any, rhs: Any) -> Optional[int]:
    lhs_shape = _shape_list(lhs)
    rhs_shape = _shape_list(rhs)
    if not lhs_shape or not rhs_shape:
        return None

    lhs_dim = len(lhs_shape)
    rhs_dim = len(rhs_shape)

    try:
        if lhs_dim == 1 and rhs_dim == 1:
            return int(lhs_shape[0])
        if lhs_dim == 1 and rhs_dim >= 2:
            batch_shape = _broadcast_shape([], rhs_shape[:-2])
            if batch_shape is None:
                return None
            k = int(lhs_shape[0])
            n = int(rhs_shape[-1])
            return _prod(batch_shape) * k * n
        if lhs_dim >= 2 and rhs_dim == 1:
            batch_shape = _broadcast_shape(lhs_shape[:-2], [])
            if batch_shape is None:
                return None
            m = int(lhs_shape[-2])
            k = int(lhs_shape[-1])
            return _prod(batch_shape) * m * k
        if lhs_dim >= 2 and rhs_dim >= 2:
            batch_shape = _broadcast_shape(lhs_shape[:-2], rhs_shape[:-2])
            if batch_shape is None:
                return None
            m = int(lhs_shape[-2])
            k = int(lhs_shape[-1])
            n = int(rhs_shape[-1])
            return _prod(batch_shape) * m * k * n
    except Exception:
        return None
    return None


def record(kind: str, lhs: Any, rhs: Any, elapsed_s: float):
    if not is_enabled():
        return
    kind_key = "public" if str(kind).lower() == "public" else "private"
    _STATE[f"{kind_key}_calls"] += 1
    _STATE[f"{kind_key}_time_s"] += float(elapsed_s)
    mul_est = estimate_mul_count(lhs, rhs)
    if mul_est is not None:
        _STATE[f"{kind_key}_mul_est"] += int(mul_est)
    if len(_STATE["preview"]) < 12:
        _STATE["preview"].append(
            {
                "kind": kind_key,
                "lhs_shape": _shape_list(lhs),
                "rhs_shape": _shape_list(rhs),
                "elapsed_s": float(elapsed_s),
                "mul_est": mul_est,
            }
        )


def summary() -> Dict[str, Any]:
    public_calls = int(_STATE["public_calls"])
    private_calls = int(_STATE["private_calls"])
    public_time_s = float(_STATE["public_time_s"])
    private_time_s = float(_STATE["private_time_s"])
    public_mul_est = int(_STATE["public_mul_est"])
    private_mul_est = int(_STATE["private_mul_est"])
    total_calls = public_calls + private_calls
    total_time_s = public_time_s + private_time_s
    total_mul_est = public_mul_est + private_mul_est
    return {
        "enabled": bool(_STATE["enabled"]),
        "public_calls": public_calls,
        "private_calls": private_calls,
        "total_calls": total_calls,
        "public_time_s": public_time_s,
        "private_time_s": private_time_s,
        "total_time_s": total_time_s,
        "public_mul_est": public_mul_est,
        "private_mul_est": private_mul_est,
        "total_mul_est": total_mul_est,
        "avg_public_time_ms": (public_time_s / public_calls * 1000.0) if public_calls > 0 else 0.0,
        "avg_private_time_ms": (private_time_s / private_calls * 1000.0) if private_calls > 0 else 0.0,
        "avg_public_mul_est": (public_mul_est / public_calls) if public_calls > 0 else 0.0,
        "avg_private_mul_est": (private_mul_est / private_calls) if private_calls > 0 else 0.0,
        "preview": list(_STATE["preview"]),
    }
