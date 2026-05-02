#!/usr/bin/env python3

# Based on the pasted beaver.py. This standalone variant adds explicit
# forward-to-backward left-token reuse without using local tag / anchor context.

import crypten
import crypten.communicator as comm
import logging
import torch
from crypten.common.util import count_wraps
from crypten.config import cfg

logger = logging.getLogger(__name__)


_STRUCTURAL_REUSE_STATS = {
    "same_left_group_calls": 0,
    "same_left_group_success": 0,
    "same_left_group_fallback": 0,
    "same_left_group_outputs": 0,
    "left_token_requested": 0,
    "left_token_created": 0,
    "left_token_reserved_triples": 0,
    "left_token_consume_calls": 0,
    "left_token_consume_success": 0,
    "left_token_consume_fallback": 0,
    "fallback_active_security": 0,
    "fallback_non_2d": 0,
    "fallback_missing_provider_api": 0,
    "fallback_provider_not_implemented": 0,
    "fallback_missing_reserved_triple": 0,
}


def _inc_stat(name, value=1):
    _STRUCTURAL_REUSE_STATS[name] = _STRUCTURAL_REUSE_STATS.get(name, 0) + value


def get_structural_reuse_stats(reset=False):
    """Returns structural reuse counters for experiment summaries."""
    stats = dict(_STRUCTURAL_REUSE_STATS)
    if reset:
        reset_structural_reuse_stats()
    return stats


def reset_structural_reuse_stats():
    for key in list(_STRUCTURAL_REUSE_STATS.keys()):
        _STRUCTURAL_REUSE_STATS[key] = 0


def log_structural_reuse_stats(level=logging.INFO):
    logger.log(level, "[structural-fb-reuse] stats=%s", get_structural_reuse_stats())


def structural_reuse_capability():
    """Reports whether the current runtime can use explicit token reuse."""
    provider = crypten.mpc.get_default_provider()
    return {
        "provider": getattr(provider, "NAME", type(provider).__name__),
        "protocol": getattr(cfg.mpc, "protocol", None),
        "active_security": bool(getattr(cfg.mpc, "active_security", False)),
        "has_same_left_triples": hasattr(provider, "generate_additive_triples_with_same_left"),
        "can_use_token_reuse": (
            not bool(getattr(cfg.mpc, "active_security", False))
            and hasattr(provider, "generate_additive_triples_with_same_left")
        ),
    }


def log_structural_reuse_capability(level=logging.INFO):
    logger.log(
        level,
        "[structural-fb-reuse] capability=%s",
        structural_reuse_capability(),
    )


class IgnoreEncodings:
    """Context Manager to ignore tensor encodings"""

    def __init__(self, list_of_tensors):
        self.list_of_tensors = list_of_tensors
        self.encodings_cache = [tensor.encoder.scale for tensor in list_of_tensors]

    def __enter__(self):
        for tensor in self.list_of_tensors:
            tensor.encoder._scale = 1

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for i, tensor in enumerate(self.list_of_tensors):
            tensor.encoder._scale = self.encodings_cache[i]


def _fallback_group_result(results, token, return_left_token):
    return (results, token) if return_left_token else results


def _fallback_group(reason, x, grouped_args, return_left_token):
    _inc_stat("same_left_group_fallback")
    _inc_stat(reason)
    if return_left_token:
        _inc_stat("left_token_requested")
    logger.debug(
        "[structural-fb-reuse] shared_left_matmul_group fallback reason=%s "
        "provider=%s active_security=%s left_shape=%s group_size=%s",
        reason,
        getattr(crypten.mpc.get_default_provider(), "NAME", type(crypten.mpc.get_default_provider()).__name__),
        bool(getattr(cfg.mpc, "active_security", False)),
        tuple(x.size()) if hasattr(x, "size") else None,
        len(grouped_args),
    )
    results = [matmul(x.t() if t else x, y) for y, t in grouped_args]
    return _fallback_group_result(results, None, return_left_token)


def _normalize_reserved_spec(spec):
    """STRUCTURAL-FB-REUSE: normalize future right-shape reservations."""
    if isinstance(spec, dict):
        return {
            "right_size": torch.Size(spec["right_size"]),
            "transpose_left": bool(spec.get("transpose_left", False)),
        }
    right_size, transpose_left = spec
    return {"right_size": torch.Size(right_size), "transpose_left": bool(transpose_left)}


def _make_left_token(a, epsilon, reserved):
    """STRUCTURAL-FB-REUSE: token passed from LoRA forward to backward.

    The token stores the shared left mask [a], opened epsilon = x - a, and
    pre-generated future (b, c) triples. We pre-generate future triples in
    forward because the pasted provider can create many triples with one shared
    left mask, but it does not expose an API to create new c = a @ b later.
    """
    return {
        "a": a,
        "epsilon": epsilon,
        "reserved": reserved,
    }


def _pop_reserved_triple(left_token, right_size, transpose_left):
    if not isinstance(left_token, dict):
        return None
    right_size = torch.Size(right_size)
    reserved = left_token.get("reserved") or []
    for idx, item in enumerate(reserved):
        if item["right_size"] == right_size and item["transpose_left"] == transpose_left:
            return reserved.pop(idx)
    return None


def _beaver_matmul_from_components(left_a, left_eps, y, b, c):
    from .arithmetic import ArithmeticSharedTensor

    with IgnoreEncodings([b, y]):
        delta = ArithmeticSharedTensor.reveal(y - b)

    c._tensor += torch.matmul(left_eps, b._tensor)
    c._tensor += torch.matmul(left_a._tensor, delta)
    c += torch.matmul(left_eps, delta)
    return c


def __beaver_protocol(op, x, y, *args, **kwargs):
    """Performs Beaver protocol for additively secret-shared tensors x and y."""
    assert op in {
        "mul",
        "matmul",
        "conv1d",
        "conv2d",
        "conv_transpose1d",
        "conv_transpose2d",
    }
    if x.device != y.device:
        raise ValueError(f"x lives on device {x.device} but y on device {y.device}")

    provider = crypten.mpc.get_default_provider()
    a, b, c = provider.generate_additive_triple(
        x.size(), y.size(), op, device=x.device, *args, **kwargs
    )

    from .arithmetic import ArithmeticSharedTensor

    if cfg.mpc.active_security:
        f, g, h = provider.generate_additive_triple(
            x.size(), y.size(), op, device=x.device, *args, **kwargs
        )

        t = ArithmeticSharedTensor.PRSS(a.size(), device=x.device)
        t_plain_text = t.get_plain_text()

        rho = (t_plain_text * a - f).get_plain_text()
        sigma = (b - g).get_plain_text()
        triples_check = t_plain_text * c - h - sigma * f - rho * g - rho * sigma
        triples_check = triples_check.get_plain_text()

        if torch.any(triples_check != 0):
            raise ValueError("Beaver Triples verification failed!")

    with IgnoreEncodings([a, b, x, y]):
        epsilon, delta = ArithmeticSharedTensor.reveal_batch([x - a, y - b])

    c._tensor += getattr(torch, op)(epsilon, b._tensor, *args, **kwargs)
    c._tensor += getattr(torch, op)(a._tensor, delta, *args, **kwargs)
    c += getattr(torch, op)(epsilon, delta, *args, **kwargs)

    return c


def mul(x, y):
    return __beaver_protocol("mul", x, y)


def matmul(x, y):
    return __beaver_protocol("matmul", x, y)


def shared_left_matmul_group(
    x,
    grouped_args,
    return_left_token=False,
    reserve_right_specs=None,
):
    """Computes a group of matmuls that share the same left operand reveal.

    STRUCTURAL-FB-REUSE: if return_left_token=True, also returns a token that
    can later compute x.T @ grad without reopening x.T - a.T. reserve_right_specs
    pre-generates future (b, c) triples for those backward right shapes.
    """
    if reserve_right_specs is None:
        reserve_right_specs = []

    _inc_stat("same_left_group_calls")
    if return_left_token:
        _inc_stat("left_token_requested")

    if cfg.mpc.active_security:
        return _fallback_group("fallback_active_security", x, grouped_args, return_left_token)

    if x.dim() != 2:
        return _fallback_group("fallback_non_2d", x, grouped_args, return_left_token)

    if any(y.dim() != 2 for y, _ in grouped_args):
        return _fallback_group("fallback_non_2d", x, grouped_args, return_left_token)

    provider = crypten.mpc.get_default_provider()
    if not hasattr(provider, "generate_additive_triples_with_same_left"):
        return _fallback_group(
            "fallback_missing_provider_api", x, grouped_args, return_left_token
        )

    right_specs = []
    for y, transpose_left in grouped_args:
        if x.device != y.device:
            raise ValueError(f"x lives on device {x.device} but y on device {y.device}")
        right_specs.append({"right_size": y.size(), "transpose_left": transpose_left})

    normalized_reserved_specs = [_normalize_reserved_spec(spec) for spec in reserve_right_specs]
    right_specs.extend(normalized_reserved_specs)

    try:
        a, triples = provider.generate_additive_triples_with_same_left(
            x.size(), right_specs, device=x.device
        )
    except NotImplementedError:
        return _fallback_group(
            "fallback_provider_not_implemented", x, grouped_args, return_left_token
        )

    actual_count = len(grouped_args)
    actual_triples = triples[:actual_count]
    reserved_triples = triples[actual_count:]
    bs = [b for b, _ in actual_triples]
    cs = [c for _, c in actual_triples]

    from .arithmetic import ArithmeticSharedTensor

    with IgnoreEncodings([a, x]):
        epsilon = ArithmeticSharedTensor.reveal(x - a)

    with IgnoreEncodings(bs + [y for y, _ in grouped_args]):
        deltas = ArithmeticSharedTensor.reveal_batch(
            [y - b for (y, _), b in zip(grouped_args, bs)]
        )
    if not isinstance(deltas, (list, tuple)):
        deltas = [deltas]

    results = []
    for (_, transpose_left), b, c, delta in zip(grouped_args, bs, cs, deltas):
        left_a = a.t() if transpose_left else a
        left_eps = epsilon.t() if transpose_left else epsilon
        c._tensor += torch.matmul(left_eps, b._tensor)
        c._tensor += torch.matmul(left_a._tensor, delta)
        c += torch.matmul(left_eps, delta)
        results.append(c)

    _inc_stat("same_left_group_success")
    _inc_stat("same_left_group_outputs", len(results))
    token = None
    if return_left_token:
        reserved = []
        for spec, (b, c) in zip(normalized_reserved_specs, reserved_triples):
            reserved.append(
                {
                    "right_size": spec["right_size"],
                    "transpose_left": spec["transpose_left"],
                    "b": b,
                    "c": c,
                }
            )
        _inc_stat("left_token_created")
        _inc_stat("left_token_reserved_triples", len(reserved))
        token = _make_left_token(a, epsilon, reserved)

    return _fallback_group_result(results, token, return_left_token)


def matmul_with_left_token(x, y, left_token, transpose_left=False):
    """STRUCTURAL-FB-REUSE: computes x @ y using a forward left token.

    If transpose_left=True, computes x.T @ y using token.a.T and token.epsilon.T.
    Falls back to regular Beaver matmul if no matching reserved triple exists.
    """
    _inc_stat("left_token_consume_calls")
    if cfg.mpc.active_security or x.dim() != 2 or y.dim() != 2:
        _inc_stat("left_token_consume_fallback")
        if cfg.mpc.active_security:
            _inc_stat("fallback_active_security")
        else:
            _inc_stat("fallback_non_2d")
        logger.debug(
            "[structural-fb-reuse] token consume fallback active_security=%s "
            "left_shape=%s right_shape=%s",
            bool(getattr(cfg.mpc, "active_security", False)),
            tuple(x.size()) if hasattr(x, "size") else None,
            tuple(y.size()) if hasattr(y, "size") else None,
        )
        return matmul(x.t() if transpose_left else x, y)

    if x.device != y.device:
        raise ValueError(f"x lives on device {x.device} but y on device {y.device}")

    reserved = _pop_reserved_triple(left_token, y.size(), transpose_left)
    if reserved is None:
        _inc_stat("left_token_consume_fallback")
        _inc_stat("fallback_missing_reserved_triple")
        logger.debug(
            "[structural-fb-reuse] token consume fallback missing_reserved "
            "right_shape=%s transpose_left=%s token_present=%s",
            tuple(y.size()),
            transpose_left,
            isinstance(left_token, dict),
        )
        return matmul(x.t() if transpose_left else x, y)

    a = left_token["a"]
    epsilon = left_token["epsilon"]
    left_a = a.t() if transpose_left else a
    left_eps = epsilon.t() if transpose_left else epsilon
    _inc_stat("left_token_consume_success")
    return _beaver_matmul_from_components(left_a, left_eps, y, reserved["b"], reserved["c"])


def conv1d(x, y, **kwargs):
    return __beaver_protocol("conv1d", x, y, **kwargs)


def conv2d(x, y, **kwargs):
    return __beaver_protocol("conv2d", x, y, **kwargs)


def conv_transpose1d(x, y, **kwargs):
    return __beaver_protocol("conv_transpose1d", x, y, **kwargs)


def conv_transpose2d(x, y, **kwargs):
    return __beaver_protocol("conv_transpose2d", x, y, **kwargs)


def square(x):
    provider = crypten.mpc.get_default_provider()
    r, r2 = provider.square(x.size(), device=x.device)

    with IgnoreEncodings([x, r]):
        epsilon = (x - r).reveal()
    return r2 + 2 * r * epsilon + epsilon * epsilon


def wraps(x):
    provider = crypten.mpc.get_default_provider()
    r, theta_r = provider.wrap_rng(x.size(), device=x.device)
    beta_xr = theta_r.clone()
    beta_xr._tensor = count_wraps([x._tensor, r._tensor])

    with IgnoreEncodings([x, r]):
        z = x + r
    theta_z = comm.get().gather(z._tensor, 0)
    theta_x = beta_xr - theta_r

    if x.rank == 0:
        theta_z = count_wraps(theta_z)
        theta_x._tensor += theta_z
    return theta_x


def truncate(x, y):
    wrap_count = wraps(x)
    x.share = x.share.div_(y, rounding_mode="trunc")
    correction = wrap_count * 4 * (int(2**62) // y)
    x.share -= correction.share
    return x


def AND(x, y):
    from .binary import BinarySharedTensor

    provider = crypten.mpc.get_default_provider()
    a, b, c = provider.generate_binary_triple(x.size(), y.size(), device=x.device)

    eps_del = BinarySharedTensor.reveal_batch([x ^ a, y ^ b])
    epsilon = eps_del[0]
    delta = eps_del[1]

    return (b & epsilon) ^ (a & delta) ^ (epsilon & delta) ^ c


def B2A_single_bit(xB):
    if comm.get().get_world_size() < 2:
        from .arithmetic import ArithmeticSharedTensor

        return ArithmeticSharedTensor(xB._tensor, precision=0, src=0)

    provider = crypten.mpc.get_default_provider()
    rA, rB = provider.B2A_rng(xB.size(), device=xB.device)

    z = (xB ^ rB).reveal()
    rA = rA * (1 - 2 * z) + z
    return rA
