#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import crypten.communicator as comm
import torch
from crypten.common.reuse_context import get_current_beaver_tag
from crypten.common.util import count_wraps
from crypten.config import cfg
from crypten.mpc.primitives.beaver_reuse import (
    get_beaver_reuse_cache,
    get_perf_counters,
    increment_perf_counter,
    reset_perf_counters,
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


def _reuse_enabled_for_op(op):
    if not getattr(cfg.mpc, "experimental_reuse_mask", False):
        return False
    reuse_ops = getattr(cfg.mpc, "reuse_op_types", ["matmul"])
    if isinstance(reuse_ops, str):
        reuse_ops = [reuse_ops]
    return op in set(str(item) for item in reuse_ops)


def _normalize_beaver_tag(op, x, y):
    if not getattr(cfg.mpc, "reuse_tagging", True):
        return None
    tag = get_current_beaver_tag()
    if isinstance(tag, dict):
        return tag
    return {
        "step_id": None,
        "layer_id": "untagged",
        "op_name": op,
        "pass_name": "untagged",
        "op_uid": id(x) ^ id(y),
        "tensor_shapes_signature": (tuple(x.size()), tuple(y.size())),
    }


def _open_missing_residuals(x, y, a, b, epsilon, delta):
    from .arithmetic import ArithmeticSharedTensor

    missing = []
    names = []
    if epsilon is None:
        missing.append(x - a)
        names.append("epsilon")
    if delta is None:
        missing.append(y - b)
        names.append("delta")

    if len(missing) == 0:
        return epsilon, delta

    with IgnoreEncodings([a, b, x, y]):
        opened = ArithmeticSharedTensor.reveal_batch(missing)
    increment_perf_counter("beaver_reveal_calls")
    increment_perf_counter("beaver_revealed_tensors", len(missing))
    if not isinstance(opened, (list, tuple)):
        opened = [opened]

    for idx, name in enumerate(names):
        if name == "epsilon":
            epsilon = opened[idx]
        else:
            delta = opened[idx]
    return epsilon, delta


def _beaver_from_provider(op, x, y, *args, **kwargs):
    from .arithmetic import ArithmeticSharedTensor

    provider = crypten.mpc.get_default_provider()
    a, b, c = provider.generate_additive_triple(
        x.size(), y.size(), op, device=x.device, *args, **kwargs
    )

    if cfg.mpc.active_security:
        """
        Reference: "Multiparty Computation from Somewhat Homomorphic Encryption"
        Link: https://eprint.iacr.org/2011/535.pdf
        """
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
    increment_perf_counter("beaver_reveal_calls")
    increment_perf_counter("beaver_revealed_tensors", 2)

    c._tensor += getattr(torch, op)(epsilon, b._tensor, *args, **kwargs)
    c._tensor += getattr(torch, op)(a._tensor, delta, *args, **kwargs)
    c += getattr(torch, op)(epsilon, delta, *args, **kwargs)
    return c


def _beaver_with_reuse(op, x, y, *args, **kwargs):
    tag = _normalize_beaver_tag(op, x, y)
    reuse_cache = get_beaver_reuse_cache()
    reuse_mode = str(getattr(cfg.mpc, "reuse_mode", "FIX_A")).upper()

    a = reuse_cache.get_or_create_A(x.size(), x.share.dtype, x.device, tag)
    if reuse_mode == "FIX_AB":
        b = reuse_cache.get_or_create_B(y.size(), y.share.dtype, y.device, tag)
        cache_c = True
    elif reuse_mode == "FIX_A":
        if isinstance(tag, dict) and tag.get("b_anchor") is not None:
            b = reuse_cache.get_or_create_B(y.size(), y.share.dtype, y.device, tag)
            cache_c = True
        else:
            b = reuse_cache.create_fresh_B(y.size(), y.share.dtype, y.device)
            cache_c = False
    else:
        raise ValueError(f"Unknown cfg.mpc.reuse_mode `{reuse_mode}`")

    c = reuse_cache.get_or_create_C_for_op(
        a, b, op, args=args, kwargs=kwargs, tag=tag, cache_result=cache_c
    )

    epsilon = None
    delta = None
    if isinstance(tag, dict):
        epsilon, delta = reuse_cache.get_opened_residual(tag)
        if epsilon is None:
            epsilon = reuse_cache.get_opened_residual_from_anchor(
                tag, residual_name="epsilon", expected_shape=x.size()
            )
        if delta is None:
            delta = reuse_cache.get_opened_residual_from_anchor(
                tag, residual_name="delta", expected_shape=y.size()
            )

    epsilon, delta = _open_missing_residuals(x, y, a, b, epsilon, delta)

    if isinstance(tag, dict):
        reuse_cache.cache_opened_residual(tag, E_pub=epsilon, F_pub=delta)

    c._tensor += getattr(torch, op)(epsilon, b._tensor, *args, **kwargs)
    c._tensor += getattr(torch, op)(a._tensor, delta, *args, **kwargs)
    c += getattr(torch, op)(epsilon, delta, *args, **kwargs)
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

    if _reuse_enabled_for_op(op):
        return _beaver_with_reuse(op, x, y, *args, **kwargs)
    return _beaver_from_provider(op, x, y, *args, **kwargs)


def mul(x, y):
    return __beaver_protocol("mul", x, y)


def matmul(x, y):
    return __beaver_protocol("matmul", x, y)


def conv1d(x, y, **kwargs):
    return __beaver_protocol("conv1d", x, y, **kwargs)


def conv2d(x, y, **kwargs):
    return __beaver_protocol("conv2d", x, y, **kwargs)


def conv_transpose1d(x, y, **kwargs):
    return __beaver_protocol("conv_transpose1d", x, y, **kwargs)


def conv_transpose2d(x, y, **kwargs):
    return __beaver_protocol("conv_transpose2d", x, y, **kwargs)


def begin_reuse_step(step_id):
    get_beaver_reuse_cache().begin_step(step_id)


def end_reuse_step(step_id=None):
    get_beaver_reuse_cache().end_step(step_id)


def reset_reuse_stats(reset_cache=False):
    reset_perf_counters()
    if reset_cache:
        get_beaver_reuse_cache().clear()


def get_reuse_stats():
    return get_perf_counters()


def square(x):
    """Computes the square of `x` for additively secret-shared tensor `x`

    1. Obtain uniformly random sharings [r] and [r2] = [r * r]
    2. Additively hide [x] with appropriately sized [r]
    3. Open ([epsilon] = [x] - [r])
    4. Return z = [r2] + 2 * epsilon * [r] + epsilon ** 2
    """
    provider = crypten.mpc.get_default_provider()
    r, r2 = provider.square(x.size(), device=x.device)

    with IgnoreEncodings([x, r]):
        epsilon = (x - r).reveal()
    return r2 + 2 * r * epsilon + epsilon * epsilon


def wraps(x):
    """Privately computes the number of wraparounds for a set a shares

    To do so, we note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where [theta_i] is the wraps for a variable i
          [beta_ij] is the differential wraps for variables i and j
          [eta_ij]  is the plaintext wraps for variables i and j

    Note: Since [eta_xr] = 0 with probability 1 - |x| / Q for modulus Q, we
    can make the assumption that [eta_xr] = 0 with high probability.
    """
    provider = crypten.mpc.get_default_provider()
    r, theta_r = provider.wrap_rng(x.size(), device=x.device)
    beta_xr = theta_r.clone()
    beta_xr._tensor = count_wraps([x._tensor, r._tensor])

    with IgnoreEncodings([x, r]):
        z = x + r
    theta_z = comm.get().gather(z._tensor, 0)
    theta_x = beta_xr - theta_r

    # TODO: Incorporate eta_xr
    if x.rank == 0:
        theta_z = count_wraps(theta_z)
        theta_x._tensor += theta_z
    return theta_x


def truncate(x, y):
    """Protocol to divide an ArithmeticSharedTensor `x` by a constant integer `y`"""
    wrap_count = wraps(x)
    x.share = x.share.div_(y, rounding_mode="trunc")
    # NOTE: The multiplication here must be split into two parts
    # to avoid long out-of-bounds when y <= 2 since (2 ** 63) is
    # larger than the largest long integer.
    correction = wrap_count * 4 * (int(2**62) // y)
    x.share -= correction.share
    return x


def AND(x, y):
    """
    Performs Beaver protocol for binary secret-shared tensors x and y

    1. Obtain uniformly random sharings [a],[b] and [c] = [a & b]
    2. XOR hide [x] and [y] with appropriately sized [a] and [b]
    3. Open ([epsilon] = [x] ^ [a]) and ([delta] = [y] ^ [b])
    4. Return [c] ^ (epsilon & [b]) ^ ([a] & delta) ^ (epsilon & delta)
    """
    from .binary import BinarySharedTensor

    provider = crypten.mpc.get_default_provider()
    a, b, c = provider.generate_binary_triple(x.size(), y.size(), device=x.device)

    # Stack to vectorize reveal
    eps_del = BinarySharedTensor.reveal_batch([x ^ a, y ^ b])
    epsilon = eps_del[0]
    delta = eps_del[1]

    return (b & epsilon) ^ (a & delta) ^ (epsilon & delta) ^ c


def B2A_single_bit(xB):
    """Converts a single-bit BinarySharedTensor xB into an
        ArithmeticSharedTensor. This is done by:

    1. Generate ArithmeticSharedTensor [rA] and BinarySharedTensor =rB= with
        a common 1-bit value r.
    2. Hide xB with rB and open xB ^ rB
    3. If xB ^ rB = 0, then return [rA], otherwise return 1 - [rA]
        Note: This is an arithmetic xor of a single bit.
    """
    if comm.get().get_world_size() < 2:
        from .arithmetic import ArithmeticSharedTensor

        return ArithmeticSharedTensor(xB._tensor, precision=0, src=0)

    provider = crypten.mpc.get_default_provider()
    rA, rB = provider.B2A_rng(xB.size(), device=xB.device)

    z = (xB ^ rB).reveal()
    rA = rA * (1 - 2 * z) + z
    return rA
