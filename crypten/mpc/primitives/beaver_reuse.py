#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import threading

import torch
from crypten.config import cfg


_PERF_COUNTERS_LOCK = threading.Lock()
_PERF_COUNTERS = {
    "triple_generate_calls": 0,
    "beaver_reveal_calls": 0,
    "beaver_revealed_tensors": 0,
    "a_cache_hit": 0,
    "a_cache_miss": 0,
    "b_cache_hit": 0,
    "b_cache_miss": 0,
    "b_fresh_generated": 0,
    "c_cache_hit": 0,
    "c_cache_miss": 0,
    "residual_cache_hit": 0,
    "residual_cache_miss": 0,
    "residual_anchor_hit": 0,
    "residual_anchor_miss": 0,
}


def increment_perf_counter(name, amount=1):
    with _PERF_COUNTERS_LOCK:
        _PERF_COUNTERS[name] = _PERF_COUNTERS.get(name, 0) + amount


def reset_perf_counters():
    with _PERF_COUNTERS_LOCK:
        for key in list(_PERF_COUNTERS.keys()):
            _PERF_COUNTERS[key] = 0


def get_perf_counters():
    with _PERF_COUNTERS_LOCK:
        return dict(_PERF_COUNTERS)


class _MaskEntry:
    __slots__ = ("plain", "shared")

    def __init__(self, plain, shared):
        self.plain = plain
        self.shared = shared


class _SharedMaskRegistry:
    __slots__ = ("_plain", "_cache_key")

    def __init__(self):
        self._plain = {}
        self._cache_key = {}

    def clear(self):
        self._plain.clear()
        self._cache_key.clear()

    def register(self, shared, plain, cache_key=None):
        shared_id = id(shared)
        self._plain[shared_id] = plain
        self._cache_key[shared_id] = cache_key

    def get_plain(self, shared):
        return self._plain.get(id(shared))

    def get_cache_key(self, shared):
        return self._cache_key.get(id(shared))


class BeaverReuseCache:
    """
    INSECURE PERFORMANCE EXPERIMENT cache for Beaver mask / residual reuse.
    """

    def __init__(self):
        self._active_step = None
        self._base_masks = {}
        self._derived_masks = {}
        self._c_cache = {}
        self._residual_cache = {}
        self._shared_registry = _SharedMaskRegistry()

    @staticmethod
    def _normalize_shape(shape):
        return tuple(int(dim) for dim in shape)

    @staticmethod
    def _normalize_device(device):
        if device is None:
            return torch.device("cpu")
        if isinstance(device, str):
            return torch.device(device)
        return device

    @staticmethod
    def _normalize_tag(tag):
        if not isinstance(tag, dict):
            tag = {}
        return {
            "step_id": tag.get("step_id"),
            "layer_id": tag.get("layer_id", "global"),
            "op_name": tag.get("op_name", "matmul"),
            "pass_name": tag.get("pass_name", "untagged"),
            "op_uid": tag.get("op_uid", -1),
            "tensor_shapes_signature": tag.get("tensor_shapes_signature"),
            "a_anchor": tag.get("a_anchor"),
            "b_anchor": tag.get("b_anchor"),
            "epsilon_anchor": tag.get("epsilon_anchor"),
            "delta_anchor": tag.get("delta_anchor"),
        }

    @staticmethod
    def _step_scope_enabled():
        return str(getattr(cfg.mpc, "reuse_scope", "STEP")).upper() == "STEP"

    def clear(self):
        self._base_masks.clear()
        self._derived_masks.clear()
        self._c_cache.clear()
        self._residual_cache.clear()
        self._shared_registry.clear()

    def begin_step(self, step_id):
        if self._active_step != step_id:
            self.clear()
            self._active_step = step_id

    def end_step(self, step_id=None):
        if step_id is None or step_id == self._active_step:
            self.clear()
            self._active_step = None

    def _activate_step_from_tag(self, tag):
        if not self._step_scope_enabled():
            return
        step_id = tag.get("step_id")
        if step_id is None:
            return
        self.begin_step(step_id)

    @staticmethod
    def _base_descriptor(tag, operand, device, pass_name=None):
        return (
            "mask_base",
            tag.get("step_id"),
            tag.get("layer_id"),
            tag.get("op_uid"),
            pass_name if pass_name is not None else tag.get("pass_name"),
            operand,
            str(device),
        )

    @staticmethod
    def _normalize_anchor(anchor, tag, default_operand):
        if not isinstance(anchor, dict):
            return None
        return {
            "step_id": anchor.get("step_id", tag.get("step_id")),
            "layer_id": anchor.get("layer_id", tag.get("layer_id")),
            "op_uid": anchor.get("op_uid", tag.get("op_uid")),
            "pass_name": anchor.get("pass_name", "forward"),
            "operand": anchor.get("operand", default_operand),
            "residual": anchor.get("residual"),
            "transform": anchor.get("transform", "identity"),
        }

    @staticmethod
    def _infer_source_shape(target_shape, transform):
        if transform == "identity":
            return target_shape
        if transform == "transpose":
            if len(target_shape) < 2:
                return target_shape
            return target_shape[:-2] + (target_shape[-1], target_shape[-2])
        raise ValueError(f"Unsupported transform `{transform}`")

    @staticmethod
    def _apply_transform(tensor, transform):
        if transform == "identity":
            return tensor
        if transform == "transpose":
            if tensor.dim() < 2:
                return tensor
            return tensor.transpose(-2, -1).contiguous()
        raise ValueError(f"Unsupported transform `{transform}`")

    @staticmethod
    def _stable_signature(value):
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return tuple(BeaverReuseCache._stable_signature(v) for v in value)
        if isinstance(value, dict):
            return tuple(
                sorted((k, BeaverReuseCache._stable_signature(v)) for k, v in value.items())
            )
        return repr(value)

    def _random_plain_mask(self, shape, device):
        import crypten
        from crypten.common.rng import generate_random_ring_element

        device = self._normalize_device(device)
        generator = crypten.generators["global"][device]
        return generate_random_ring_element(shape, generator=generator, device=device)

    def _create_mask_entry(self, plain):
        from .arithmetic import ArithmeticSharedTensor

        shared = ArithmeticSharedTensor(plain, precision=0, src=0)
        return _MaskEntry(plain=plain, shared=shared)

    def _register_entry(self, entry, cache_key=None):
        self._shared_registry.register(entry.shared, entry.plain, cache_key=cache_key)

    def _create_and_register_base_mask(self, base_key, shape, device):
        plain = self._random_plain_mask(shape, device=device)
        entry = self._create_mask_entry(plain)
        self._base_masks[base_key] = entry
        identity_key = (base_key, "identity")
        self._derived_masks[identity_key] = entry
        self._register_entry(entry, cache_key=identity_key)
        return entry

    def _get_or_create_mask(self, operand, shape, device, tag):
        normalized_tag = self._normalize_tag(tag)
        self._activate_step_from_tag(normalized_tag)
        shape = self._normalize_shape(shape)
        device = self._normalize_device(device)
        anchor_name = f"{operand}_anchor"
        anchor = self._normalize_anchor(
            normalized_tag.get(anchor_name), normalized_tag, default_operand=operand
        )
        counter_prefix = "a" if operand == "a" else "b"

        if anchor is None:
            base_key = self._base_descriptor(normalized_tag, operand, device)
            if base_key in self._base_masks:
                increment_perf_counter(f"{counter_prefix}_cache_hit")
                entry = self._base_masks[base_key]
                self._register_entry(entry, cache_key=(base_key, "identity"))
                return entry.shared
            increment_perf_counter(f"{counter_prefix}_cache_miss")
            entry = self._create_and_register_base_mask(base_key, shape, device=device)
            return entry.shared

        source_tag = {
            "step_id": anchor["step_id"],
            "layer_id": anchor["layer_id"],
            "op_uid": anchor["op_uid"],
            "pass_name": anchor["pass_name"],
        }
        source_base_key = self._base_descriptor(
            source_tag, anchor["operand"], device, pass_name=anchor["pass_name"]
        )

        source_entry = self._base_masks.get(source_base_key)
        if source_entry is None:
            source_shape = self._infer_source_shape(shape, anchor["transform"])
            source_entry = self._create_and_register_base_mask(
                source_base_key, source_shape, device=device
            )

        derived_key = (source_base_key, anchor["transform"])
        entry = self._derived_masks.get(derived_key)
        if entry is not None:
            if entry.plain.size() == torch.Size(shape):
                increment_perf_counter(f"{counter_prefix}_cache_hit")
                self._register_entry(entry, cache_key=derived_key)
                return entry.shared

        transformed_plain = self._apply_transform(source_entry.plain, anchor["transform"])
        if transformed_plain.size() != torch.Size(shape):
            transformed_plain = self._random_plain_mask(shape, device=device)
        entry = self._create_mask_entry(transformed_plain)
        self._derived_masks[derived_key] = entry
        increment_perf_counter(f"{counter_prefix}_cache_miss")
        self._register_entry(entry, cache_key=derived_key)
        return entry.shared

    def get_or_create_A(self, shape, dtype, device, tag):
        del dtype
        return self._get_or_create_mask("a", shape, device, tag)

    def get_or_create_B(self, shape, dtype, device, tag):
        del dtype
        return self._get_or_create_mask("b", shape, device, tag)

    def create_fresh_B(self, shape, dtype, device):
        del dtype
        shape = self._normalize_shape(shape)
        plain = self._random_plain_mask(shape, device=device)
        entry = self._create_mask_entry(plain)
        increment_perf_counter("b_fresh_generated")
        self._register_entry(entry, cache_key=None)
        return entry.shared

    def get_or_create_C_for_op(
        self, A, B, op, args=(), kwargs=None, tag=None, cache_result=True
    ):
        del tag
        if kwargs is None:
            kwargs = {}
        a_plain = self._shared_registry.get_plain(A)
        b_plain = self._shared_registry.get_plain(B)
        if a_plain is None or b_plain is None:
            raise RuntimeError("Unable to locate plaintext masks for cached Beaver C.")

        a_cache_key = self._shared_registry.get_cache_key(A)
        b_cache_key = self._shared_registry.get_cache_key(B)
        c_key = (
            "mask_c",
            op,
            a_cache_key,
            b_cache_key,
            self._stable_signature(args),
            self._stable_signature(kwargs),
        )
        if cache_result and c_key in self._c_cache:
            increment_perf_counter("c_cache_hit")
            entry = self._c_cache[c_key]
            self._register_entry(entry, cache_key=c_key)
            return entry.shared

        increment_perf_counter("c_cache_miss")
        c_plain = getattr(torch, op)(a_plain, b_plain, *args, **kwargs)
        entry = self._create_mask_entry(c_plain)
        if cache_result:
            self._c_cache[c_key] = entry
        self._register_entry(entry, cache_key=c_key if cache_result else None)
        return entry.shared

    @staticmethod
    def _residual_key(tag, pass_name, residual_name):
        return (
            "residual",
            tag.get("step_id"),
            tag.get("layer_id"),
            tag.get("op_uid"),
            pass_name,
            residual_name,
        )

    def cache_opened_residual(self, tag, E_pub=None, F_pub=None):
        normalized_tag = self._normalize_tag(tag)
        if E_pub is not None:
            key = self._residual_key(
                normalized_tag, normalized_tag.get("pass_name"), "epsilon"
            )
            self._residual_cache[key] = E_pub
        if F_pub is not None:
            key = self._residual_key(
                normalized_tag, normalized_tag.get("pass_name"), "delta"
            )
            self._residual_cache[key] = F_pub

    def get_opened_residual(self, tag):
        normalized_tag = self._normalize_tag(tag)
        epsilon_key = self._residual_key(
            normalized_tag, normalized_tag.get("pass_name"), "epsilon"
        )
        delta_key = self._residual_key(normalized_tag, normalized_tag.get("pass_name"), "delta")
        epsilon = self._residual_cache.get(epsilon_key)
        delta = self._residual_cache.get(delta_key)
        if epsilon is not None:
            increment_perf_counter("residual_cache_hit")
        else:
            increment_perf_counter("residual_cache_miss")
        if delta is not None:
            increment_perf_counter("residual_cache_hit")
        else:
            increment_perf_counter("residual_cache_miss")
        return epsilon, delta

    def get_opened_residual_from_anchor(self, tag, residual_name, expected_shape):
        normalized_tag = self._normalize_tag(tag)
        anchor_key = f"{residual_name}_anchor"
        anchor = self._normalize_anchor(
            normalized_tag.get(anchor_key), normalized_tag, default_operand="a"
        )
        if anchor is None:
            increment_perf_counter("residual_anchor_miss")
            return None
        source_residual = anchor.get("residual") or residual_name
        source_key = self._residual_key(anchor, anchor["pass_name"], source_residual)
        source_value = self._residual_cache.get(source_key)
        if source_value is None:
            increment_perf_counter("residual_anchor_miss")
            return None
        transformed = self._apply_transform(source_value, anchor["transform"])
        if transformed.size() != torch.Size(self._normalize_shape(expected_shape)):
            increment_perf_counter("residual_anchor_miss")
            return None
        increment_perf_counter("residual_anchor_hit")
        return transformed


_BEAVER_REUSE_CACHE = BeaverReuseCache()


def get_beaver_reuse_cache():
    return _BEAVER_REUSE_CACHE
