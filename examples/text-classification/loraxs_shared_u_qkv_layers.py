#!/usr/bin/env python3

"""
Shared-U qkv LoRA-XS layers for newSHAFT private finetuning.

Design goal:
- for each attention layer, compute a shared low-rank public projection once:
      T = X U
- then let q / k / v each consume that same left operand via its own private
  latent mapping:
      Q_lora = (T G_q) B_q
      K_lora = (T G_k) B_k
      V_lora = (T G_v) B_v

This keeps the baseline LoRA-XS path available elsewhere while providing a
clean, separate implementation for the shared-U qkv experiment line.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from loraxs_public_layers import LoRAXSPublicLinear


_QKV_BRANCHES = ("query", "key", "value")


def _normalize_target_keywords(target_keywords: Iterable[str]) -> List[str]:
    if isinstance(target_keywords, str):
        target_keywords = target_keywords.split(",")
    return [keyword.strip() for keyword in target_keywords if str(keyword).strip()]


def _matches_target(full_name: str, target_keywords: Iterable[str]) -> bool:
    full_name_lower = full_name.lower()
    return any(keyword.lower() in full_name_lower for keyword in target_keywords)


def _compute_shared_encoder_and_branch_decoders(base_layers: Sequence[nn.Linear], rank: int):
    requested_rank = int(rank)
    if requested_rank <= 0 or len(base_layers) == 0:
        return None, None, 0

    weight_ts = [
        layer.weight.detach().to(dtype=torch.float32, device="cpu").T for layer in base_layers
    ]
    concat_weight_t = torch.cat(weight_ts, dim=1)
    max_rank = min(requested_rank, concat_weight_t.shape[0], concat_weight_t.shape[1])
    if max_rank <= 0:
        return None, None, 0

    u, s, vh = torch.linalg.svd(concat_weight_t, full_matrices=False)
    u_r = u[:, :max_rank]
    s_r = s[:max_rank]
    vh_r = vh[:max_rank, :]

    first_weight = base_layers[0].weight
    enc = (u_r * s_r.unsqueeze(0)).to(dtype=first_weight.dtype, device=first_weight.device)

    branch_decoders = []
    offset = 0
    for weight_t, base_layer in zip(weight_ts, base_layers):
        width = int(weight_t.shape[1])
        dec = vh_r[:, offset : offset + width]
        offset += width
        branch_decoders.append(dec.to(dtype=base_layer.weight.dtype, device=base_layer.weight.device))

    return enc, branch_decoders, max_rank


class SharedQKVEncoderBundle(nn.Module):
    def __init__(self, encoder_weight: torch.Tensor, dropout: float, expected_consumers: int):
        super().__init__()
        in_features = int(encoder_weight.size(0))
        out_features = int(encoder_weight.size(1))
        self.dropout_p = float(dropout)
        self.expected_consumers = max(1, int(expected_consumers))

        self.lora_A = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            self.lora_A.weight.copy_(encoder_weight.T)
        self.lora_A.weight.requires_grad = False

        self._cached_input_ref = None
        self._cached_projection = None
        self._cache_hits = 0

    def _clear_cache(self):
        self._cached_input_ref = None
        self._cached_projection = None
        self._cache_hits = 0

    def train(self, mode: bool = True):
        self._clear_cache()
        return super().train(mode)

    def project(self, x):
        if self._cached_input_ref is x and self._cached_projection is not None:
            projection = self._cached_projection
            self._cache_hits += 1
            if self._cache_hits >= self.expected_consumers:
                self._clear_cache()
            return projection

        x_lora = F.dropout(x, p=self.dropout_p, training=self.training) if self.dropout_p > 0 else x
        projection = self.lora_A(x_lora)
        self._cached_input_ref = x
        self._cached_projection = projection
        self._cache_hits = 1
        if self.expected_consumers <= 1:
            self._clear_cache()
        return projection


class SharedUQKVLoRAXSPath(nn.Module):
    def __init__(
        self,
        shared_bundle: SharedQKVEncoderBundle,
        branch_decoder: torch.Tensor,
        r: int,
        alpha: int,
        branch_name: str,
    ):
        super().__init__()
        self.shared_bundle = shared_bundle
        self.branch_name = str(branch_name)
        self.requested_r = int(r)
        self.r = int(branch_decoder.size(0))
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(self.r) if self.r > 0 else 0.0

        if self.r > 0:
            self.lora_latent = nn.Linear(self.r, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, int(branch_decoder.size(1)), bias=False)
            with torch.no_grad():
                self.lora_B.weight.copy_(branch_decoder.T)
            self.lora_B.weight.requires_grad = False
            nn.init.normal_(self.lora_latent.weight, mean=0.0, std=1e-5)
        else:
            self.lora_latent = None
            self.lora_B = None

    @property
    def enabled(self) -> bool:
        return self.r > 0 and self.lora_latent is not None and self.lora_B is not None

    def forward(self, x):
        if not self.enabled:
            raise RuntimeError("SharedUQKVLoRAXSPath.forward() called while LoRA-XS is disabled.")

        shared_projection = self.shared_bundle.project(x)
        latent_projection = self.lora_latent(shared_projection)
        return self.lora_B(latent_projection) * self.scaling


class SharedUQKVLoRAXSPublicLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        shared_bundle: SharedQKVEncoderBundle,
        branch_decoder: torch.Tensor,
        r: int = 8,
        alpha: int = 16,
        module_name: str = "",
        branch_name: str = "",
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"SharedUQKVLoRAXSPublicLinear expects nn.Linear, got {type(base_layer).__name__}"
            )

        self.module_name = str(module_name)
        self.branch_name = str(branch_name)
        self.backbone_path = base_layer
        for parameter in self.backbone_path.parameters():
            parameter.requires_grad = False

        self.lora_path = SharedUQKVLoRAXSPath(
            shared_bundle=shared_bundle,
            branch_decoder=branch_decoder,
            r=r,
            alpha=alpha,
            branch_name=branch_name,
        )

    def forward(self, x):
        backbone_path = self.backbone_path(x)
        if not self.lora_path.enabled:
            return backbone_path
        return backbone_path + self.lora_path(x)


def _can_share_qkv(branch_layers: Sequence[nn.Linear]) -> bool:
    if len(branch_layers) < 2:
        return False
    first = branch_layers[0]
    return all(
        isinstance(layer, nn.Linear)
        and layer.in_features == first.in_features
        and layer.out_features == first.out_features
        for layer in branch_layers
    )


def inject_shared_u_qkv_loraxs_layers(module, target_keywords, r, alpha, dropout, prefix=""):
    normalized_keywords = _normalize_target_keywords(target_keywords)
    replaced = []

    child_items = list(module.named_children())
    child_map = dict(child_items)

    targeted_qkv = []
    for branch_name in _QKV_BRANCHES:
        child = child_map.get(branch_name)
        full_name = f"{prefix}.{branch_name}" if prefix else branch_name
        if isinstance(child, nn.Linear) and _matches_target(full_name, normalized_keywords):
            targeted_qkv.append((branch_name, full_name, child))

    replaced_branch_names = set()
    if len(targeted_qkv) >= 2 and _can_share_qkv([child for _, _, child in targeted_qkv]):
        enc, branch_decoders, effective_rank = _compute_shared_encoder_and_branch_decoders(
            [child for _, _, child in targeted_qkv],
            rank=r,
        )
        if effective_rank > 0 and enc is not None and branch_decoders is not None:
            shared_bundle = SharedQKVEncoderBundle(
                encoder_weight=enc,
                dropout=dropout,
                expected_consumers=len(targeted_qkv),
            )
            for (branch_name, full_name, child), branch_decoder in zip(targeted_qkv, branch_decoders):
                wrapped = SharedUQKVLoRAXSPublicLinear(
                    child,
                    shared_bundle=shared_bundle,
                    branch_decoder=branch_decoder,
                    r=effective_rank,
                    alpha=alpha,
                    module_name=full_name,
                    branch_name=branch_name,
                )
                setattr(module, branch_name, wrapped)
                replaced.append(f"{full_name} [shared_u_qkv]")
                replaced_branch_names.add(branch_name)

    for child_name, child in list(module.named_children()):
        if child_name in replaced_branch_names:
            continue

        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and _matches_target(full_name, normalized_keywords):
            wrapped = LoRAXSPublicLinear(
                child,
                r=r,
                alpha=alpha,
                dropout=dropout,
                module_name=full_name,
            )
            setattr(module, child_name, wrapped)
            replaced.append(full_name)
            continue

        replaced.extend(
            inject_shared_u_qkv_loraxs_layers(
                child,
                target_keywords=normalized_keywords,
                r=r,
                alpha=alpha,
                dropout=dropout,
                prefix=full_name,
            )
        )

    return replaced
