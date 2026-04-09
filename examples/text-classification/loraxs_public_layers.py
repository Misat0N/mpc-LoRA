#!/usr/bin/env python3

"""
LoRA-XS style layers for newSHAFT private finetuning.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _normalize_target_keywords(target_keywords):
    if isinstance(target_keywords, str):
        target_keywords = target_keywords.split(",")
    return [keyword.strip() for keyword in target_keywords if str(keyword).strip()]


def _matches_target(full_name, target_keywords):
    full_name_lower = full_name.lower()
    return any(keyword.lower() in full_name_lower for keyword in target_keywords)


def _compute_svd_encoder_decoder(weight: torch.Tensor, rank: int):
    requested_rank = int(rank)
    if requested_rank <= 0:
        return None, None, 0

    weight_t = weight.detach().to(dtype=torch.float32, device="cpu").T
    max_rank = min(requested_rank, weight_t.shape[0], weight_t.shape[1])
    if max_rank <= 0:
        return None, None, 0

    u, s, vh = torch.linalg.svd(weight_t, full_matrices=False)
    u_r = u[:, :max_rank]
    s_r = s[:max_rank]
    vh_r = vh[:max_rank, :]

    enc = u_r * s_r.unsqueeze(0)
    dec = vh_r
    return (
        enc.to(dtype=weight.dtype, device=weight.device),
        dec.to(dtype=weight.dtype, device=weight.device),
        max_rank,
    )


class LoRAXSPath(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.requested_r = int(r)
        self.alpha = int(alpha)
        self.dropout_p = float(dropout)

        enc, dec, effective_rank = _compute_svd_encoder_decoder(base_layer.weight, self.requested_r)
        self.r = int(effective_rank)
        self.scaling = float(alpha) / float(self.r) if self.r > 0 else 0.0

        if self.r > 0:
            self.lora_A = nn.Linear(base_layer.in_features, self.r, bias=False)
            self.lora_latent = nn.Linear(self.r, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, base_layer.out_features, bias=False)

            with torch.no_grad():
                self.lora_A.weight.copy_(enc.T)
                self.lora_B.weight.copy_(dec.T)

            self.lora_A.weight.requires_grad = False
            self.lora_B.weight.requires_grad = False
            nn.init.normal_(self.lora_latent.weight, mean=0.0, std=1e-5)
        else:
            self.lora_A = None
            self.lora_latent = None
            self.lora_B = None

    @property
    def enabled(self) -> bool:
        return (
            self.r > 0
            and self.lora_A is not None
            and self.lora_latent is not None
            and self.lora_B is not None
        )

    def forward(self, x):
        if not self.enabled:
            raise RuntimeError("LoRAXSPath.forward() called while LoRA-XS is disabled.")

        x_lora = F.dropout(x, p=self.dropout_p, training=self.training) if self.dropout_p > 0 else x
        xa = self.lora_A(x_lora)
        xr = self.lora_latent(xa)
        return self.lora_B(xr) * self.scaling


class LoRAXSPublicLinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.0, module_name=""):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRAXSPublicLinear expects nn.Linear, got {type(base_layer).__name__}")

        self.module_name = str(module_name)
        self.backbone_path = base_layer
        for parameter in self.backbone_path.parameters():
            parameter.requires_grad = False

        self.lora_path = LoRAXSPath(base_layer=self.backbone_path, r=r, alpha=alpha, dropout=dropout)

    def forward(self, x):
        backbone_path = self.backbone_path(x)
        if not self.lora_path.enabled:
            return backbone_path
        return backbone_path + self.lora_path(x)


def inject_loraxs_layers(module, target_keywords, r, alpha, dropout, prefix=""):
    normalized_keywords = _normalize_target_keywords(target_keywords)
    replaced = []
    for child_name, child in list(module.named_children()):
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
            inject_loraxs_layers(
                child,
                target_keywords=normalized_keywords,
                r=r,
                alpha=alpha,
                dropout=dropout,
                prefix=full_name,
            )
        )
    return replaced
