#!/usr/bin/env python3

"""
LoRA helpers for the shared-left-v1.2 MPC experiments.

This file is intentionally lightweight:
- it reuses the existing training / CrypTen runtime in
  `run_glue_private_mpc_lora_train.py`
- it only replaces the LoRA-wrapped linear layer with a structurally clearer
  split between the frozen backbone path and the trainable LoRA path
"""

import math
from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F


_ATTENTION_QKV_KEYWORDS = ("query", "key", "value")


def _normalize_target_keywords(target_keywords: Iterable[str]) -> List[str]:
    if isinstance(target_keywords, str):
        target_keywords = target_keywords.split(",")
    return [keyword.strip() for keyword in target_keywords if str(keyword).strip()]


def _matches_target(full_name: str, target_keywords: Iterable[str]) -> bool:
    full_name_lower = full_name.lower()
    return any(keyword.lower() in full_name_lower for keyword in target_keywords)


def _is_attention_qkv_name(module_name: str) -> bool:
    module_name_lower = module_name.lower()
    return any(keyword in module_name_lower for keyword in _ATTENTION_QKV_KEYWORDS)


class SharedLeftLoRAPath(nn.Module):
    """
    Trainable low-rank branch that explicitly exposes the `xA` -> `(xA)B` path.

    For attention Q / K / V, the same left activation X fans out into multiple
    sibling LoRA-A projections. That `xA` stage is the main shared-left target:

        Q_lora = (X A_Q) B_Q
        K_lora = (X A_K) B_K
        V_lora = (X A_V) B_V

    We keep the branch explicit so the code and the design document can point to
    the exact place where the repeated left operand appears.
    """

    def __init__(self, in_features: int, out_features: int, r: int, alpha: int, dropout: float):
        super().__init__()
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r) if r > 0 else 0.0
        self.dropout_p = float(dropout)

        if self.r > 0:
            self.lora_A = nn.Linear(in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    @property
    def enabled(self) -> bool:
        return self.r > 0 and self.lora_A is not None and self.lora_B is not None

    def forward(self, x):
        if not self.enabled:
            raise RuntimeError("SharedLeftLoRAPath.forward() called while LoRA is disabled.")

        x_lora = F.dropout(x, p=self.dropout_p, training=self.training) if self.dropout_p > 0 else x

        # Shared-left reuse should target this repeated left operand first when
        # attention Q / K / V branches consume the same hidden-state tensor X.
        xa = self.lora_A(x_lora)
        return self.lora_B(xa) * self.scaling


class SharedLeftSplitLoRALinear(nn.Module):
    """
    Explicit split:

        output = backbone_path + lora_path
               = xW + (xA)B

    The frozen backbone is kept structurally separate even though the current
    CrypTen runtime still encrypts the whole converted model via `.encrypt()`.
    That separation is deliberate: it makes the intended future
    `secret activation x public frozen weight` fast path visible in code now.
    """

    def __init__(self, base_layer, r=8, alpha=16, dropout=0.0, module_name=""):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"SharedLeftSplitLoRALinear expects nn.Linear, got {type(base_layer).__name__}"
            )

        self.module_name = str(module_name)
        self.is_attention_qkv = _is_attention_qkv_name(self.module_name)
        self.shared_left_hint = "attention_qkv_lora_A" if self.is_attention_qkv else "generic_lora_A"

        self.backbone_path = base_layer
        for parameter in self.backbone_path.parameters():
            parameter.requires_grad = False

        self.lora_path = SharedLeftLoRAPath(
            in_features=self.backbone_path.in_features,
            out_features=self.backbone_path.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x):
        backbone_path = self.backbone_path(x)
        if not self.lora_path.enabled:
            return backbone_path

        lora_path = self.lora_path(x)
        return backbone_path + lora_path


def inject_shared_left_lora_layers(module, target_keywords, r, alpha, dropout, prefix=""):
    """
    Minimal-intrusion LoRA injection helper used by the new shared-left wrapper
    script. It preserves the existing repository's replacement strategy
    (recursive `nn.Linear` substitution) while swapping in the explicit split
    module above.
    """

    normalized_keywords = _normalize_target_keywords(target_keywords)
    replaced = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and _matches_target(full_name, normalized_keywords):
            wrapped = SharedLeftSplitLoRALinear(
                child,
                r=r,
                alpha=alpha,
                dropout=dropout,
                module_name=full_name,
            )
            setattr(module, child_name, wrapped)
            scope = "attention-qkv" if wrapped.is_attention_qkv else "other"
            replaced.append(f"{full_name} [{scope}:{wrapped.shared_left_hint}]")
            continue

        replaced.extend(
            inject_shared_left_lora_layers(
                child,
                target_keywords=normalized_keywords,
                r=r,
                alpha=alpha,
                dropout=dropout,
                prefix=full_name,
            )
        )
    return replaced

