#!/usr/bin/env python3

"""
Shared-left grouping helpers for CrypTen graphs.

Why this file exists:
- the original grouping logic only grouped sibling matmuls whose immediate
  left-input node name was identical
- in practice, ONNX export can insert per-branch alias / view-like nodes, so
  Q / K / V LoRA-A branches may still share the same semantic left operand but
  fail the naive string-equality test

This helper therefore does two passes:
1. explicit Q / K / V LoRA-A grouping, keyed by attention-layer prefix and a
   canonicalized left-input source
2. generic structural grouping fallback, now also using canonicalized left
   inputs instead of only the immediate input name
"""

from __future__ import annotations

import re
from collections import defaultdict

import crypten as ct
from crypten.nn import module as ct_module


_QKV_LORA_A_NODE_RE = re.compile(
    r"^(?P<prefix>.*?)/(?P<branch>query|key|value)/lora_path/lora_A/MatMul_output(?:_\d+)?$"
)

_CANONICAL_LEFT_ALIAS_MODULES = (
    ct_module.Identity,
    ct.nn.Reshape,
    ct.nn.Flatten,
    ct.nn.Squeeze,
    ct.nn.Unsqueeze,
    ct.nn.Cast,
    ct.nn.Dropout,
    ct.nn.Dropout2d,
    ct.nn.Dropout3d,
)


def _normalize_graph_name(name):
    return name if name else "root"


def _normalize_node_name(name):
    return str(name).replace("\\", "/")


def _is_shared_left_groupable_module(module):
    return isinstance(module, (ct.nn.Gemm, ct.nn.Linear, ct.nn.MatMul))


def _shared_left_operand_transform(module):
    if isinstance(module, ct.nn.Gemm):
        return "transpose" if getattr(module, "trans_a", False) else "identity"
    return "identity"


def _canonicalize_left_input_name(graph, input_name, max_hops=8):
    current_name = input_name
    visited = set()
    hops = 0
    while hops < max_hops and current_name not in visited:
        visited.add(current_name)
        module = graph._modules.get(current_name)
        if module is None or not isinstance(module, _CANONICAL_LEFT_ALIAS_MODULES):
            break

        parent_inputs = graph._graph.get(current_name, [])
        if len(parent_inputs) != 1:
            break

        parent_name = parent_inputs[0]
        parent_module = graph._modules.get(parent_name)
        if isinstance(parent_module, ct.nn.Parameter):
            break

        current_name = parent_name
        hops += 1
    return current_name


def _extract_qkv_lora_a_signature(node_name):
    match = _QKV_LORA_A_NODE_RE.match(_normalize_node_name(node_name))
    if match is None:
        return None
    prefix = match.group("prefix") or "root"
    branch = match.group("branch")
    return prefix, branch


def _annotate_explicit_qkv_lora_a_groups(graph, graph_prefix, summary, min_fanout, grouped_nodes):
    candidates_by_family = defaultdict(list)
    diagnostics_by_family = defaultdict(list)

    for node_name, input_names in graph._graph.items():
        module = graph._modules.get(node_name)
        if not isinstance(module, ct.nn.MatMul):
            continue
        if len(input_names) == 0:
            continue

        signature = _extract_qkv_lora_a_signature(node_name)
        if signature is None:
            continue

        family_prefix, branch = signature
        left_input_name = input_names[0]
        canonical_left_input = _canonicalize_left_input_name(graph, left_input_name)
        family_key = (family_prefix, canonical_left_input)
        candidates_by_family[family_key].append(
            (branch, node_name, module, left_input_name, canonical_left_input)
        )
        diagnostics_by_family[family_prefix].append(
            {
                "branch": branch,
                "node": node_name,
                "left_input": left_input_name,
                "canonical_left_input": canonical_left_input,
            }
        )

    for family_prefix, items in sorted(diagnostics_by_family.items()):
        summary["qkv_lora_a_diagnostics"].append(
            {
                "graph": graph_prefix,
                "family": family_prefix,
                "fanout_candidates": len(items),
                "branches": sorted(item["branch"] for item in items),
                "items": items,
            }
        )

    for (family_prefix, canonical_left_input), consumers in candidates_by_family.items():
        distinct_branches = sorted({branch for branch, *_ in consumers})
        if len(consumers) < min_fanout or len(distinct_branches) < min_fanout:
            continue

        group_tag = (
            f"shared_left:qkv_lora_a:{graph_prefix}:{family_prefix}:{canonical_left_input}"
        )
        group_nodes = []
        raw_left_inputs = []
        for branch, node_name, module, left_input_name, _ in consumers:
            setattr(module, "beaver_a_group", group_tag)
            setattr(module, "beaver_layer_tag", f"{graph_prefix}:{node_name}")
            grouped_nodes.add((graph_prefix, node_name))
            group_nodes.append(node_name)
            raw_left_inputs.append(left_input_name)

        summary["num_groups"] += 1
        summary["num_grouped_modules"] += len(consumers)
        summary["strategy_counts"]["explicit_qkv_lora_a"] += 1
        summary["groups"].append(
            {
                "graph": graph_prefix,
                "strategy": "explicit_qkv_lora_a",
                "family": family_prefix,
                "left_input": canonical_left_input,
                "left_transform": "identity",
                "fanout": len(consumers),
                "branches": distinct_branches,
                "raw_left_inputs": raw_left_inputs,
                "nodes": group_nodes,
            }
        )


def _annotate_generic_canonical_left_groups(graph, graph_prefix, summary, min_fanout, grouped_nodes):
    consumers_by_left_input = defaultdict(list)
    for node_name, input_names in graph._graph.items():
        if (graph_prefix, node_name) in grouped_nodes:
            continue

        module = graph._modules.get(node_name)
        if module is None or not _is_shared_left_groupable_module(module):
            continue
        if len(input_names) == 0:
            continue

        left_input_name = input_names[0]
        left_input_module = graph._modules.get(left_input_name)
        if isinstance(left_input_module, ct.nn.Parameter):
            continue

        canonical_left_input = _canonicalize_left_input_name(graph, left_input_name)
        left_transform = _shared_left_operand_transform(module)
        group_key = (canonical_left_input, left_transform)
        consumers_by_left_input[group_key].append((node_name, module, left_input_name))

    for (canonical_left_input, left_transform), consumers in consumers_by_left_input.items():
        if len(consumers) < min_fanout:
            continue

        group_tag = f"shared_left:{graph_prefix}:{canonical_left_input}:{left_transform}"
        group_nodes = []
        raw_left_inputs = []
        for node_name, module, left_input_name in consumers:
            setattr(module, "beaver_a_group", group_tag)
            setattr(module, "beaver_layer_tag", f"{graph_prefix}:{node_name}")
            grouped_nodes.add((graph_prefix, node_name))
            group_nodes.append(node_name)
            raw_left_inputs.append(left_input_name)

        summary["num_groups"] += 1
        summary["num_grouped_modules"] += len(consumers)
        summary["strategy_counts"]["generic_canonical_left"] += 1
        summary["groups"].append(
            {
                "graph": graph_prefix,
                "strategy": "generic_canonical_left",
                "left_input": canonical_left_input,
                "left_transform": left_transform,
                "fanout": len(consumers),
                "raw_left_inputs": raw_left_inputs,
                "nodes": group_nodes,
            }
        )


def annotate_shared_left_groups_crypten_model(model, min_fanout=2):
    summary = {
        "num_graphs": 0,
        "num_groups": 0,
        "num_grouped_modules": 0,
        "groups": [],
        "strategy_counts": {
            "explicit_qkv_lora_a": 0,
            "generic_canonical_left": 0,
        },
        "qkv_lora_a_diagnostics": [],
    }

    if min_fanout < 2:
        min_fanout = 2

    grouped_nodes = set()
    for graph_name, graph in model.named_modules():
        if not isinstance(graph, ct.nn.Graph):
            continue
        graph_prefix = _normalize_graph_name(graph_name)
        summary["num_graphs"] += 1

        _annotate_explicit_qkv_lora_a_groups(
            graph=graph,
            graph_prefix=graph_prefix,
            summary=summary,
            min_fanout=min_fanout,
            grouped_nodes=grouped_nodes,
        )
        _annotate_generic_canonical_left_groups(
            graph=graph,
            graph_prefix=graph_prefix,
            summary=summary,
            min_fanout=min_fanout,
            grouped_nodes=grouped_nodes,
        )

    summary["groups"] = sorted(
        summary["groups"],
        key=lambda item: (
            item["graph"],
            item.get("strategy", ""),
            item.get("family", ""),
            item["left_input"],
        ),
    )
    summary["qkv_lora_a_diagnostics"] = sorted(
        summary["qkv_lora_a_diagnostics"],
        key=lambda item: (item["graph"], item["family"]),
    )
    return summary
