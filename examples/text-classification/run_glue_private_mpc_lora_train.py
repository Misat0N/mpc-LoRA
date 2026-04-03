# coding=utf-8
# Modified by SHAFT's team: Private Text Classification on GLUE.
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing a Transformers model in priavte for sequence classification on GLUE."""

import argparse
import builtins
from collections import defaultdict, deque
import json
import logging
import math
import os
import sys
import time

import datasets
import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
    set_seed,
)

from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import crypten as ct
from crypten.common.reuse_context import clear_current_reuse_step, set_current_reuse_step
from crypten.config import cfg
from crypten.mpc.primitives import beaver as beaver_protocol
from multiprocess_launcher import MultiProcessLauncher

try:
    import evaluate
except ModuleNotFoundError:
    evaluate = None


# from star_linear_fixed import replace_linear_with_star_fixed

# # 可选：训练中需要失效时
# from crypten_ext.star_matmul_fixed import invalidate_weight, reset_all


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.42.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)
_PROCESS_LOG_TEE_INSTALLED = False
_PROCESS_LOG_MIRROR = None


class _TeeStream:
    def __init__(self, stream, mirror):
        self._stream = stream
        self._mirror = mirror

    def write(self, data):
        written = self._stream.write(data)
        self._mirror.write(data)
        return written

    def flush(self):
        self._stream.flush()
        self._mirror.flush()

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _require_evaluate(need_eval):
    if evaluate is None and need_eval:
        raise ModuleNotFoundError(
            "No module named 'evaluate'. Install it in your current env: "
            "`python -m pip install evaluate`."
        )


def _get_rank():
    try:
        return ct.communicator.get().get_rank()
    except Exception:
        return -1


def _shape_of(tensor):
    try:
        return tuple(tensor.size())
    except Exception:
        return "unknown"


def _cfg_snapshot():
    snapshot = {}
    try:
        snapshot["top_level_keys"] = list(cfg.config.keys())
    except Exception as err:
        snapshot["top_level_keys"] = f"<error: {type(err).__name__}: {err}>"

    for section in ("encoder", "debug", "cost", "mpc"):
        try:
            snapshot[f"has_{section}"] = hasattr(cfg, section)
        except Exception as err:
            snapshot[f"has_{section}"] = f"<error: {type(err).__name__}: {err}>"

    try:
        snapshot["precision_bits"] = cfg.encoder.precision_bits
    except Exception as err:
        snapshot["precision_bits"] = f"<error: {type(err).__name__}: {err}>"

    try:
        snapshot["validation_mode"] = cfg.debug.validation_mode
    except Exception as err:
        snapshot["validation_mode"] = f"<error: {type(err).__name__}: {err}>"

    return snapshot


def _loss_snapshot(loss_tensor):
    snapshot = {
        "loss_type": type(loss_tensor).__name__,
        "loss_shape": _shape_of(loss_tensor),
        "python_recursion_limit": sys.getrecursionlimit(),
    }
    try:
        snapshot["grad_fn"] = type(loss_tensor.grad_fn).__name__ if loss_tensor.grad_fn is not None else None
    except Exception as err:
        snapshot["grad_fn"] = f"<error: {type(err).__name__}: {err}>"
    try:
        snapshot["children_len"] = len(loss_tensor.children)
    except Exception as err:
        snapshot["children_len"] = f"<error: {type(err).__name__}: {err}>"
    return snapshot


def _tensor_preview(tensor, limit=8):
    try:
        flat = tensor.detach().reshape(-1).cpu().to(torch.float32)
        keep = min(limit, flat.numel())
        return [float(v) for v in flat[:keep].tolist()]
    except Exception as err:
        return f"<error: {type(err).__name__}: {err}>"


def _tensor_stats(tensor):
    try:
        flat = tensor.detach().reshape(-1).cpu().to(torch.float32)
        if flat.numel() == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "mean": float(flat.mean().item()),
        }
    except Exception as err:
        return {"error": f"{type(err).__name__}: {err}"}


def _tensor_abs_diff_stats(lhs, rhs):
    try:
        diff = (lhs.detach().cpu().to(torch.float32) - rhs.detach().cpu().to(torch.float32)).abs().reshape(-1)
        if diff.numel() == 0:
            return {"max_abs": 0.0, "mean_abs": 0.0}
        return {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
        }
    except Exception as err:
        return {"error": f"{type(err).__name__}: {err}"}


def _run_eval_numeric_probe(rank, model, private_model, batch, token_type_ids, device, preview_limit=8):
    summary = None
    plain_training = model.training
    private_training = private_model.training
    try:
        model.train(False)
        private_model.train(False)
        with torch.no_grad():
            plain_outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=token_type_ids,
            )
            plain_logits = plain_outputs.logits.detach().cpu()
        with ct.no_grad():
            inputs_probe = ct.cryptensor(batch["input_ids"]).to(device)
            attention_mask_probe = ct.cryptensor(batch["attention_mask"]).to(device)
            token_type_probe = ct.cryptensor(token_type_ids).to(device)
            private_logits_eval_nograd_enc = private_model(
                inputs_probe, attention_mask_probe, token_type_probe
            )
            private_logits_eval_nograd = private_logits_eval_nograd_enc.get_plain_text().detach().cpu()
        private_logits_eval_grad_enc = private_model(inputs_probe, attention_mask_probe, token_type_probe)
        private_logits_eval_grad = private_logits_eval_grad_enc.get_plain_text().detach().cpu()
        embeddings_probe = _run_embeddings_numeric_probe(
            model,
            batch["input_ids"],
            token_type_ids,
            device,
            preview_limit=preview_limit,
        )
        y_onehot = F.one_hot(batch["labels"], num_classes=plain_logits.size(-1)).float().cpu()
        summary = {
            "embeddings_probe": embeddings_probe,
            "plain_logits_preview": _tensor_preview(plain_logits, limit=preview_limit),
            "private_logits_preview": _tensor_preview(private_logits_eval_nograd, limit=preview_limit),
            "private_logits_eval_nograd_preview": _tensor_preview(private_logits_eval_nograd, limit=preview_limit),
            "private_logits_eval_grad_preview": _tensor_preview(private_logits_eval_grad, limit=preview_limit),
            "plain_logits_stats": _tensor_stats(plain_logits),
            "private_logits_stats": _tensor_stats(private_logits_eval_nograd),
            "private_logits_eval_nograd_stats": _tensor_stats(private_logits_eval_nograd),
            "private_logits_eval_grad_stats": _tensor_stats(private_logits_eval_grad),
            "plain_private_abs_diff": _tensor_abs_diff_stats(private_logits_eval_nograd, plain_logits),
            "plain_private_eval_nograd_abs_diff": _tensor_abs_diff_stats(private_logits_eval_nograd, plain_logits),
            "plain_private_eval_grad_abs_diff": _tensor_abs_diff_stats(private_logits_eval_grad, plain_logits),
            "eval_nograd_vs_grad_abs_diff": _tensor_abs_diff_stats(
                private_logits_eval_nograd, private_logits_eval_grad
            ),
            "plain_eval_mse": float(((plain_logits - y_onehot) * (plain_logits - y_onehot)).mean().item()),
            "private_eval_mse_from_revealed_logits": float(
                ((private_logits_eval_nograd - y_onehot) * (private_logits_eval_nograd - y_onehot)).mean().item()
            ),
            "private_eval_nograd_mse_from_revealed_logits": float(
                ((private_logits_eval_nograd - y_onehot) * (private_logits_eval_nograd - y_onehot)).mean().item()
            ),
            "private_eval_grad_mse_from_revealed_logits": float(
                ((private_logits_eval_grad - y_onehot) * (private_logits_eval_grad - y_onehot)).mean().item()
            ),
        }
        if rank == 0:
            logger.info("[numeric-probe] eval_forward_compare=%s", summary)
    finally:
        model.train(plain_training)
        private_model.train(private_training)
    return summary


def _run_embeddings_numeric_probe(model, input_ids, token_type_ids, device, preview_limit=8):
    from crypten.nn.module import Embedding as CrypTenEmbedding
    from crypten.nn.module import LayerNormalization as CrypTenLayerNormalization

    bert_embeddings = model.bert.embeddings
    seq_len = input_ids.size(1)
    position_ids = bert_embeddings.position_ids[:, :seq_len].to(input_ids.device)
    dummy_padding_idx = torch.tensor(-1, device=device)

    with torch.no_grad():
        plain_word = bert_embeddings.word_embeddings(input_ids).detach().cpu()
        plain_token = bert_embeddings.token_type_embeddings(token_type_ids).detach().cpu()
        plain_position = bert_embeddings.position_embeddings(position_ids).detach().cpu()
        plain_sum = (plain_word + plain_token + plain_position).detach().cpu()
        plain_norm = bert_embeddings.LayerNorm(plain_sum).detach().cpu()

    embed_op = CrypTenEmbedding()
    norm_op = CrypTenLayerNormalization(axis=-1, eps=bert_embeddings.LayerNorm.eps)

    word_weight_enc = ct.cryptensor(bert_embeddings.word_embeddings.weight.detach()).to(device)
    token_weight_enc = ct.cryptensor(bert_embeddings.token_type_embeddings.weight.detach()).to(device)
    position_weight_enc = ct.cryptensor(bert_embeddings.position_embeddings.weight.detach()).to(device)
    norm_weight_enc = ct.cryptensor(bert_embeddings.LayerNorm.weight.detach()).to(device)
    norm_bias_enc = ct.cryptensor(bert_embeddings.LayerNorm.bias.detach()).to(device)

    input_ids_default = ct.cryptensor(input_ids).to(device)
    token_type_default = ct.cryptensor(token_type_ids).to(device)
    position_ids_default = ct.cryptensor(position_ids).to(device)

    input_ids_p0 = ct.cryptensor(input_ids, precision=0).to(device)
    token_type_p0 = ct.cryptensor(token_type_ids, precision=0).to(device)
    position_ids_p0 = ct.cryptensor(position_ids, precision=0).to(device)

    with ct.no_grad():
        word_default = embed_op((word_weight_enc, input_ids_default, dummy_padding_idx)).get_plain_text().detach().cpu()
        word_p0 = embed_op((word_weight_enc, input_ids_p0, dummy_padding_idx)).get_plain_text().detach().cpu()

        token_default = embed_op((token_weight_enc, token_type_default, dummy_padding_idx)).get_plain_text().detach().cpu()
        token_p0 = embed_op((token_weight_enc, token_type_p0, dummy_padding_idx)).get_plain_text().detach().cpu()

        position_default = embed_op((position_weight_enc, position_ids_default, dummy_padding_idx)).get_plain_text().detach().cpu()
        position_p0 = embed_op((position_weight_enc, position_ids_p0, dummy_padding_idx)).get_plain_text().detach().cpu()

        sum_default_enc = (
            embed_op((word_weight_enc, input_ids_default, dummy_padding_idx))
            + embed_op((token_weight_enc, token_type_default, dummy_padding_idx))
            + embed_op((position_weight_enc, position_ids_default, dummy_padding_idx))
        )
        sum_p0_enc = (
            embed_op((word_weight_enc, input_ids_p0, dummy_padding_idx))
            + embed_op((token_weight_enc, token_type_p0, dummy_padding_idx))
            + embed_op((position_weight_enc, position_ids_p0, dummy_padding_idx))
        )
        sum_default = sum_default_enc.get_plain_text().detach().cpu()
        sum_p0 = sum_p0_enc.get_plain_text().detach().cpu()
        norm_default = norm_op((sum_default_enc, norm_weight_enc, norm_bias_enc)).get_plain_text().detach().cpu()
        norm_p0 = norm_op((sum_p0_enc, norm_weight_enc, norm_bias_enc)).get_plain_text().detach().cpu()

    return {
        "plain_word_preview": _tensor_preview(plain_word, limit=preview_limit),
        "private_word_default_preview": _tensor_preview(word_default, limit=preview_limit),
        "private_word_p0_preview": _tensor_preview(word_p0, limit=preview_limit),
        "word_default_abs_diff": _tensor_abs_diff_stats(word_default, plain_word),
        "word_p0_abs_diff": _tensor_abs_diff_stats(word_p0, plain_word),
        "token_default_abs_diff": _tensor_abs_diff_stats(token_default, plain_token),
        "token_p0_abs_diff": _tensor_abs_diff_stats(token_p0, plain_token),
        "position_default_abs_diff": _tensor_abs_diff_stats(position_default, plain_position),
        "position_p0_abs_diff": _tensor_abs_diff_stats(position_p0, plain_position),
        "sum_default_abs_diff": _tensor_abs_diff_stats(sum_default, plain_sum),
        "sum_p0_abs_diff": _tensor_abs_diff_stats(sum_p0, plain_sum),
        "norm_default_abs_diff": _tensor_abs_diff_stats(norm_default, plain_norm),
        "norm_p0_abs_diff": _tensor_abs_diff_stats(norm_p0, plain_norm),
        "norm_default_preview": _tensor_preview(norm_default, limit=preview_limit),
        "norm_p0_preview": _tensor_preview(norm_p0, limit=preview_limit),
        "plain_norm_preview": _tensor_preview(plain_norm, limit=preview_limit),
    }


def _recompute_loss_from_revealed_logits(logits_plain, labels_plain, loss_type):
    logits_plain = logits_plain.detach().cpu().to(torch.float32)
    labels_plain = labels_plain.detach().cpu().to(torch.float32)
    if loss_type == "ce":
        probs = torch.softmax(logits_plain, dim=-1)
        return float(-(labels_plain * torch.log(probs)).sum(dim=-1).mean().item())
    return float(((logits_plain - labels_plain) * (logits_plain - labels_plain)).mean().item())


def _collect_train_numeric_probe(rank, logits_enc, loss_enc, y_onehot, loss_type, preview_limit=8):
    logits_plain = logits_enc.get_plain_text().detach().cpu()
    revealed_loss = float(loss_enc.get_plain_text().item())
    labels_plain = y_onehot.detach().cpu()
    recomputed_loss = _recompute_loss_from_revealed_logits(logits_plain, labels_plain, loss_type)
    summary = {
        "loss_type": loss_type,
        "revealed_logits_preview": _tensor_preview(logits_plain, limit=preview_limit),
        "revealed_logits_stats": _tensor_stats(logits_plain),
        "label_onehot_preview": _tensor_preview(labels_plain, limit=preview_limit),
        "revealed_loss": revealed_loss,
        "recomputed_loss_from_revealed_logits": recomputed_loss,
        "loss_minus_recomputed": float(revealed_loss - recomputed_loss),
    }
    if rank == 0:
        logger.info("[numeric-probe] train_loss_compare=%s", summary)
    return summary


def _collect_train_pre_backward_numeric_probe(rank, logits_enc, loss_enc, y_onehot, loss_type, preview_limit=8):
    logits_plain = logits_enc.get_plain_text().detach().cpu()
    revealed_loss = float(loss_enc.get_plain_text().item())
    labels_plain = y_onehot.detach().cpu()
    recomputed_loss = _recompute_loss_from_revealed_logits(logits_plain, labels_plain, loss_type)
    summary = {
        "loss_type": loss_type,
        "revealed_logits_preview": _tensor_preview(logits_plain, limit=preview_limit),
        "revealed_logits_stats": _tensor_stats(logits_plain),
        "label_onehot_preview": _tensor_preview(labels_plain, limit=preview_limit),
        "revealed_loss": revealed_loss,
        "recomputed_loss_from_revealed_logits": recomputed_loss,
        "loss_minus_recomputed": float(revealed_loss - recomputed_loss),
    }
    if rank == 0:
        logger.info("[numeric-probe] train_pre_backward_compare=%s", summary)
    return summary


def _delta_dict(after, before):
    keys = set(before.keys()) | set(after.keys())
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def _mean(values):
    if len(values) == 0:
        return 0.0
    return float(sum(values) / len(values))


def _new_reuse_runtime_profile():
    return {
        "prep_time_s": [],
        "forward_time_s": [],
        "backward_time_s": [],
        "optimizer_time_s": [],
        "step_time_s": [],
        "comm_rounds": [],
        "comm_bytes": [],
        "comm_time_s": [],
        "triple_generate_calls": [],
        "beaver_reveal_calls": [],
        "beaver_revealed_tensors": [],
        "a_cache_hit": [],
        "a_cache_miss": [],
        "a_base_cache_hit": [],
        "a_base_cache_miss": [],
        "a_derived_cache_hit": [],
        "a_derived_generated": [],
        "b_cache_hit": [],
        "b_cache_miss": [],
        "b_base_cache_hit": [],
        "b_base_cache_miss": [],
        "b_derived_cache_hit": [],
        "b_derived_generated": [],
        "b_fresh_generated": [],
        "c_cache_hit": [],
        "c_cache_miss": [],
        "c_cache_probe_hit": [],
        "c_cache_probe_miss": [],
        "c_cache_bypassed": [],
        "c_fresh_generated": [],
        "residual_cache_hit": [],
        "residual_cache_miss": [],
        "residual_anchor_hit": [],
        "residual_anchor_miss": [],
    }


def _finalize_reuse_runtime_profile(profile):
    return {key: _mean(values) for key, values in profile.items()}


def _configure_reuse_experiment(args):
    cfg.mpc.experimental_reuse_mask = args.experimental_reuse_mask
    cfg.mpc.reuse_mode = str(args.reuse_mode).upper()
    cfg.mpc.reuse_scope = "STEP"
    cfg.mpc.reuse_op_types = ["matmul"]
    cfg.mpc.reuse_tagging = True
    if args.reuse_profile:
        cfg.communicator.verbose = True


def _synchronize_timing_device(device):
    if isinstance(device, torch.device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return
    if isinstance(device, str):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return
    try:
        device_type = getattr(device, "type", None)
    except Exception:
        device_type = None
    if device_type == "cuda":
        torch.cuda.synchronize(device)


def _is_shared_left_groupable_module(module):
    return isinstance(module, (ct.nn.Gemm, ct.nn.Linear, ct.nn.MatMul))


def _shared_left_operand_transform(module):
    if isinstance(module, ct.nn.Gemm):
        return "transpose" if getattr(module, "trans_a", False) else "identity"
    return "identity"


def _annotate_shared_left_groups_crypten_model(model, min_fanout=2):
    summary = {
        "num_graphs": 0,
        "num_groups": 0,
        "num_grouped_modules": 0,
        "groups": [],
    }
    if min_fanout < 2:
        min_fanout = 2

    for graph_name, graph in model.named_modules():
        if not isinstance(graph, ct.nn.Graph):
            continue
        summary["num_graphs"] += 1
        consumers_by_left_input = defaultdict(list)
        for node_name, input_names in graph._graph.items():
            module = graph._modules.get(node_name)
            if module is None or not _is_shared_left_groupable_module(module):
                continue
            if len(input_names) == 0:
                continue
            left_input_name = input_names[0]
            left_input_module = graph._modules.get(left_input_name)
            if isinstance(left_input_module, ct.nn.Parameter):
                continue
            left_transform = _shared_left_operand_transform(module)
            group_key = (left_input_name, left_transform)
            consumers_by_left_input[group_key].append((node_name, module))

        for (left_input_name, left_transform), consumers in consumers_by_left_input.items():
            if len(consumers) < min_fanout:
                continue
            graph_prefix = graph_name if graph_name else "root"
            group_tag = f"shared_left:{graph_prefix}:{left_input_name}:{left_transform}"
            group_nodes = []
            for node_name, module in consumers:
                setattr(module, "beaver_a_group", group_tag)
                setattr(module, "beaver_layer_tag", f"{graph_prefix}:{node_name}")
                group_nodes.append(node_name)
            summary["num_groups"] += 1
            summary["num_grouped_modules"] += len(consumers)
            summary["groups"].append(
                {
                    "graph": graph_prefix,
                    "left_input": left_input_name,
                    "left_transform": left_transform,
                    "fanout": len(consumers),
                    "nodes": group_nodes,
                }
            )

    summary["groups"] = sorted(
        summary["groups"],
        key=lambda item: (item["graph"], item["left_input"]),
    )
    return summary


def _safe_metric_compute(metric, steps, rank, phase):
    if steps <= 0:
        logger.warning(
            "[rank %s] %s metric skipped: no batches were added (likely filtered by --len_data).",
            rank,
            phase,
        )
        return {"skipped": True, "reason": "no_batches", "steps": steps}
    return metric.compute()


def _set_classifier_only_trainable(model):
    for param in model.parameters():
        param.requires_grad = False

    trainable_names = []
    for name, param in model.named_parameters():
        if name.startswith("classifier.") or name.startswith("score."):
            param.requires_grad = True
            trainable_names.append(name)

    # Fallback: keep at least one parameter trainable to avoid optimizer failures.
    if not trainable_names:
        named_params = list(model.named_parameters())
        if named_params:
            last_name, last_param = named_params[-1]
            last_param.requires_grad = True
            trainable_names.append(last_name)
    return trainable_names


def _count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer).__name__}")

        self.base = base_layer
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r) if r > 0 else 0.0
        # Keep dropout as functional op to avoid introducing Identity modules that
        # can break CrypTen -> PyTorch conversion for custom wrapped layers.
        self.lora_dropout_p = float(dropout)

        for p in self.base.parameters():
            p.requires_grad = False

        if self.r > 0:
            self.lora_A = nn.Linear(self.base.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, self.base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        result = self.base(x)
        if self.r > 0:
            x_lora = F.dropout(x, p=self.lora_dropout_p, training=self.training) if self.lora_dropout_p > 0 else x
            lora_out = self.lora_B(self.lora_A(x_lora)) * self.scaling
            result = result + lora_out
        return result


def _inject_lora_layers(module, target_keywords, r, alpha, dropout, prefix=""):
    replaced = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and any(k in full_name for k in target_keywords):
            setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
        else:
            replaced.extend(
                _inject_lora_layers(
                    child,
                    target_keywords=target_keywords,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    prefix=full_name,
                )
            )
    return replaced


def _set_lora_trainable(model, train_classifier_head=True):
    trainable = []
    for name, param in model.named_parameters():
        is_lora = ("lora_A." in name) or ("lora_B." in name)
        is_classifier = train_classifier_head and (name.startswith("classifier.") or name.startswith("score."))
        param.requires_grad = is_lora or is_classifier
        if param.requires_grad:
            trainable.append(name)
    return trainable


def _to_torch_state_value(value, _seen=None):
    if _seen is None:
        _seen = set()
    obj_id = id(value)
    if obj_id in _seen:
        return None
    _seen.add(obj_id)

    if torch.is_tensor(value):
        return value.detach().cpu()

    if hasattr(value, "get_plain_text"):
        try:
            plain = value.get_plain_text()
            if torch.is_tensor(plain):
                return plain.detach().cpu()
        except Exception:
            pass

    for attr_name in ("data", "_tensor", "share"):
        child = getattr(value, attr_name, None)
        if child is not None and child is not value:
            child_tensor = _to_torch_state_value(child, _seen=_seen)
            if child_tensor is not None:
                return child_tensor

    return None


def _strip_state_key_suffixes(key):
    suffixes = (".data", "._tensor", ".share")
    changed = True
    while changed and key:
        changed = False
        for suffix in suffixes:
            if key.endswith(suffix):
                key = key[: -len(suffix)]
                changed = True
    return key


def _canonicalize_state_key(key):
    # CrypTen modules may serialize keys with internal path markers.
    # Normalize to standard PyTorch-style dotted names.
    key = _strip_state_key_suffixes(key)
    parts = [p for p in key.split(".") if p not in {"_modules", "_parameters", "_buffers"}]
    if parts and parts[0] == "module":
        parts = parts[1:]
    return _strip_state_key_suffixes(".".join(parts))


def _candidate_target_keys(raw_key):
    stripped_raw = _strip_state_key_suffixes(raw_key)
    cands = [raw_key, stripped_raw]
    canon = _canonicalize_state_key(raw_key)
    if canon and canon not in cands:
        cands.append(canon)
    if raw_key.startswith("module."):
        cands.append(raw_key[len("module."):])
    if stripped_raw.startswith("module."):
        cands.append(stripped_raw[len("module."):])
    if canon.startswith("module."):
        cands.append(canon[len("module."):])
    # Deduplicate while preserving order
    uniq = []
    seen = set()
    for k in cands:
        if k and k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def _recover_plain_model_from_private(private_model, template_model, rank):
    # Preferred path: load decrypted CrypTen state into the original PyTorch template.
    try:
        ct_state = private_model.state_dict()
    except Exception:
        logger.exception("[rank %s] failed to read private_model.state_dict()", rank)
        ct_state = None

    if ct_state is None:
        raise RuntimeError("state_dict_unavailable")

    template_state = template_model.state_dict()
    template_keys = set(template_state.keys())
    mapped = {}

    for raw_key, raw_value in ct_state.items():
        tensor_value = _to_torch_state_value(raw_value)
        if tensor_value is None:
            continue

        target_key = None
        for cand in _candidate_target_keys(raw_key):
            if cand in template_keys:
                target_key = cand
                break

        if target_key is None:
            continue

        target_tensor = template_state[target_key]
        if tuple(target_tensor.shape) != tuple(tensor_value.shape):
            continue

        mapped[target_key] = tensor_value.to(dtype=target_tensor.dtype)

    if not mapped:
        sample_keys = list(ct_state.keys())[:10]
        logger.error("[rank %s] state_dict recovery mapped 0 params. sample_ct_keys=%s", rank, sample_keys)
        raise RuntimeError("state_dict_recovery_no_match")

    updated_state = dict(template_state)
    updated_state.update(mapped)
    load_result = template_model.load_state_dict(updated_state, strict=False)
    logger.info(
        "[rank %s] recovered PyTorch model via state_dict: mapped=%s missing=%s unexpected=%s",
        rank,
        len(mapped),
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )
    return template_model


def _install_print_filter():
    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        if args:
            first = args[0]
            if isinstance(first, str):
                text = first.strip()
                if "index_add_" in text:
                    return
                if text.startswith("index.shape:"):
                    return
                if text.startswith("flat_index"):
                    return
                if text.startswith("grad_output_flat"):
                    return
                if text.startswith("grad type:"):
                    return
                if text.startswith("grad.shape:"):
                    return
                if text.startswith("comm byte:"):
                    return
        return original_print(*args, **kwargs)

    builtins.print = filtered_print


def _default_log_dir():
    return os.path.join(os.getcwd(), "logs", "run_glue_private_mpc_lora_train")


def _resolve_process_log_path(args):
    log_dir = args.output_dir if args.output_dir is not None else _default_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    rank = os.environ.get("RANK", "main")
    return os.path.join(log_dir, f"rank{rank}.log")


def _tee_process_streams(log_path):
    global _PROCESS_LOG_TEE_INSTALLED, _PROCESS_LOG_MIRROR
    if _PROCESS_LOG_TEE_INSTALLED:
        return log_path

    mirror = open(log_path, "a", buffering=1, encoding="utf-8")
    _PROCESS_LOG_MIRROR = mirror
    sys.stdout = _TeeStream(sys.stdout, mirror)
    sys.stderr = _TeeStream(sys.stderr, mirror)
    _PROCESS_LOG_TEE_INSTALLED = True
    return log_path


def _configure_process_logging(args):
    log_path = _tee_process_streams(_resolve_process_log_path(args))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True,
    )

    if args.allow_spam_logs:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        logging.getLogger("torch.distributed").setLevel(logging.INFO)
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.INFO)
    else:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        logging.getLogger("torch.distributed").setLevel(logging.WARNING)
        logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    return log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=-1,
        help="Number of validation data to run, set to -1 if run the whole dataset.",
    )
    parser.add_argument(
        "--len_data",
        type=int,
        default=-1,
        help="Sequence length of data to run, set to -1 if run the whole dataset.",
    )
    parser.add_argument(
        "--comp",
        action="store_true",
        help="If passed, estimate computation time (without communication).",
    )
    parser.add_argument(
        "--acc",
        action="store_true",
        help="If passed, evaluate private inference accuracy on the entire dataset.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the train dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5,
        help="Maximum number of training steps to run.",
    )
    parser.add_argument(
        "--log_every_steps",
        type=int,
        default=1,
        help="Logging interval for training steps.",
    )
    parser.add_argument(
        "--eval_max_steps",
        type=int,
        default=-1,
        help="Maximum number of evaluation steps after training. -1 means full eval split.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the output.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--train_max_samples",
        type=int,
        default=-1,
        help="Cap train split size before tokenization. -1 means full split.",
    )
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=-1,
        help="Cap eval split size before tokenization. -1 means full split.",
    )
    parser.add_argument(
        "--train_classifier_only",
        action="store_true",
        help="If passed, freeze encoder and train only classifier head parameters.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="query,value",
        help="Comma-separated name keywords of Linear modules to apply LoRA on.",
    )
    parser.add_argument(
        "--freeze_classifier_head",
        action="store_true",
        help="If passed, freeze classifier/score head and train only LoRA params.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate for MPC optimizer.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for MPC SGD optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for MPC SGD optimizer.",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="Enable Nesterov momentum for MPC SGD optimizer.",
    )
    parser.add_argument(
        "--grad_threshold",
        type=float,
        default=-1.0,
        help=(
            "If > 0, zero out gradient elements whose absolute value exceeds this threshold "
            "inside CrypTen SGD."
        ),
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="auto",
        choices=["auto", "mse", "ce"],
        help=(
            "Training loss to use. 'auto' resolves to 'mse' in --quick_run for stability "
            "and 'ce' otherwise for better classification accuracy."
        ),
    )
    parser.add_argument(
        "--ce_softmax_method",
        type=str,
        default="default",
        choices=["default", "reciprocal", "ode"],
        help=(
            "Softmax approximation to use inside CrypTen cross-entropy. "
            "'default' keeps cfg.functions.softmax_method. "
            "Use 'reciprocal' only for explicit experiments."
        ),
    )
    parser.add_argument(
        "--train_loss_window",
        type=int,
        default=8,
        help="Window size for reporting running average train loss alongside the raw per-step loss.",
    )
    parser.add_argument(
        "--skip_private_eval",
        action="store_true",
        help="If passed, skip private evaluation after training.",
    )
    parser.add_argument(
        "--skip_plain_eval",
        action="store_true",
        help="If passed, skip plaintext evaluation after decrypting model.",
    )
    parser.add_argument(
        "--quick_run",
        action="store_true",
        help="Ultra-light preset: tiny sample, short sequence, 1 train step, reduced LoRA rank, skip eval.",
    )
    parser.add_argument(
        "--print_comm_cost",
        action="store_true",
        help="If passed, print CrypTen communication cost statistics during execution.",
    )
    parser.add_argument(
        "--experimental_reuse_mask",
        action="store_true",
        help="Enable the Beaver mask reuse performance experiment in CrypTen matmul.",
    )
    parser.add_argument(
        "--reuse_mode",
        type=str,
        default="SHARED_LEFT",
        choices=["SHARED_LEFT", "FIX_A", "FIX_AB"],
        help=(
            "Beaver reuse mode when --experimental_reuse_mask is enabled. "
            "SHARED_LEFT is the main path: reuse left Beaver mask / epsilon across sibling matmuls "
            "that share the same left operand."
        ),
    )
    parser.add_argument(
        "--shared_left_min_fanout",
        type=int,
        default=2,
        help=(
            "Minimum number of sibling Gemm/Linear consumers with the same left input required "
            "to auto-annotate a shared-left reuse group in the CrypTen graph."
        ),
    )
    parser.add_argument(
        "--shared_left_log_groups",
        type=int,
        default=12,
        help="Maximum number of detected shared-left groups to print in startup logs.",
    )
    parser.add_argument(
        "--reuse_profile",
        action="store_true",
        help="Collect per-step Beaver/communication runtime stats during training.",
    )
    parser.add_argument(
        "--reuse_log_every_steps",
        type=int,
        default=1,
        help="Logging interval for reuse runtime stats when --reuse_profile is enabled.",
    )
    parser.add_argument(
        "--allow_spam_logs",
        action="store_true",
        help="If passed, do not filter verbose third-party debug prints (e.g. index_add debug).",
    )
    parser.add_argument(
        "--debug_numeric_probe",
        action="store_true",
        help=(
            "Run an extra step-0 numeric probe that compares plaintext vs CrypTen logits and checks "
            "whether revealed loss matches loss recomputed from revealed logits."
        ),
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help=(
            "Comma-separated GPU ids for process mapping, e.g. '0,1'. "
            "If empty, use all visible CUDA devices and map by rank."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.validation_file is None:
        raise ValueError("Need either a task name or a validation file.")
    else:
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


def main():
    script_start_time = time.time()
    args = parse_args()
    _configure_reuse_experiment(args)

    if args.quick_run:
        args.pad_to_max_length = True
        args.max_length = min(args.max_length, 32)
        args.len_data = args.max_length
        args.max_train_steps = 1
        args.log_every_steps = 1
        args.eval_max_steps = 1
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        args.train_max_samples = 64 if args.train_max_samples < 0 else min(args.train_max_samples, 64)
        args.eval_max_samples = 64 if args.eval_max_samples < 0 else min(args.eval_max_samples, 64)
        args.lora_r = min(args.lora_r, 4)
        args.freeze_classifier_head = False
        args.skip_private_eval = True
        args.skip_plain_eval = True

    if args.loss_type == "auto":
        args.loss_type = "mse" if args.quick_run else "ce"

    need_eval = (not args.skip_private_eval) or (not args.skip_plain_eval)
    _require_evaluate(need_eval)

    if args.seed is None:
        args.seed = 1234
    set_seed(args.seed)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_private", args)

    process_log_path = _configure_process_logging(args)
    if not args.allow_spam_logs:
        _install_print_filter()

    # CrypTen autograd walks graph recursively; deep graphs can exceed Python's default recursion limit (1000).
    old_recursion_limit = sys.getrecursionlimit()
    target_recursion_limit = max(old_recursion_limit, 20000)
    if target_recursion_limit != old_recursion_limit:
        sys.setrecursionlimit(target_recursion_limit)
    logger.info("train-smoke start pid=%s argv=%s", os.getpid(), " ".join(sys.argv))
    logger.info("process log path=%s", process_log_path)
    logger.info("python recursion limit %s -> %s", old_recursion_limit, sys.getrecursionlimit())
    logger.info("initial cfg snapshot=%s", _cfg_snapshot())
    logger.info("resolved seed=%s", args.seed)
    logger.info("resolved loss_type=%s", args.loss_type)
    logger.info("resolved ce_softmax_method=%s", args.ce_softmax_method)
    if args.loss_type == "ce" and args.ce_softmax_method == "reciprocal":
        logger.warning(
            "[ce] softmax_method=reciprocal can still be numerically fragile in MPC fixed-point; "
            "prefer --ce_softmax_method default for stable training."
        )
    if args.quick_run:
        logger.info(
            "[quick-run] enabled: len=%s max_length=%s train_steps=%s train_samples=%s eval_samples=%s "
            "lora_r=%s freeze_classifier_head=%s skip_private_eval=%s skip_plain_eval=%s loss_type=%s",
            args.len_data,
            args.max_length,
            args.max_train_steps,
            args.train_max_samples,
            args.eval_max_samples,
            args.lora_r,
            args.freeze_classifier_head,
            args.skip_private_eval,
            args.skip_plain_eval,
            args.loss_type,
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("nyu-mll/glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    validation_key = "validation_matched" if args.task_name == "mnli" else "validation"

    if args.train_max_samples > 0 and "train" in raw_datasets:
        train_keep = min(args.train_max_samples, len(raw_datasets["train"]))
        raw_datasets["train"] = raw_datasets["train"].select(range(train_keep))
    if args.eval_max_samples > 0 and validation_key in raw_datasets:
        eval_keep = min(args.eval_max_samples, len(raw_datasets[validation_key]))
        raw_datasets[validation_key] = raw_datasets[validation_key].select(range(eval_keep))

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets[validation_key].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets[validation_key].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets[validation_key].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )
    lora_targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    replaced_lora_modules = _inject_lora_layers(
        model,
        target_keywords=lora_targets,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if len(replaced_lora_modules) == 0:
        raise ValueError(
            f"No Linear layers matched --lora_target_modules='{args.lora_target_modules}'. "
            "Please adjust target keywords."
        )

    trainable_names = _set_lora_trainable(model, train_classifier_head=(not args.freeze_classifier_head))
    if args.train_classifier_only:
        trainable_names = _set_classifier_only_trainable(model)
        logger.warning(
            "[train-params] --train_classifier_only is enabled. This overrides LoRA trainability selection."
        )

    logger.info(
        "[lora] injected=%s first_modules=%s",
        len(replaced_lora_modules),
        replaced_lora_modules[:12],
    )
    logger.info(
        "[train-params] total_trainable_params=%s trainable_name_count=%s",
        _count_trainable_params(model),
        len(trainable_names),
    )

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets[validation_key].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None
    
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            print(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    train_dataset = processed_datasets["train"]
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        generator=train_generator,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    logger.info("[rank %s] before ct.init initialized=%s", _get_rank(), ct.is_initialized())
    if not ct.is_initialized():
        ct.init()
    else:
        logger.warning("[rank %s] skip ct.init(): already initialized in launcher subprocess", _get_rank())
    rank = _get_rank()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for private light training, but no CUDA device is available.")

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count < 1:
        raise RuntimeError("No visible CUDA devices.")

    if args.gpu_ids.strip():
        gpu_id_list = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        gpu_id_list = list(range(visible_gpu_count))

    if len(gpu_id_list) < 1:
        raise ValueError("--gpu_ids is empty after parsing.")

    local_gpu_id = gpu_id_list[rank % len(gpu_id_list)]
    if local_gpu_id < 0 or local_gpu_id >= visible_gpu_count:
        raise ValueError(
            f"Invalid gpu id {local_gpu_id}. Visible cuda device count={visible_gpu_count}, --gpu_ids='{args.gpu_ids}'."
        )
    torch.cuda.set_device(local_gpu_id)
    device = f"cuda:{local_gpu_id}"

    logger.info("[rank %s] crypten initialized=%s", rank, ct.is_initialized())
    logger.info(
        "[rank %s] device mapping: visible_gpus=%s gpu_ids=%s selected_device=%s",
        rank,
        visible_gpu_count,
        gpu_id_list,
        device,
    )
    logger.info("[rank %s] cfg after crypten init=%s", rank, _cfg_snapshot())
    logger.info(
        "[rank %s] reuse config: experimental=%s mode=%s profile=%s log_every=%s shared_left_min_fanout=%s",
        rank,
        args.experimental_reuse_mask,
        args.reuse_mode,
        args.reuse_profile,
        args.reuse_log_every_steps,
        args.shared_left_min_fanout,
    )
    # print("done")
    # exit()
    dummy = torch.zeros_like(model.dummy_inputs["input_ids"])
    private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).encrypt().to(device)
    shared_left_group_summary = None
    if args.experimental_reuse_mask and str(args.reuse_mode).upper() == "SHARED_LEFT":
        shared_left_group_summary = _annotate_shared_left_groups_crypten_model(
            private_model, min_fanout=args.shared_left_min_fanout
        )
        if rank == 0:
            logger.info(
                "[shared-left] grouped_graphs=%s groups=%s grouped_modules=%s min_fanout=%s",
                shared_left_group_summary["num_graphs"],
                shared_left_group_summary["num_groups"],
                shared_left_group_summary["num_grouped_modules"],
                args.shared_left_min_fanout,
            )
            preview_limit = max(0, args.shared_left_log_groups)
            for group in shared_left_group_summary["groups"][:preview_limit]:
                logger.info(
                    "[shared-left] graph=%s left_input=%s left_transform=%s fanout=%s nodes=%s",
                    group["graph"],
                    group["left_input"],
                    group["left_transform"],
                    group["fanout"],
                    group["nodes"],
                )
            if shared_left_group_summary["num_groups"] == 0:
                logger.warning(
                    "[shared-left] no eligible grouped Gemm/Linear fan-out was detected in the CrypTen graph"
                )
    private_model.train()
    lr = args.learning_rate
    optimizer = ct.optim.SGD(
        private_model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        grad_threshold=(args.grad_threshold if args.grad_threshold > 0 else None),
    )
    logger.info(
        "[rank %s] model set to train mode; optimizer initialized (lr=%s momentum=%s weight_decay=%s nesterov=%s grad_threshold=%s)",
        rank,
        lr,
        args.momentum,
        args.weight_decay,
        args.nesterov,
        (args.grad_threshold if args.grad_threshold > 0 else None),
    )
    # 模型不加密
    # private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).to(device)


    # ct_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).to(device)
    # print(ct_model)
    # print(model)
    # print(private_model)
    # exit()  # USER MOD: removed to run training smoke test
    # print("==== Crypten model repr ====")
    # print(ct_model)  # 有些分支会给出层级结构
    # print("\n==== Modules that look like they carry weights ====")
    # for name, mod in ct_model.named_modules():
    #     # 1) 有 .weight/.bias 的
    #     for attr in ("weight", "bias"):
    #         t = getattr(mod, attr, None)
    #         if t is not None and hasattr(t, "size"):
    #             try:
    #                 print(f"[PARAM] {name}.{attr}: shape={tuple(t.size())}, type={type(t).__name__}")
    #             except Exception:
    #                 print(f"[PARAM] {name}.{attr}: type={type(t).__name__}")

    #     # 2) 其它“像矩阵”的属性（有 size 且维度≥2）
    #     for k, v in vars(mod).items():
    #         if k in ("weight", "bias"):
    #             continue
    #         if hasattr(v, "size"):
    #             try:
    #                 shp = tuple(v.size())
    #                 if len(shp) >= 2:
    #                     print(f"[MATRIX] {name}.{k}: shape={shp}, type={type(v).__name__}")
    #             except Exception:
    #                 pass
    # exit()  # USER MOD: removed to run training smoke test
    # for name, mod in ct_model.named_modules():
    #     w = getattr(mod, "weight", None)
    #     if w is not None:
    #         print("w is not None")
    #         exit()
    #         try:
    #             shape = tuple(w.size())
    #         except Exception:
    #             shape = "?"
    #         if isinstance(shape, tuple) and len(shape) == 2:
    #             b = getattr(mod, "bias", None)
    #             bshape = tuple(b.size()) if b is not None and hasattr(b, "size") else None
    #             print(f"[CAND] {name:40s} {type(mod).__name__:25s} weight={shape} bias={bshape}")
    # exit()

    # ct_model = replace_linear_with_star_fixed(ct_model)   # 仅这一行
    # private_model = ct_model.encrypt().to(device)

    train_start_time = time.time()
    global_step = 0
    reuse_runtime_profile = _new_reuse_runtime_profile() if args.reuse_profile else None
    numeric_probe_summary = None
    if args.reuse_profile:
        beaver_protocol.reset_reuse_stats(reset_cache=True)
        ct.reset_communication_stats()
    logger.info(
        "[rank %s] entering short-train loop max_train_steps=%s train_batch=%s train_loss_window=%s",
        rank,
        args.max_train_steps,
        args.per_device_train_batch_size,
        args.train_loss_window,
    )
    recent_train_losses = deque(maxlen=max(1, args.train_loss_window))
    for _, batch in enumerate(train_dataloader):
        rank = _get_rank()
        logger.info(
            "[rank %s] train_step=%03d batch_keys=%s input_shape=%s label_shape=%s",
            rank,
            global_step,
            sorted(batch.keys()),
            _shape_of(batch["input_ids"]),
            _shape_of(batch["labels"]),
        )
        if args.len_data > 0 and batch["input_ids"].shape[1] != args.len_data:
            continue

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(batch["input_ids"])
        if args.debug_numeric_probe and global_step == 0 and numeric_probe_summary is None:
            numeric_probe_summary = {
                "eval_forward_compare": _run_eval_numeric_probe(
                    rank,
                    model,
                    private_model,
                    batch,
                    token_type_ids,
                    device,
                )
            }
        step_id = global_step
        if args.experimental_reuse_mask:
            set_current_reuse_step(step_id)
            beaver_protocol.begin_reuse_step(step_id)
        comm_before = ct.get_communication_stats() if args.reuse_profile else None
        beaver_before = beaver_protocol.get_reuse_stats() if args.reuse_profile else None
        _synchronize_timing_device(device)
        step_start = time.perf_counter()
        prep_start = step_start
        inputs_enc = ct.cryptensor(batch["input_ids"]).to(device)
        attention_mask_enc = ct.cryptensor(batch["attention_mask"]).to(device)
        token_type_enc = ct.cryptensor(token_type_ids).to(device)
        optimizer.zero_grad()
        _synchronize_timing_device(device)
        prep_end = time.perf_counter()

        # forward (NO ct.no_grad for training)
        # 在不设置ct.no_grad时自动默认训练模式，forward过程会记录backward所需结果
        forward_start = prep_end
        logits_enc = private_model(inputs_enc, attention_mask_enc, token_type_enc)  # [B, num_labels]
        _synchronize_timing_device(device)
        forward_end = time.perf_counter()
        logger.info(
            "[rank %s] train_step=%03d forward_done dt=%.3fs logits_shape=%s",
            rank,
            global_step,
            forward_end - forward_start,
            _shape_of(logits_enc),
        )

        # For smoke tests we keep MSE available, but normal training should use CE.
        num_labels = logits_enc.size(-1)
        y_onehot = F.one_hot(batch["labels"], num_classes=num_labels).float()
        y_enc = ct.cryptensor(y_onehot).to(device)
        if args.loss_type == "ce":
            if args.ce_softmax_method == "default":
                loss_enc = logits_enc.cross_entropy(y_enc)
            else:
                with cfg.temp_override({"functions.softmax_method": args.ce_softmax_method}):
                    loss_enc = logits_enc.cross_entropy(y_enc)
        elif args.loss_type == "mse":
            diff = logits_enc - y_enc
            loss_enc = (diff * diff).mean()
        else:
            raise ValueError(f"Unsupported loss_type: {args.loss_type}")

        # optimizer (create once on first step)
        logger.info("[rank %s] train_step=%03d loss_snapshot=%s", rank, global_step, _loss_snapshot(loss_enc))
        if args.debug_numeric_probe and global_step == 0:
            numeric_probe_summary["train_pre_backward_compare"] = _collect_train_pre_backward_numeric_probe(
                rank, logits_enc, loss_enc, y_onehot, args.loss_type
            )

        if global_step == 1:
            logger.info("[rank %s] train_step=%03d cfg consistency check start", rank, global_step)
            from crypten.config import cfg as cfg_main
            import crypten.encoder as enc

            logger.info(
                "[rank %s] cfg id(main/imported)=%s/%s same=%s enc_has_encoder=%s snapshot=%s",
                rank,
                id(cfg_main),
                id(enc.cfg),
                id(cfg_main) == id(enc.cfg),
                hasattr(enc.cfg, "encoder"),
                _cfg_snapshot(),
            )
            # exit()
        logger.info("[rank %s] train_step=%03d backward_start", rank, global_step)
        backward_start = time.perf_counter()
        try:
            loss_enc.backward()
        except Exception:
            if args.experimental_reuse_mask:
                beaver_protocol.end_reuse_step(step_id)
                clear_current_reuse_step()
            logger.exception(
                "[rank %s] train_step=%03d backward_failed cfg=%s loss=%s",
                rank,
                global_step,
                _cfg_snapshot(),
                _loss_snapshot(loss_enc),
            )
            raise
        _synchronize_timing_device(device)
        backward_end = time.perf_counter()
        logger.info("[rank %s] train_step=%03d backward_done", rank, global_step)

        optimizer_start = time.perf_counter()
        try:
            optimizer.step()
        except Exception:
            if args.experimental_reuse_mask:
                beaver_protocol.end_reuse_step(step_id)
                clear_current_reuse_step()
            logger.exception("[rank %s] train_step=%03d optimizer_step_failed", rank, global_step)
            raise
        _synchronize_timing_device(device)
        optimizer_end = time.perf_counter()
        logger.info("[rank %s] train_step=%03d optimizer_step_done", rank, global_step)

        # reveal loss (ALL ranks must call get_plain_text / reveal)
        if args.debug_numeric_probe and global_step == 0:
            numeric_probe_summary["train_loss_compare"] = _collect_train_numeric_probe(
                rank, logits_enc, loss_enc, y_onehot, args.loss_type
            )
            loss_plain = numeric_probe_summary["train_loss_compare"]["revealed_loss"]
        else:
            loss_plain = loss_enc.get_plain_text().item()
        _synchronize_timing_device(device)
        step_end = time.perf_counter()
        if args.experimental_reuse_mask:
            beaver_protocol.end_reuse_step(step_id)
            clear_current_reuse_step()
        if args.reuse_profile:
            comm_after = ct.get_communication_stats()
            beaver_after = beaver_protocol.get_reuse_stats()
            comm_delta = _delta_dict(comm_after, comm_before)
            beaver_delta = _delta_dict(beaver_after, beaver_before)

            reuse_runtime_profile["prep_time_s"].append(prep_end - prep_start)
            reuse_runtime_profile["forward_time_s"].append(forward_end - forward_start)
            reuse_runtime_profile["backward_time_s"].append(backward_end - backward_start)
            reuse_runtime_profile["optimizer_time_s"].append(optimizer_end - optimizer_start)
            reuse_runtime_profile["step_time_s"].append(step_end - step_start)
            reuse_runtime_profile["comm_rounds"].append(comm_delta.get("rounds", 0))
            reuse_runtime_profile["comm_bytes"].append(comm_delta.get("bytes", 0))
            reuse_runtime_profile["comm_time_s"].append(comm_delta.get("time", 0.0))
            reuse_runtime_profile["triple_generate_calls"].append(
                beaver_delta.get("triple_generate_calls", 0)
            )
            reuse_runtime_profile["beaver_reveal_calls"].append(
                beaver_delta.get("beaver_reveal_calls", 0)
            )
            reuse_runtime_profile["beaver_revealed_tensors"].append(
                beaver_delta.get("beaver_revealed_tensors", 0)
            )
            reuse_runtime_profile["a_cache_hit"].append(beaver_delta.get("a_cache_hit", 0))
            reuse_runtime_profile["a_cache_miss"].append(beaver_delta.get("a_cache_miss", 0))
            reuse_runtime_profile["a_base_cache_hit"].append(
                beaver_delta.get("a_base_cache_hit", 0)
            )
            reuse_runtime_profile["a_base_cache_miss"].append(
                beaver_delta.get("a_base_cache_miss", 0)
            )
            reuse_runtime_profile["a_derived_cache_hit"].append(
                beaver_delta.get("a_derived_cache_hit", 0)
            )
            reuse_runtime_profile["a_derived_generated"].append(
                beaver_delta.get("a_derived_generated", 0)
            )
            reuse_runtime_profile["b_cache_hit"].append(beaver_delta.get("b_cache_hit", 0))
            reuse_runtime_profile["b_cache_miss"].append(beaver_delta.get("b_cache_miss", 0))
            reuse_runtime_profile["b_base_cache_hit"].append(
                beaver_delta.get("b_base_cache_hit", 0)
            )
            reuse_runtime_profile["b_base_cache_miss"].append(
                beaver_delta.get("b_base_cache_miss", 0)
            )
            reuse_runtime_profile["b_derived_cache_hit"].append(
                beaver_delta.get("b_derived_cache_hit", 0)
            )
            reuse_runtime_profile["b_derived_generated"].append(
                beaver_delta.get("b_derived_generated", 0)
            )
            reuse_runtime_profile["b_fresh_generated"].append(
                beaver_delta.get("b_fresh_generated", 0)
            )
            reuse_runtime_profile["c_cache_hit"].append(beaver_delta.get("c_cache_hit", 0))
            reuse_runtime_profile["c_cache_miss"].append(beaver_delta.get("c_cache_miss", 0))
            reuse_runtime_profile["c_cache_probe_hit"].append(
                beaver_delta.get("c_cache_probe_hit", 0)
            )
            reuse_runtime_profile["c_cache_probe_miss"].append(
                beaver_delta.get("c_cache_probe_miss", 0)
            )
            reuse_runtime_profile["c_cache_bypassed"].append(
                beaver_delta.get("c_cache_bypassed", 0)
            )
            reuse_runtime_profile["c_fresh_generated"].append(
                beaver_delta.get("c_fresh_generated", 0)
            )
            reuse_runtime_profile["residual_cache_hit"].append(
                beaver_delta.get("residual_cache_hit", 0)
            )
            reuse_runtime_profile["residual_cache_miss"].append(
                beaver_delta.get("residual_cache_miss", 0)
            )
            reuse_runtime_profile["residual_anchor_hit"].append(
                beaver_delta.get("residual_anchor_hit", 0)
            )
            reuse_runtime_profile["residual_anchor_miss"].append(
                beaver_delta.get("residual_anchor_miss", 0)
            )

            if rank == 0 and (global_step + 1) % max(1, args.reuse_log_every_steps) == 0:
                logger.info(
                    "[reuse-profile] step=%03d prep=%.4fs fwd=%.4fs bwd=%.4fs opt=%.4fs step=%.4fs "
                    "rounds=%s bytes=%s triple=%s reveals=%s reveal_tensors=%s "
                    "a_base_hit=%s a_base_miss=%s a_der_hit=%s a_der_new=%s "
                    "b_base_hit=%s b_base_miss=%s b_der_hit=%s b_der_new=%s b_fresh=%s "
                    "c_hit=%s c_miss=%s c_bypass=%s c_fresh=%s "
                    "res_hit=%s res_miss=%s anchor_hit=%s anchor_miss=%s",
                    global_step + 1,
                    prep_end - prep_start,
                    forward_end - forward_start,
                    backward_end - backward_start,
                    optimizer_end - optimizer_start,
                    step_end - step_start,
                    comm_delta.get("rounds", 0),
                    comm_delta.get("bytes", 0),
                    beaver_delta.get("triple_generate_calls", 0),
                    beaver_delta.get("beaver_reveal_calls", 0),
                    beaver_delta.get("beaver_revealed_tensors", 0),
                    beaver_delta.get("a_base_cache_hit", 0),
                    beaver_delta.get("a_base_cache_miss", 0),
                    beaver_delta.get("a_derived_cache_hit", 0),
                    beaver_delta.get("a_derived_generated", 0),
                    beaver_delta.get("b_base_cache_hit", 0),
                    beaver_delta.get("b_base_cache_miss", 0),
                    beaver_delta.get("b_derived_cache_hit", 0),
                    beaver_delta.get("b_derived_generated", 0),
                    beaver_delta.get("b_fresh_generated", 0),
                    beaver_delta.get("c_cache_probe_hit", 0),
                    beaver_delta.get("c_cache_probe_miss", 0),
                    beaver_delta.get("c_cache_bypassed", 0),
                    beaver_delta.get("c_fresh_generated", 0),
                    beaver_delta.get("residual_cache_hit", 0),
                    beaver_delta.get("residual_cache_miss", 0),
                    beaver_delta.get("residual_anchor_hit", 0),
                    beaver_delta.get("residual_anchor_miss", 0),
                )
        global_step += 1
        if global_step % max(1, args.log_every_steps) == 0:
            recent_train_losses.append(float(loss_plain))
            running_loss = sum(recent_train_losses) / len(recent_train_losses)
            logger.info(
                "[rank %s] [train] step=%03d loss=%.6f running_loss(window=%s)=%.6f",
                rank,
                global_step,
                loss_plain,
                len(recent_train_losses),
                running_loss,
            )
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            logger.info("[rank %s] reached max_train_steps=%s", rank, args.max_train_steps)
            break
    reuse_profile_summary = None
    if args.reuse_profile:
        reuse_profile_summary = _finalize_reuse_runtime_profile(reuse_runtime_profile)
        if rank == 0:
            logger.info("[reuse-profile] summary=%s", reuse_profile_summary)
    logger.info(
        "[rank %s] short-train finished steps=%s elapsed=%.3fs",
        rank,
        global_step,
        time.time() - train_start_time,
    )
    # Phase barrier: private eval / decrypt also use collectives. Without an
    # explicit sync here, one rank can leave training and enter the next phase
    # while another rank is still in the previous collective-heavy section.
    ct.barrier()

    private_eval_metric = {"skipped": True, "reason": "disabled"}
    if not args.skip_private_eval:
        # Step 3A: private evaluation on validation split.
        private_model.eval()
        private_metric = evaluate.load("glue", args.task_name) if args.task_name is not None else evaluate.load("accuracy")
        eval_steps = 0
        eval_skipped_by_len = 0
        for _, batch in enumerate(eval_dataloader):
            if args.len_data > 0 and batch["input_ids"].shape[1] != args.len_data:
                eval_skipped_by_len += 1
                continue

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(batch["input_ids"])

            inputs_enc = ct.cryptensor(batch["input_ids"]).to(device)
            attention_mask_enc = ct.cryptensor(batch["attention_mask"]).to(device)
            token_type_enc = ct.cryptensor(token_type_ids).to(device)
            with ct.no_grad():
                outputs_enc = private_model(inputs_enc, attention_mask_enc, token_type_enc)

            outputs = outputs_enc.get_plain_text().cpu()
            predictions = outputs.argmax(dim=-1) if not is_regression else outputs.squeeze()
            private_metric.add_batch(predictions=predictions, references=batch["labels"])
            eval_steps += 1

            if args.eval_max_steps > 0 and eval_steps >= args.eval_max_steps:
                break

        private_eval_metric = _safe_metric_compute(private_metric, eval_steps, rank, "eval-private")
        if rank == 0:
            logger.info(
                "[eval-private] steps=%s skipped_by_len=%s metric=%s",
                eval_steps,
                eval_skipped_by_len,
                private_eval_metric,
            )
    elif rank == 0:
        logger.info("[eval-private] skipped by flag --skip_private_eval")

    # Keep all ranks aligned before entering decrypt / plaintext recovery.
    ct.barrier()

    # Step 3B: recover a plaintext model and run plaintext evaluation.
    plain_eval_metric = {"skipped": True, "reason": "disabled"}
    plain_steps = 0
    plain_skipped_by_len = 0
    plain_recovery_error = None
    need_decrypt = (not args.skip_plain_eval) or (args.output_dir is not None)
    trained_model = None
    if need_decrypt:
        private_model.decrypt()
        # Keep decrypted CrypTen parameters on CPU so to_pytorch() can assign storage safely.
        private_model = private_model.to("cpu")
    ct.barrier()
    if rank == 0 and ((not args.skip_plain_eval) or (args.output_dir is not None)):
        try:
            trained_model = _recover_plain_model_from_private(private_model, model, rank)
            if not args.skip_plain_eval:
                trained_model = trained_model.to(device)
            trained_model.eval()
        except Exception as err:
            plain_recovery_error = f"{type(err).__name__}: {err}"
            logger.exception("[rank %s] recover_plain_model_failed: %s", rank, plain_recovery_error)
            trained_model = None
            plain_eval_metric = {
                "skipped": True,
                "reason": "plain_model_recovery_failed",
                "error": plain_recovery_error,
            }

    if rank == 0 and not args.skip_plain_eval and trained_model is not None:
        plain_metric = evaluate.load("glue", args.task_name) if args.task_name is not None else evaluate.load("accuracy")
        for _, batch in enumerate(eval_dataloader):
            if args.len_data > 0 and batch["input_ids"].shape[1] != args.len_data:
                plain_skipped_by_len += 1
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            with torch.no_grad():
                if token_type_ids is not None:
                    outputs = trained_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                else:
                    outputs = trained_model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            plain_metric.add_batch(predictions=predictions.cpu(), references=batch["labels"])
            plain_steps += 1

            if args.eval_max_steps > 0 and plain_steps >= args.eval_max_steps:
                break

        plain_eval_metric = _safe_metric_compute(plain_metric, plain_steps, rank, "eval-plain")
        logger.info(
            "[eval-plain] steps=%s skipped_by_len=%s metric=%s",
            plain_steps,
            plain_skipped_by_len,
            plain_eval_metric,
        )
    elif rank == 0 and not args.skip_plain_eval and trained_model is None:
        logger.warning("[eval-plain] skipped: plaintext model unavailable (%s)", plain_recovery_error)
    elif rank == 0:
        logger.info("[eval-plain] skipped by flag --skip_plain_eval")

    if rank == 0 and args.output_dir is not None:
        trained_model_dir = os.path.join(args.output_dir, "trained_model")
        if trained_model is not None:
            os.makedirs(trained_model_dir, exist_ok=True)
            trained_model.save_pretrained(trained_model_dir)
            tokenizer.save_pretrained(trained_model_dir)
            logger.info("[save] trained model saved to %s", trained_model_dir)
        else:
            logger.warning("[save] skip trained model export: plaintext model unavailable")

        summary = {
            "train_steps": global_step,
            "private_eval_metric": private_eval_metric,
            "plain_eval_metric": plain_eval_metric,
            "task_name": args.task_name,
            "max_train_steps": args.max_train_steps,
            "eval_max_steps": args.eval_max_steps,
            "experimental_reuse_mask": args.experimental_reuse_mask,
            "reuse_mode": args.reuse_mode,
            "shared_left_group_summary": shared_left_group_summary,
            "reuse_profile_summary": reuse_profile_summary,
            "numeric_probe_summary": numeric_probe_summary,
        }
        summary_path = os.path.join(args.output_dir, "train_eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("[save] summary saved to %s", summary_path)

    if rank == 0:
        logger.info("[train+eval] total elapsed=%.3fs", time.time() - script_start_time)
    return


if __name__ == "__main__":
    args = parse_args()
    if args.comp:
        # run without communication
        with cfg.temp_override({"cost.estimate_cost": True, "cost.estimate_mode": "comp"}):
            main()
    elif args.acc:
        # run without communication and cost printing
        with cfg.temp_override({"cost.estimate_cost": False}):
            main()
    else:
        # run with communication
        with cfg.temp_override({"cost.estimate_cost": args.print_comm_cost, "cost.estimate_mode": "comm"}):

            launcher = MultiProcessLauncher(2, main)
            launcher.start()
            launcher.join()
            launcher.terminate()
