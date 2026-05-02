# coding=utf-8

import argparse
import builtins
import json
import logging
import math
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUILD_LIB_ROOT = os.path.join(REPO_ROOT, "build", "lib")
if os.path.isdir(os.path.join(BUILD_LIB_ROOT, "crypten")) and BUILD_LIB_ROOT not in sys.path:
    sys.path.insert(0, BUILD_LIB_ROOT)

import datasets
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset

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

import crypten as ct
from crypten.config import cfg
from multiprocess_launcher import MultiProcessLauncher

try:
    import evaluate
except ModuleNotFoundError:
    evaluate = None


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


def _progress_print(message):
    print(message, flush=True)


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
                if text.startswith("[DEBUG index_add_"):
                    return
                if text.startswith("flat_index"):
                    return
                if text.startswith("grad_output_flat"):
                    return
                if text.startswith("grad type:"):
                    return
                if text.startswith("grad.shape:"):
                    return
        return original_print(*args, **kwargs)

    builtins.print = filtered_print


def _require_package(name, package_obj, install_hint):
    if package_obj is None:
        raise ModuleNotFoundError(f"No module named '{name}'. Install it with: {install_hint}")


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


def _count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _apply_trainable_scope(model, trainable_scope):
    if trainable_scope == "full":
        for _, param in model.named_parameters():
            param.requires_grad = True
        return

    if trainable_scope == "last_layer_classifier":
        trainable_prefixes = (
            "bert.encoder.layer.11.",
            "bert.pooler.",
            "classifier.",
        )
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)
        return

    if trainable_scope == "lora":
        for _, param in model.named_parameters():
            param.requires_grad = False
        return

    raise ValueError(f"Unsupported trainable_scope: {trainable_scope}")


class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer).__name__}")

        self.base = base_layer
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r) if r > 0 else 0.0
        self.lora_dropout_p = float(dropout)

        for param in self.base.parameters():
            param.requires_grad = False

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
            x_lora = (
                nn.functional.dropout(x, p=self.lora_dropout_p, training=self.training)
                if self.lora_dropout_p > 0
                else x
            )
            lora_out = self.lora_B(self.lora_A(x_lora)) * self.scaling
            result = result + lora_out
        return result


def _inject_lora_layers(module, target_keywords, r, alpha, dropout, prefix=""):
    replaced = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and any(keyword in full_name for keyword in target_keywords):
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


def _get_submodule_by_name(module, full_name):
    current = module
    for part in full_name.split("."):
        current = getattr(current, part)
    return current


def _module_name_to_graph_prefix(full_name):
    pieces = []
    for part in full_name.split("."):
        if part.isdigit() and pieces:
            pieces[-1] = f"{pieces[-1]}.{part}"
        else:
            pieces.append(part)
    return "/" + "/".join(pieces)


def _collapse_lora_graph_modules(private_model, pytorch_model, replaced_modules):
    collapsed = []
    for full_name in replaced_modules:
        graph_prefix = _module_name_to_graph_prefix(full_name)
        final_node = graph_prefix + "/Add_output_0"
        base_matmul_node = graph_prefix + "/base/MatMul_output_0"
        if final_node not in private_model._graph or base_matmul_node not in private_model._graph:
            continue

        input_name = private_model._graph[base_matmul_node][0]
        pytorch_lora = _get_submodule_by_name(pytorch_model, full_name)
        base_weight = pytorch_lora.base.weight.detach().clone()
        base_bias = pytorch_lora.base.bias.detach().clone() if pytorch_lora.base.bias is not None else None
        lora_A = pytorch_lora.lora_A.weight.detach().clone()
        lora_B = pytorch_lora.lora_B.weight.detach().clone()

        out_features, in_features = base_weight.shape
        mpc_layer = ct.nn.MPCLoRALinear(
            in_features=in_features,
            out_features=out_features,
            r=pytorch_lora.r,
            alpha=pytorch_lora.alpha,
            dropout=pytorch_lora.lora_dropout_p,
            bias=base_bias is not None,
            base_weight=base_weight,
            base_bias=base_bias,
        )
        with torch.no_grad():
            mpc_layer.lora_A.copy_(lora_A)
            mpc_layer.lora_B.copy_(lora_B)
        mpc_layer.training = pytorch_lora.training

        private_model._modules[final_node] = mpc_layer
        private_model._graph[final_node] = [input_name]

        obsolete_graph_nodes = [
            graph_prefix + "/base/Transpose_output_0",
            graph_prefix + "/base/MatMul_output_0",
            graph_prefix + "/base/Add_output_0",
            graph_prefix + "/lora_A/Transpose_output_0",
            graph_prefix + "/lora_A/MatMul_output_0",
            graph_prefix + "/lora_B/Transpose_output_0",
            graph_prefix + "/lora_B/MatMul_output_0",
            graph_prefix + "/Constant_output_0",
            graph_prefix + "/Mul_output_0",
        ]
        obsolete_module_nodes = obsolete_graph_nodes + [
            full_name + ".base.weight",
            full_name + ".base.bias",
            full_name + ".lora_A.weight",
            full_name + ".lora_B.weight",
        ]

        for node_name in obsolete_graph_nodes + obsolete_module_nodes:
            private_model._graph.pop(node_name, None)
        for node_name in obsolete_module_nodes:
            private_model._modules.pop(node_name, None)

        collapsed.append(full_name)
    return collapsed


def _set_lora_trainable(model, train_classifier_head=True):
    trainable_names = []
    for name, param in model.named_parameters():
        is_lora = ("lora_A" in name) or ("lora_B" in name)
        is_classifier = train_classifier_head and (name.startswith("classifier.") or name.startswith("score."))
        param.requires_grad = is_lora or is_classifier
        if param.requires_grad:
            trainable_names.append(name)
    return trainable_names


def _moving_average(values, window_size):
    if not values:
        return []
    smoothed = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window_size:
            running_sum -= values[index - window_size]
        denom = min(index + 1, window_size)
        smoothed.append(running_sum / denom)
    return smoothed


def _set_optimizer_lr(optimizer, lr_value):
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def _build_private_optimizer(args, parameters):
    trainable_params = [param for param in parameters if getattr(param, "requires_grad", False)]
    if args.optimizer == "sgd":
        return ct.optim.SGD(
            trainable_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw":
        return ct.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _compute_linear_warmup_lr(step, total_steps, base_lr, warmup_steps):
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return 0.0
    remaining = max(0, total_steps - step)
    decay_steps = max(1, total_steps - warmup_steps)
    return base_lr * float(remaining) / float(decay_steps)


def _private_batch_inputs(batch, device):
    token_type_ids = batch.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(batch["input_ids"])
    return (
        ct.cryptensor(batch["input_ids"]).to(device),
        ct.cryptensor(batch["attention_mask"]).to(device),
        ct.cryptensor(token_type_ids).to(device),
    )


def _private_cross_entropy_manual(logits_enc, labels_enc):
    dim = -1 if logits_enc.dim() > 1 else 0
    softmax = logits_enc.softmax(dim)
    log_probs = softmax.mul(100).log().sub(4.605170)
    loss_values = log_probs.mul(labels_enc).neg()
    return loss_values.sum().div(labels_enc.size(0))


def _compute_private_train_loss(
    logits_enc,
    labels,
    device,
    private_loss_mode,
    ce_softmax_method=None,
    ce_softmax_ode_lb=None,
):
    num_classes = logits_enc.size(-1)
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    labels_enc = ct.cryptensor(labels_one_hot).to(device)
    if private_loss_mode == "manual":
        override_config = {}
        if ce_softmax_method is not None:
            override_config["functions.softmax_method"] = ce_softmax_method
        if ce_softmax_ode_lb is not None and (ce_softmax_method == "ode" or cfg.functions.softmax_method == "ode"):
            override_config["functions.softmax_ode_lb"] = ce_softmax_ode_lb
        if override_config:
            with cfg.temp_override(override_config):
                return _private_cross_entropy_manual(logits_enc, labels_enc)
        return _private_cross_entropy_manual(logits_enc, labels_enc)
    return ct.nn.CrossEntropyLoss()(logits_enc, labels_enc)


def _build_findings_lines(args, summary):
    best_eval = summary.get("best_private_eval") or {}
    final_eval = summary.get("final_private_eval") or {}
    if args.trainable_scope == "lora":
        experiment_label = "MPC LoRA"
    elif args.trainable_scope == "last_layer_classifier":
        experiment_label = "MPC Last-Layer Classifier"
    else:
        experiment_label = "MPC 无 LoRA"
    if args.private_loss_mode == "manual":
        mode_note = "- 该实验使用 manual private CE，目标是验证正式训练脚本里的 loss 曲线和短程 eval 是否较内置 CE 更合理。"
        root_cause_note = "- 若训练 loss 更平稳下降，且评估指标不再明显异常，可基本确认内置 CE 是正式训练异常的重要根因。"
    else:
        mode_note = "- 该实验使用 CrypTen 内置 CrossEntropyLoss，作为与 manual private CE 的同配置对照。"
        root_cause_note = "- 由于 batch size = 1 时内置 CE 存在已知归一化问题，这组 loss 数值不能直接与 manual 模式按同一尺度比较。"
    if args.softmax_method == "ideal":
        softmax_note = "- 当前实验使用 ideal softmax，作为 masked attention softmax 的参考上界。"
    elif args.softmax_method == "ode" and bool(getattr(cfg.functions, "softmax_ode_center_by_max", True)):
        if bool(getattr(cfg.functions, "softmax_ode_zero_masked", False)):
            softmax_note = "- 当前实验使用 ode softmax，并启用 center-by-max + masked-zero 修复：先按行减去最大值，再做 clip 与 ODE 迭代，最后对明显的 masked 位置清零并重归一化。"
        else:
            softmax_note = "- 当前实验使用 ode softmax，并启用 center-by-max 修复：先按行减去最大值，再做 clip 与 ODE 迭代。"
    else:
        softmax_note = f"- 当前实验使用 {args.softmax_method} softmax。"
    return [
        f"{experiment_label} 短程训练验证结论",
        "",
        f"任务: {args.task_name}",
        f"模型: {args.model_name_or_path}",
        f"private loss 模式: {args.private_loss_mode}",
        f"softmax 模式: {args.softmax_method}",
        f"sqrt 模式: {args.sqrt_method}",
        f"训练步数: {summary['train_steps']}",
        f"eval 间隔: {args.eval_every_steps}",
        f"max_length: {args.max_length}",
        "",
        "训练摘要:",
        f"- 最小 train loss: {summary['train_loss_min']}",
        f"- 最后一个 train loss: {summary['train_loss_last']}",
        f"- 最后一个平滑 train loss: {summary['train_loss_ma_last']}",
        "",
        "评估摘要:",
        f"- 最终 private eval: {final_eval}",
        f"- 最佳 private eval: {best_eval}",
        "",
        "说明:",
        mode_note,
        softmax_note,
        "- 当前脚本也支持切换 sqrt_method；当设为 ideal 时，可作为修正 LayerNorm inv_sqrt / rsqrt 误差的临时工程策略。",
        root_cause_note,
    ]


def _evaluate_private_model(private_model, eval_dataloader, task_name, device, is_regression, max_eval_steps=-1):
    private_model.eval()
    rank = _get_rank()
    metric = evaluate.load("glue", task_name) if task_name is not None else evaluate.load("accuracy")
    total_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(eval_dataloader):
        if max_eval_steps > 0 and step >= max_eval_steps:
            break

        inputs_enc, attention_mask_enc, token_type_enc = _private_batch_inputs(batch, device)
        with ct.no_grad():
            logits_enc = private_model(inputs_enc, attention_mask_enc, token_type_enc)

        logits = logits_enc.get_plain_text().cpu()
        labels = batch["labels"].cpu()

        if is_regression:
            eval_loss = F.mse_loss(logits.squeeze(), labels.float())
            predictions = logits.squeeze()
        else:
            eval_loss = F.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=-1)

        total_loss += eval_loss.item()
        total_steps += 1
        metric.add_batch(predictions=predictions, references=labels)

        if rank == 0 and total_steps % 10 == 0:
            _progress_print(
                f"[private-eval] progress steps={total_steps}/{max_eval_steps if max_eval_steps > 0 else 'full'}"
            )

    metrics = metric.compute() if total_steps > 0 else {"skipped": True, "reason": "no_eval_steps"}
    metrics["eval_loss"] = (total_loss / total_steps) if total_steps > 0 else None
    metrics["eval_steps"] = total_steps
    return metrics


def _plot_training_curves(output_dir, train_history, eval_history, smoothing_window, trainable_scope):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        logger.warning("matplotlib is not installed; skip plotting curves")
        return

    if trainable_scope == "lora":
        train_loss_title = "MPC Train Loss (LoRA)"
    elif trainable_scope == "last_layer_classifier":
        train_loss_title = "MPC Train Loss (Last-Layer Classifier)"
    else:
        train_loss_title = "MPC Train Loss (No LoRA)"

    train_steps = [item["step"] for item in train_history]
    train_losses = [item["loss"] for item in train_history]
    smoothed_losses = _moving_average(train_losses, smoothing_window)
    train_lrs = [item["lr"] for item in train_history]

    figure, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=False)

    axes[0].plot(train_steps, train_losses, alpha=0.35, label="train_loss_raw")
    axes[0].plot(train_steps, smoothed_losses, linewidth=2.0, label=f"train_loss_ma{min(smoothing_window, len(train_losses))}")
    axes[0].set_title(train_loss_title)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(train_steps, train_lrs, label="learning_rate")
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("LR")
    axes[1].grid(True)
    axes[1].legend()

    if eval_history:
        eval_steps = [item["step"] for item in eval_history]
        accuracy_key = next((key for key in eval_history[0].keys() if key not in {"step", "eval_loss", "eval_steps"}), None)
        if accuracy_key is not None:
            axes[2].plot(eval_steps, [item.get(accuracy_key) for item in eval_history], marker="o", label=accuracy_key)
            axes[2].set_title("Private Eval Metric")
            axes[2].set_xlabel("Step")
            axes[2].set_ylabel(accuracy_key)
            axes[2].grid(True)
            axes[2].legend()
        else:
            axes[2].set_visible(False)
    else:
        axes[2].set_visible(False)

    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close(figure)


def _save_training_artifacts(output_dir, train_history, eval_history, smoothing_window, trainable_scope):
    with open(os.path.join(output_dir, "train_history.json"), "w") as file_obj:
        json.dump(train_history, file_obj)
    with open(os.path.join(output_dir, "eval_history.json"), "w") as file_obj:
        json.dump(eval_history, file_obj, indent=2)
    _plot_training_curves(output_dir, train_history, eval_history, smoothing_window, trainable_scope)


def parse_args():
    parser = argparse.ArgumentParser(description="Private MPC full finetuning control experiment with private eval on GLUE")
    parser.add_argument("--task_name", type=str, required=True, choices=list(task_to_keys.keys()))
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_train_steps", type=int, default=6000)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--eval_every_steps", type=int, default=2000)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--max_eval_steps", type=int, default=-1)
    parser.add_argument("--train_max_samples", type=int, default=-1)
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--shuffle_train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--smoothing_window", type=int, default=50)
    parser.add_argument("--trainable_scope", type=str, default="full", choices=["full", "last_layer_classifier", "lora"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="query,value")
    parser.add_argument("--freeze_classifier_head", action="store_true")
    parser.add_argument("--private_loss_mode", type=str, choices=["manual", "standard"], default="manual")
    parser.add_argument("--softmax_method", type=str, choices=["ideal", "ode", "reciprocal"], default="ideal")
    parser.add_argument("--softmax_ode_iter_num", type=int, default=None)
    parser.add_argument("--softmax_ode_clip", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--softmax_ode_center_by_max", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--softmax_ode_zero_masked", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--softmax_ode_mask_margin", type=float, default=None)
    parser.add_argument("--ce_softmax_method", type=str, choices=["ideal", "ode", "reciprocal"], default=None)
    parser.add_argument("--ce_softmax_ode_lb", type=float, default=None)
    parser.add_argument("--sqrt_method", type=str, choices=["NR", "ideal"], default="NR")
    parser.add_argument("--sqrt_nr_iters", type=int, default=None)
    parser.add_argument("--sqrt_nr_initial", type=float, default=None)
    parser.add_argument("--sqrt_nr_initial_exp_iterations", type=int, default=None)
    parser.add_argument("--sqrt_nr_linear_divisor", type=float, default=None)
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--world_size", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    _require_package("evaluate", evaluate, "python -m pip install evaluate")
    cfg.functions.softmax_method = args.softmax_method
    if args.softmax_ode_iter_num is not None:
        cfg.functions.softmax_ode_iter_num = args.softmax_ode_iter_num
    if args.softmax_ode_clip is not None:
        cfg.functions.softmax_ode_clip = args.softmax_ode_clip.lower() == "true"
    if args.softmax_ode_center_by_max is not None:
        cfg.functions.softmax_ode_center_by_max = args.softmax_ode_center_by_max.lower() == "true"
    if args.softmax_ode_zero_masked is not None:
        cfg.functions.softmax_ode_zero_masked = args.softmax_ode_zero_masked.lower() == "true"
    if args.softmax_ode_mask_margin is not None:
        cfg.functions.softmax_ode_mask_margin = args.softmax_ode_mask_margin
    cfg.functions.sqrt_method = args.sqrt_method
    if args.sqrt_nr_iters is not None:
        cfg.functions.sqrt_nr_iters = args.sqrt_nr_iters
    if args.sqrt_nr_initial is not None:
        cfg.functions.sqrt_nr_initial = args.sqrt_nr_initial
    if args.sqrt_nr_initial_exp_iterations is not None:
        cfg.functions.sqrt_nr_initial_exp_iterations = args.sqrt_nr_initial_exp_iterations
    if args.sqrt_nr_linear_divisor is not None:
        cfg.functions.sqrt_nr_linear_divisor = args.sqrt_nr_linear_divisor

    old_recursion_limit = sys.getrecursionlimit()
    target_recursion_limit = max(old_recursion_limit, 20000)
    if target_recursion_limit != old_recursion_limit:
        sys.setrecursionlimit(target_recursion_limit)

    _install_print_filter()

    set_seed(args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    raw_datasets = load_dataset("nyu-mll/glue", args.task_name)
    validation_key = "validation_matched" if args.task_name == "mnli" else "validation"

    if args.train_max_samples > 0:
        raw_datasets["train"] = raw_datasets["train"].select(range(min(args.train_max_samples, len(raw_datasets["train"])) ))
    if args.eval_max_samples > 0:
        raw_datasets[validation_key] = raw_datasets[validation_key].select(
            range(min(args.eval_max_samples, len(raw_datasets[validation_key])))
        )

    is_regression = args.task_name == "stsb"
    num_labels = 1 if is_regression else len(raw_datasets[validation_key].features["label"].names)
    label_list = None if is_regression else raw_datasets[validation_key].features["label"].names

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    label_to_id = None
    if not is_regression and model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        label_name_to_id = {key.lower(): value for key, value in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {index: label_name_to_id[label_list[index]] for index in range(num_labels)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {idx: label for label, idx in config.label2id.items()}
    elif not is_regression:
        model.config.label2id = {label: idx for idx, label in enumerate(label_list)}
        model.config.id2label = {idx: label for label, idx in model.config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        encoded = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        if label_to_id is not None:
            encoded["labels"] = [label_to_id[label] for label in examples["label"]]
        else:
            encoded["labels"] = examples["label"]
        return encoded

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets[validation_key]
    data_collator = default_data_collator if args.pad_to_max_length else DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=args.shuffle_train,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    if not ct.is_initialized():
        ct.init()
    rank = _get_rank()

    if rank == 0:
        _progress_print(f"[env] python recursion limit: {old_recursion_limit} -> {sys.getrecursionlimit()}")
        _progress_print(f"[env] crypten module path: {getattr(ct, '__file__', 'unknown')}")

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count < 1:
        raise RuntimeError("CUDA is required for MPC full finetuning control experiment")

    gpu_id_list = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    local_gpu_id = gpu_id_list[rank % len(gpu_id_list)]
    torch.cuda.set_device(local_gpu_id)
    device = f"cuda:{local_gpu_id}"

    dummy = torch.zeros_like(model.dummy_inputs["input_ids"])
    private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy))

    lora_replaced_modules = []
    trainable_names = []
    if args.trainable_scope == "lora":
        if float(args.lora_dropout) != 0.0:
            raise ValueError("Current MPCLoRALinear only supports --lora_dropout 0.0.")
        torch.manual_seed(args.seed)
        target_keywords = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
        lora_replaced_modules = _inject_lora_layers(
            model,
            target_keywords=target_keywords,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        if not lora_replaced_modules:
            raise ValueError(f"No Linear layers matched --lora_target_modules='{args.lora_target_modules}'.")
        private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy))
        collapsed_lora_modules = _collapse_lora_graph_modules(private_model, model, lora_replaced_modules)
        if len(collapsed_lora_modules) != len(lora_replaced_modules):
            raise RuntimeError(
                f"Only collapsed {len(collapsed_lora_modules)} / {len(lora_replaced_modules)} LoRA modules into MPCLoRALinear."
            )
        _apply_trainable_scope(private_model, args.trainable_scope)
        trainable_names = _set_lora_trainable(private_model, train_classifier_head=(not args.freeze_classifier_head))
        if not trainable_names:
            raise RuntimeError("No trainable parameters after LoRA setup")
    else:
        _apply_trainable_scope(model, args.trainable_scope)
        private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy))
        _apply_trainable_scope(private_model, args.trainable_scope)
        trainable_names = [
            name for name, param in private_model.named_parameters() if getattr(param, "requires_grad", False)
        ]

    private_model = private_model.encrypt().to(device)
    private_model.train()

    trainable_params = [param for param in private_model.parameters() if getattr(param, "requires_grad", False)]
    optimizer = _build_private_optimizer(args, trainable_params)

    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(args.max_train_steps * args.warmup_ratio)
    warmup_steps = min(warmup_steps, args.max_train_steps)

    if rank == 0:
        _progress_print(f"[setup] device={device}")
        _progress_print(f"[setup] train_dataset={len(train_dataset)} eval_dataset={len(eval_dataset)}")
        _progress_print(f"[setup] shuffle_train={args.shuffle_train}")
        _progress_print(
            "[setup] optimizer={} learning_rate={} momentum={} weight_decay={} adam_beta1={} adam_beta2={} adam_epsilon={} warmup_steps={} total_train_steps={}".format(
                args.optimizer,
                args.learning_rate,
                args.momentum,
                args.weight_decay,
                args.adam_beta1,
                args.adam_beta2,
                args.adam_epsilon,
                warmup_steps,
                args.max_train_steps,
            )
        )
        _progress_print(
            f"[setup] trainable_scope={args.trainable_scope} trainable_param_count={_count_trainable_params(private_model)} eval_every_steps={args.eval_every_steps}"
        )
        if args.trainable_scope == "lora":
            _progress_print(
                f"[setup] lora_r={args.lora_r} lora_alpha={args.lora_alpha} lora_dropout={args.lora_dropout} freeze_classifier_head={args.freeze_classifier_head} replaced_modules={len(lora_replaced_modules)} first={lora_replaced_modules[:12]} trainable_name_count={len(trainable_names)}"
            )
        _progress_print(f"[setup] skip_eval={args.skip_eval}")
        _progress_print(f"[setup] private_loss_mode={args.private_loss_mode}")
        _progress_print(
            "[setup] softmax_method={} ode_iter={} ode_clip={} ode_center_by_max={} ode_zero_masked={} ode_mask_margin={} ce_softmax_method={} ce_softmax_ode_lb={} sqrt_method={} sqrt_nr_iters={} sqrt_nr_initial_exp_iterations={} sqrt_nr_linear_divisor={}".format(
                args.softmax_method,
                getattr(cfg.functions, "softmax_ode_iter_num", None),
                getattr(cfg.functions, "softmax_ode_clip", None),
                getattr(cfg.functions, "softmax_ode_center_by_max", None),
                getattr(cfg.functions, "softmax_ode_zero_masked", None),
                getattr(cfg.functions, "softmax_ode_mask_margin", None),
                args.ce_softmax_method,
                args.ce_softmax_ode_lb,
                args.sqrt_method,
                cfg.functions.sqrt_nr_iters,
                getattr(cfg.functions, "sqrt_nr_initial_exp_iterations", None),
                getattr(cfg.functions, "sqrt_nr_linear_divisor", None),
            )
        )

    train_history = []
    eval_history = []
    global_step = 0
    train_start_time = time.time()
    stop_training = False

    while not stop_training:
        for batch in train_dataloader:
            private_model.train()
            current_step = global_step + 1
            current_lr = _compute_linear_warmup_lr(current_step, args.max_train_steps, args.learning_rate, warmup_steps)
            _set_optimizer_lr(optimizer, current_lr)

            optimizer.zero_grad()
            inputs_enc, attention_mask_enc, token_type_enc = _private_batch_inputs(batch, device)
            logits_enc = private_model(inputs_enc, attention_mask_enc, token_type_enc)

            loss_enc = _compute_private_train_loss(
                logits_enc,
                batch["labels"],
                device,
                args.private_loss_mode,
                ce_softmax_method=args.ce_softmax_method,
                ce_softmax_ode_lb=args.ce_softmax_ode_lb,
            )

            loss_enc.backward()
            optimizer.step()

            loss_value = float(loss_enc.get_plain_text().item())
            global_step = current_step
            train_history.append({"step": global_step, "loss": loss_value, "lr": current_lr})

            if rank == 0 and global_step % max(1, args.log_every_steps) == 0:
                _save_training_artifacts(args.output_dir, train_history, eval_history, args.smoothing_window, args.trainable_scope)
                _progress_print(
                    f"[train] step={global_step} loss={loss_value:.6f} lr={current_lr:.8f} input_shape={_shape_of(batch['input_ids'])}"
                )

            if (not args.skip_eval) and global_step % max(1, args.eval_every_steps) == 0:
                eval_result = _evaluate_private_model(
                    private_model,
                    eval_dataloader,
                    args.task_name,
                    device,
                    is_regression,
                    args.max_eval_steps,
                )
                eval_result["step"] = global_step
                eval_history.append(eval_result)
                if rank == 0:
                    _save_training_artifacts(args.output_dir, train_history, eval_history, args.smoothing_window, args.trainable_scope)
                    _progress_print(f"[private-eval] step={global_step} metrics={eval_result}")

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop_training = True
                break

        if len(train_dataloader) == 0:
            break

    if (not args.skip_eval) and (not eval_history or eval_history[-1]["step"] != global_step):
        final_eval = _evaluate_private_model(
            private_model,
            eval_dataloader,
            args.task_name,
            device,
            is_regression,
            args.max_eval_steps,
        )
        final_eval["step"] = global_step
        eval_history.append(final_eval)
        if rank == 0:
            _save_training_artifacts(args.output_dir, train_history, eval_history, args.smoothing_window, args.trainable_scope)
            _progress_print(f"[private-eval] final step={global_step} metrics={final_eval}")

    if rank == 0:
        train_losses = [item["loss"] for item in train_history]
        metric_keys = [key for key in eval_history[0].keys() if key not in {"step", "eval_loss", "eval_steps"}] if eval_history else []
        primary_metric_key = metric_keys[0] if metric_keys else None
        best_eval = max(eval_history, key=lambda item: item.get(primary_metric_key, float("-inf"))) if primary_metric_key else None
        final_eval = eval_history[-1] if eval_history else None

        # STRUCTURAL-FB-REUSE: capture LoRA reuse capability and counters before
        # the MPC worker process exits; these counters live only in-process.
        try:
            from crypten.gradients import (
                get_structural_reuse_stats,
                structural_reuse_capability,
            )

            structural_reuse = {
                "available": True,
                "capability": structural_reuse_capability(),
                "stats": get_structural_reuse_stats(reset=False),
            }
        except Exception as exc:
            structural_reuse = {
                "available": False,
                "error": repr(exc),
            }

        summary = {
            "task_name": args.task_name,
            "model_name_or_path": args.model_name_or_path,
            "train_steps": global_step,
            "train_dataset_size": len(train_dataset),
            "eval_dataset_size": len(eval_dataset),
            "shuffle_train": args.shuffle_train,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon,
            "warmup_steps": warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "train_batch_size": args.per_device_train_batch_size,
            "eval_batch_size": args.per_device_eval_batch_size,
            "eval_every_steps": args.eval_every_steps,
            "max_eval_steps": args.max_eval_steps,
            "max_length": args.max_length,
            "final_learning_rate": train_history[-1]["lr"] if train_history else None,
            "train_loss_min": min(train_losses) if train_losses else None,
            "train_loss_max": max(train_losses) if train_losses else None,
            "train_loss_last": train_losses[-1] if train_losses else None,
            "train_loss_ma_last": _moving_average(train_losses, args.smoothing_window)[-1] if train_losses else None,
            "best_private_eval": best_eval,
            "final_private_eval": final_eval,
            "elapsed_seconds": time.time() - train_start_time,
            "train_loss_type": "manual_one_hot_cross_entropy" if args.private_loss_mode == "manual" else "cross_entropy_one_hot",
            "private_loss_mode": args.private_loss_mode,
            "softmax_method": args.softmax_method,
            "softmax_ode_iter_num": getattr(cfg.functions, "softmax_ode_iter_num", None),
            "softmax_ode_clip": getattr(cfg.functions, "softmax_ode_clip", None),
            "softmax_ode_center_by_max": getattr(cfg.functions, "softmax_ode_center_by_max", None),
            "softmax_ode_zero_masked": getattr(cfg.functions, "softmax_ode_zero_masked", None),
            "softmax_ode_mask_margin": getattr(cfg.functions, "softmax_ode_mask_margin", None),
            "ce_softmax_method": args.ce_softmax_method,
            "ce_softmax_ode_lb": args.ce_softmax_ode_lb,
            "sqrt_method": args.sqrt_method,
            "sqrt_nr_iters": getattr(cfg.functions, "sqrt_nr_iters", None),
            "sqrt_nr_initial_exp_iterations": getattr(cfg.functions, "sqrt_nr_initial_exp_iterations", None),
            "sqrt_nr_linear_divisor": getattr(cfg.functions, "sqrt_nr_linear_divisor", None),
            "full_finetune": args.trainable_scope == "full",
            "trainable_scope": args.trainable_scope,
            "lora_enabled": args.trainable_scope == "lora",
            "lora_r": args.lora_r if args.trainable_scope == "lora" else None,
            "lora_alpha": args.lora_alpha if args.trainable_scope == "lora" else None,
            "lora_dropout": args.lora_dropout if args.trainable_scope == "lora" else None,
            "lora_target_modules": args.lora_target_modules if args.trainable_scope == "lora" else None,
            "freeze_classifier_head": args.freeze_classifier_head if args.trainable_scope == "lora" else None,
            "lora_replaced_module_count": len(lora_replaced_modules),
            "lora_replaced_modules": lora_replaced_modules,
            "private_eval_during_training": not args.skip_eval,
            "structural_reuse": structural_reuse,
        }

        _save_training_artifacts(args.output_dir, train_history, eval_history, args.smoothing_window, args.trainable_scope)
        with open(os.path.join(args.output_dir, "summary.json"), "w") as file_obj:
            json.dump(summary, file_obj, indent=2)
        with open(os.path.join(args.output_dir, "findings.txt"), "w") as file_obj:
            file_obj.write("\n".join(_build_findings_lines(args, summary)) + "\n")

        _progress_print(f"[summary] {summary}")
        _progress_print(f"[done] artifacts saved under {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    with cfg.temp_override({"cost.estimate_cost": False}):
        launcher = MultiProcessLauncher(args.world_size, main)
        launcher.start()
        launcher.join()
        launcher.terminate()
