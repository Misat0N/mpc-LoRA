#!/usr/bin/env python3

"""
Plaintext GLUE finetuning for custom low-rank adapters.

This mirrors the repository's custom LoRA / LoRA-XS layer implementations so
the user can compare:
- Plaintext LoRA
- Plaintext LoRA-XS
- MPC LoRA
- MPC LoRA-XS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
    set_seed,
)

from loraxs_public_layers import inject_loraxs_layers
from run_glue_private_mpc_lora_train import (
    _count_param_numel,
    _inject_lora_layers,
    task_to_keys,
)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Plaintext GLUE low-rank finetuning")
    parser.add_argument("--adapter_type", type=str, choices=["lora", "loraxs"], required=True)
    parser.add_argument("--task_name", type=str, default="sst2", choices=list(task_to_keys.keys()))
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--len_data", type=int, default=64)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--train_max_samples", type=int, default=-1)
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--max_train_steps", type=int, default=300)
    parser.add_argument("--log_every_steps", type=int, default=5)
    parser.add_argument("--eval_max_steps", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="query,key,value")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--classifier_learning_rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--ignore_mismatched_sizes", action="store_true")
    parser.add_argument("--skip_model_save", action="store_true")
    return parser.parse_args()


def _pick_device(gpu_ids: str) -> torch.device:
    if torch.cuda.is_available():
        first_gpu = str(gpu_ids).split(",")[0].strip()
        if first_gpu:
            return torch.device(f"cuda:{first_gpu}")
        return torch.device("cuda")
    return torch.device("cpu")


def _set_loraxs_trainable(model):
    trainable = []
    for name, param in model.named_parameters():
        is_loraxs = "lora_latent." in name
        is_classifier = name.startswith("classifier.") or name.startswith("score.")
        param.requires_grad = is_loraxs or is_classifier
        if param.requires_grad:
            trainable.append(name)
    return trainable


def _build_optimizer(model, args):
    classifier_lr = args.classifier_learning_rate
    if classifier_lr is None or classifier_lr <= 0:
        trainable_params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        return optimizer, {
            "optimizer_type": "torch_sgd_single_lr",
            "learning_rate": args.learning_rate,
            "classifier_learning_rate": None,
            "num_trainable_tensors": len(trainable_params),
            "num_trainable_parameters": _count_param_numel(trainable_params),
        }

    classifier_params = []
    non_classifier_params = []
    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        if name.startswith("classifier.") or name.startswith("score."):
            classifier_params.append(param)
        else:
            non_classifier_params.append(param)

    param_groups = []
    if non_classifier_params:
        param_groups.append({"params": non_classifier_params, "lr": args.learning_rate})
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": classifier_lr})

    optimizer = torch.optim.SGD(
        param_groups,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    return optimizer, {
        "optimizer_type": "torch_sgd_grouped_lr",
        "learning_rate": args.learning_rate,
        "classifier_learning_rate": classifier_lr,
        "num_non_classifier_tensors": len(non_classifier_params),
        "num_non_classifier_parameters": _count_param_numel(non_classifier_params),
        "num_classifier_tensors": len(classifier_params),
        "num_classifier_parameters": _count_param_numel(classifier_params),
    }


def _preprocess_datasets(args, tokenizer):
    raw_datasets = load_dataset("nyu-mll/glue", args.task_name)
    is_regression = args.task_name == "stsb"
    validation_key = "validation_matched" if args.task_name == "mnli" else "validation"

    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        label_list = None
        num_labels = 1

    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        result["labels"] = examples["label"]
        return result

    processed = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed["train"]
    eval_dataset = processed[validation_key]
    if args.train_max_samples > 0:
        train_dataset = train_dataset.select(range(min(args.train_max_samples, len(train_dataset))))
    if args.eval_max_samples > 0:
        eval_dataset = eval_dataset.select(range(min(args.eval_max_samples, len(eval_dataset))))

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    return train_dataset, eval_dataset, data_collator, num_labels, label_list, is_regression


def _inject_low_rank_layers(model, args):
    targets = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    if args.adapter_type == "lora":
        replaced = _inject_lora_layers(
            model,
            target_keywords=targets,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        trainable_names = []
        for name, param in model.named_parameters():
            is_lora = ("lora_A." in name) or ("lora_B." in name)
            is_classifier = name.startswith("classifier.") or name.startswith("score.")
            param.requires_grad = is_lora or is_classifier
            if param.requires_grad:
                trainable_names.append(name)
        return replaced, trainable_names

    replaced = inject_loraxs_layers(
        model,
        target_keywords=targets,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    trainable_names = _set_loraxs_trainable(model)
    return replaced, trainable_names


def _evaluate_model(model, eval_dataloader, device, args, is_regression):
    metric = evaluate.load("glue", args.task_name) if args.task_name else evaluate.load("accuracy")
    model.eval()
    skipped_by_len = 0
    steps = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            if args.len_data > 0 and batch["input_ids"].shape[1] != args.len_data:
                skipped_by_len += 1
                continue

            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["labels"],
            )
            predictions = outputs.logits.squeeze() if is_regression else outputs.logits.argmax(dim=-1)
            metric.add_batch(predictions=predictions.detach().cpu(), references=batch["labels"].detach().cpu())
            steps += 1
            if args.eval_max_steps > 0 and steps >= args.eval_max_steps:
                break

    return {
        "steps": steps,
        "skipped_by_len": skipped_by_len,
        "metric": metric.compute() if steps > 0 else {"skipped": True},
    }


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    set_seed(args.seed)
    device = _pick_device(args.gpu_ids)
    script_start_time = time.time()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(
        "[plain-%s] start task=%s model=%s device=%s steps=%s targets=%s",
        args.adapter_type,
        args.task_name,
        args.model_name_or_path,
        device,
        args.max_train_steps,
        args.lora_target_modules,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    train_dataset, eval_dataset, data_collator, num_labels, label_list, is_regression = _preprocess_datasets(
        args, tokenizer
    )

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    if not is_regression and label_list is not None:
        model.config.label2id = {label: idx for idx, label in enumerate(label_list)}
        model.config.id2label = {idx: label for idx, label in enumerate(label_list)}
    else:
        model.config.label2id = config.label2id or PretrainedConfig(num_labels=num_labels).label2id
        model.config.id2label = config.id2label or PretrainedConfig(num_labels=num_labels).id2label

    replaced_layers, trainable_names = _inject_low_rank_layers(model, args)
    model.to(device)

    optimizer, optimizer_summary = _build_optimizer(model, args)
    trainable_params = [param for param in model.parameters() if getattr(param, "requires_grad", False)]
    logger.info(
        "[plain-%s] replaced_layers=%s trainable_tensors=%s trainable_parameters=%s optimizer=%s",
        args.adapter_type,
        len(replaced_layers),
        len(trainable_params),
        _count_param_numel(trainable_params),
        optimizer_summary,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    global_step = 0
    recent_losses = []
    model.train()

    while global_step < args.max_train_steps:
        exhausted = True
        for batch in train_dataloader:
            exhausted = False
            if args.len_data > 0 and batch["input_ids"].shape[1] != args.len_data:
                continue

            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_value = float(loss.detach().item())
            recent_losses.append(loss_value)
            if len(recent_losses) > 8:
                recent_losses.pop(0)

            if global_step % max(1, args.log_every_steps) == 0:
                running_loss = sum(recent_losses) / len(recent_losses)
                logger.info(
                    "[plain-%s] [train] step=%03d loss=%.6f running_loss(window=%s)=%.6f",
                    args.adapter_type,
                    global_step,
                    loss_value,
                    len(recent_losses),
                    running_loss,
                )

            if global_step >= args.max_train_steps:
                logger.info("[plain-%s] reached max_train_steps=%s", args.adapter_type, args.max_train_steps)
                break

        if exhausted:
            break

    eval_summary = _evaluate_model(model, eval_dataloader, device, args, is_regression)
    logger.info(
        "[plain-%s] [eval] steps=%s skipped_by_len=%s metric=%s",
        args.adapter_type,
        eval_summary["steps"],
        eval_summary["skipped_by_len"],
        eval_summary["metric"],
    )

    trained_model_dir = None
    if args.output_dir is not None and not args.skip_model_save:
        trained_model_dir = os.path.join(args.output_dir, "trained_model")
        os.makedirs(trained_model_dir, exist_ok=True)
        model.save_pretrained(trained_model_dir)
        tokenizer.save_pretrained(trained_model_dir)
        logger.info("[plain-%s] [save] trained model saved to %s", args.adapter_type, trained_model_dir)

    total_elapsed_s = time.time() - script_start_time
    summary = {
        "adapter_type": args.adapter_type,
        "task_name": args.task_name,
        "model_name_or_path": args.model_name_or_path,
        "train_steps": global_step,
        "eval_metric": eval_summary["metric"],
        "plain_eval_metric": eval_summary["metric"],
        "private_eval_metric": None,
        "eval_steps": eval_summary["steps"],
        "eval_skipped_by_len": eval_summary["skipped_by_len"],
        "learning_rate": args.learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
        "optimizer_summary": optimizer_summary,
        "lora_target_modules": args.lora_target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_param_summary": {
            "num_trainable_tensors": len(trainable_params),
            "num_trainable_parameters": _count_param_numel(trainable_params),
            "trainable_preview": trainable_names[:24],
            "num_replaced_layers": len(replaced_layers),
        },
        "output_dir": args.output_dir,
        "trained_model_dir": trained_model_dir,
        "total_elapsed_s": total_elapsed_s,
    }

    if args.output_dir is not None:
        summary_path = os.path.join(args.output_dir, "train_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("[plain-%s] [save] summary saved to %s", args.adapter_type, summary_path)

    logger.info("[plain-%s] total elapsed=%.3fs", args.adapter_type, total_elapsed_s)


if __name__ == "__main__":
    main()
