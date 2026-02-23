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
import json
import logging
import os
import sys
import time

import datasets
import evaluate
import torch
import torch.nn.functional as F
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
)

from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import crypten as ct
from crypten.config import cfg
from multiprocess_launcher import MultiProcessLauncher


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


def _safe_metric_compute(metric, steps, rank, phase):
    if steps <= 0:
        logger.warning(
            "[rank %s] %s metric skipped: no batches were added (likely filtered by --len_data).",
            rank,
            phase,
        )
        return {"skipped": True, "reason": "no_batches", "steps": steps}
    return metric.compute()


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
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_private", args)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # CrypTen autograd walks graph recursively; deep graphs can exceed Python's default recursion limit (1000).
    old_recursion_limit = sys.getrecursionlimit()
    target_recursion_limit = max(old_recursion_limit, 20000)
    if target_recursion_limit != old_recursion_limit:
        sys.setrecursionlimit(target_recursion_limit)
    logger.info("train-smoke start pid=%s argv=%s", os.getpid(), " ".join(sys.argv))
    logger.info("python recursion limit %s -> %s", old_recursion_limit, sys.getrecursionlimit())
    logger.info("initial cfg snapshot=%s", _cfg_snapshot())


    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    
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
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)

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
    # print("done")
    # exit()
    dummy = torch.zeros_like(model.dummy_inputs["input_ids"])
    private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).encrypt().to(device)
    private_model.train()
    lr = 0.01
    optimizer = ct.optim.SGD(private_model.parameters(), lr=lr)
    logger.info("[rank %s] model set to train mode; optimizer initialized (lr=%s)", rank, lr)
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
    logger.info(
        "[rank %s] entering short-train loop max_train_steps=%s train_batch=%s",
        rank,
        args.max_train_steps,
        args.per_device_train_batch_size,
    )
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
        inputs_enc = ct.cryptensor(batch["input_ids"]).to(device)
        attention_mask_enc = ct.cryptensor(batch["attention_mask"]).to(device)
        token_type_enc = ct.cryptensor(token_type_ids).to(device)

        # forward (NO ct.no_grad for training)
        # 在不设置ct.no_grad时自动默认训练模式，forward过程会记录backward所需结果
        forward_start = time.time()
        logits_enc = private_model(inputs_enc, attention_mask_enc, token_type_enc)  # [B, num_labels]
        logger.info(
            "[rank %s] train_step=%03d forward_done dt=%.3fs logits_shape=%s",
            rank,
            global_step,
            time.time() - forward_start,
            _shape_of(logits_enc),
        )

        # MSE loss with one-hot labels (more stable than CE for MPC smoke test)
        num_labels = logits_enc.size(-1)
        y_onehot = F.one_hot(batch["labels"], num_classes=num_labels).float()
        y_enc = ct.cryptensor(y_onehot).to(device)

        diff = logits_enc - y_enc
        loss_enc = (diff * diff).mean()

        # optimizer (create once on first step)
        logger.info("[rank %s] train_step=%03d loss_snapshot=%s", rank, global_step, _loss_snapshot(loss_enc))

        optimizer.zero_grad()
        
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
        try:
            loss_enc.backward()
        except Exception:
            logger.exception(
                "[rank %s] train_step=%03d backward_failed cfg=%s loss=%s",
                rank,
                global_step,
                _cfg_snapshot(),
                _loss_snapshot(loss_enc),
            )
            raise
        logger.info("[rank %s] train_step=%03d backward_done", rank, global_step)

        try:
            optimizer.step()
        except Exception:
            logger.exception("[rank %s] train_step=%03d optimizer_step_failed", rank, global_step)
            raise
        logger.info("[rank %s] train_step=%03d optimizer_step_done", rank, global_step)

        # reveal loss (ALL ranks must call get_plain_text / reveal)
        loss_plain = loss_enc.get_plain_text().item()
        global_step += 1
        if global_step % max(1, args.log_every_steps) == 0:
            logger.info("[rank %s] [train] step=%03d loss=%.6f", rank, global_step, loss_plain)
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            logger.info("[rank %s] reached max_train_steps=%s", rank, args.max_train_steps)
            break
    logger.info(
        "[rank %s] short-train finished steps=%s elapsed=%.3fs",
        rank,
        global_step,
        time.time() - train_start_time,
    )

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

    # Step 3B: recover a plaintext model and run plaintext evaluation.
    private_model.decrypt()
    trained_model = private_model.to_pytorch().to(device)
    trained_model.eval()

    plain_eval_metric = None
    plain_steps = 0
    plain_skipped_by_len = 0
    if rank == 0:
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

    if rank == 0 and args.output_dir is not None:
        trained_model_dir = os.path.join(args.output_dir, "trained_model")
        os.makedirs(trained_model_dir, exist_ok=True)
        trained_model.save_pretrained(trained_model_dir)
        tokenizer.save_pretrained(trained_model_dir)

        summary = {
            "train_steps": global_step,
            "private_eval_metric": private_eval_metric,
            "plain_eval_metric": plain_eval_metric,
            "task_name": args.task_name,
            "max_train_steps": args.max_train_steps,
            "eval_max_steps": args.eval_max_steps,
        }
        summary_path = os.path.join(args.output_dir, "train_eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("[save] trained model saved to %s", trained_model_dir)
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
        with cfg.temp_override({"cost.estimate_cost": True, "cost.estimate_mode": "comm"}):

            launcher = MultiProcessLauncher(2, main)
            launcher.start()
            launcher.join()
            launcher.terminate()
