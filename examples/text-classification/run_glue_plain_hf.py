#!/usr/bin/env python3

"""
Plaintext GLUE baseline wrapper.

This is intentionally a thin adapter over the local Hugging Face
`run_glue_no_trainer.py` example so the repository keeps using a standard,
well-understood training path for the plaintext baseline.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "transformers" / "src"
HF_RUNNER_PATH = (
    REPO_ROOT / "transformers" / "examples" / "pytorch" / "text-classification" / "run_glue_no_trainer.py"
)

if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))


def _load_hf_runner():
    spec = importlib.util.spec_from_file_location("hf_run_glue_no_trainer", HF_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Plaintext Hugging Face GLUE baseline")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--task_name",
        type=str,
        default="sst2",
        choices=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
    )
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--batch_size", "--per_device_train_batch_size", dest="batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", "--per_device_eval_batch_size", dest="eval_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--ignore_mismatched_sizes", action="store_true")
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--trust_remote_code", action="store_true")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def _build_hf_argv(args, passthrough):
    output_dir = args.output_dir or os.path.join("eval_plain_hf", args.task_name)
    eval_batch_size = args.eval_batch_size or args.batch_size

    argv = [
        "--model_name_or_path",
        args.model_name_or_path,
        "--max_length",
        str(args.max_length),
        "--per_device_train_batch_size",
        str(args.batch_size),
        "--per_device_eval_batch_size",
        str(eval_batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--lr_scheduler_type",
        str(args.lr_scheduler_type),
        "--num_warmup_steps",
        str(args.num_warmup_steps),
        "--seed",
        str(args.seed),
        "--output_dir",
        output_dir,
    ]

    if args.task_name is not None:
        argv.extend(["--task_name", args.task_name])
    if args.train_file is not None:
        argv.extend(["--train_file", args.train_file])
    if args.validation_file is not None:
        argv.extend(["--validation_file", args.validation_file])
    if args.max_train_steps is not None:
        argv.extend(["--max_train_steps", str(args.max_train_steps)])
    if args.pad_to_max_length:
        argv.append("--pad_to_max_length")
    if args.use_slow_tokenizer:
        argv.append("--use_slow_tokenizer")
    if args.ignore_mismatched_sizes:
        argv.append("--ignore_mismatched_sizes")
    if args.with_tracking:
        argv.append("--with_tracking")
        argv.extend(["--report_to", args.report_to])
    if args.trust_remote_code:
        argv.extend(["--trust_remote_code", "True"])

    argv.extend(passthrough)
    return argv


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args, passthrough = parse_args()
    hf_runner = _load_hf_runner()
    forwarded_argv = _build_hf_argv(args, passthrough)

    original_argv = list(sys.argv)
    sys.argv = [str(HF_RUNNER_PATH)] + forwarded_argv
    try:
        hf_runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
