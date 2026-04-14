#!/usr/bin/env python3

"""Thin plaintext LoRA wrapper for the minimal LoRA / LoRA-XS comparison."""

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_glue_plain_low_rank_train as base


def _contains_cli_arg(argv, flag_name):
    return any(argument == flag_name or argument.startswith(f"{flag_name}=") for argument in argv)


def _inject_default_argv(argv):
    patched = list(argv)
    if not _contains_cli_arg(patched, "--adapter_type"):
        patched.extend(["--adapter_type", "lora"])
    if not _contains_cli_arg(patched, "--model_name_or_path"):
        patched.extend(["--model_name_or_path", "bert-base-uncased"])
    if not _contains_cli_arg(patched, "--task_name"):
        patched.extend(["--task_name", "sst2"])
    if not _contains_cli_arg(patched, "--lora_target_modules"):
        patched.extend(["--lora_target_modules", "query,key,value"])
    if not _contains_cli_arg(patched, "--lora_dropout"):
        patched.extend(["--lora_dropout", "0.0"])
    return patched


def main():
    original_argv = list(sys.argv)
    sys.argv = [original_argv[0]] + _inject_default_argv(original_argv[1:])
    try:
        base.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
