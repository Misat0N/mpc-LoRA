#!/usr/bin/env python3

"""Aligned full-private LoRA entrypoint.

This wrapper keeps the encrypted standard-LoRA/reuse training path from
``run_glue_private_mpc_lora_train.py`` while injecting the BERT-tiny SST-2
finetuning defaults used by the private LoRA control script.
"""

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_glue_private_mpc_lora_train as base


_ORIGINAL_PARSE_ARGS = base.parse_args


def _contains_cli_arg(argv, flag_name):
    return any(argument == flag_name or argument.startswith(f"{flag_name}=") for argument in argv)


def _append_default(patched, flag_name, value):
    if not _contains_cli_arg(patched, flag_name):
        patched.extend([flag_name, str(value)])


def _append_default_bool(patched, flag_name):
    if not _contains_cli_arg(patched, flag_name):
        patched.append(flag_name)


def _inject_default_argv(argv):
    patched = list(argv)

    scalar_defaults = {
        "--model_name_or_path": "prajjwal1/bert-tiny",
        "--task_name": "sst2",
        "--max_length": 64,
        "--train_max_samples": 10000,
        "--eval_max_samples": -1,
        "--max_train_steps": 1250,
        "--eval_every_steps": 2000,
        "--eval_max_steps": -1,
        "--log_every_steps": 10,
        "--per_device_train_batch_size": 8,
        "--per_device_eval_batch_size": 8,
        "--optimizer": "sgd",
        "--learning_rate": 0.003,
        "--momentum": 0.9,
        "--weight_decay": 0.0,
        "--lr_scheduler_type": "linear",
        "--warmup_ratio": 0.06,
        "--trainable_scope": "lora",
        "--lora_r": 8,
        "--lora_alpha": 16,
        "--lora_dropout": 0.0,
        "--lora_target_modules": "query,value",
        "--private_loss_mode": "manual",
        "--softmax_method": "ode",
        "--softmax_ode_iter_num": 16,
        "--softmax_ode_iter_source": "fixed",
        "--softmax_ode_clip": "true",
        "--softmax_ode_center_by_max": "true",
        "--softmax_ode_zero_masked": "true",
        "--softmax_ode_mask_margin": 100.0,
        "--ce_softmax_method": "ideal",
        "--sqrt_method": "NR",
        "--sqrt_nr_iters": 10,
        "--sqrt_nr_initial_exp_iterations": 7,
        "--sqrt_nr_linear_divisor": 1536.0,
        "--reuse_mode": "SHARED_LEFT",
        "--shared_left_min_fanout": 2,
        "--shared_left_log_groups": 12,
        "--world_size": 2,
    }
    for flag_name, value in scalar_defaults.items():
        _append_default(patched, flag_name, value)

    if not _contains_cli_arg(patched, "--shuffle_train") and not _contains_cli_arg(patched, "--no_shuffle_train"):
        patched.append("--shuffle_train")
    _append_default_bool(patched, "--experimental_reuse_mask")
    _append_default_bool(patched, "--skip_plain_eval")
    return patched


def _patched_parse_args():
    original_argv = list(sys.argv)
    sys.argv = [original_argv[0]] + _inject_default_argv(original_argv[1:])
    try:
        return _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv


base.parse_args = _patched_parse_args


def main():
    args = base.parse_args()
    if args.comp:
        with base.cfg.temp_override({"cost.estimate_cost": True, "cost.estimate_mode": "comp"}):
            base.main()
    elif args.acc:
        with base.cfg.temp_override({"cost.estimate_cost": False}):
            base.main()
    else:
        with base.cfg.temp_override({"cost.estimate_cost": args.print_comm_cost, "cost.estimate_mode": "comm"}):
            launcher = base.MultiProcessLauncher(args.world_size, base.main)
            launcher.start()
            launcher.join()
            launcher.terminate()


if __name__ == "__main__":
    main()
