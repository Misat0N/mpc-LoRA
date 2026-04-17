#!/usr/bin/env python3

"""
Shared-U qkv LoRA-XS + shared-left private finetuning entrypoint.
"""

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_glue_private_mpc_lora_train as base
from loraxs_public_layers import LoRAXSPublicLinear
from loraxs_shared_u_qkv_layers import inject_shared_u_qkv_loraxs_layers


_ORIGINAL_PARSE_ARGS = base.parse_args


def _contains_cli_arg(argv, flag_name):
    return any(argument == flag_name or argument.startswith(f"{flag_name}=") for argument in argv)


def _inject_default_argv(argv):
    patched = list(argv)

    if not _contains_cli_arg(patched, "--model_name_or_path"):
        patched.extend(["--model_name_or_path", "bert-base-uncased"])
    if not _contains_cli_arg(patched, "--task_name"):
        patched.extend(["--task_name", "sst2"])
    if not _contains_cli_arg(patched, "--adapter_type_label"):
        patched.extend(["--adapter_type_label", "loraxs_shared_u_qkv"])
    if not _contains_cli_arg(patched, "--lora_target_modules"):
        patched.extend(["--lora_target_modules", "query,key,value"])
    if not _contains_cli_arg(patched, "--lora_dropout"):
        patched.extend(["--lora_dropout", "0.0"])
    if not _contains_cli_arg(patched, "--public_non_lora_weights"):
        patched.append("--public_non_lora_weights")
    if not _contains_cli_arg(patched, "--encrypted_param_keywords"):
        patched.extend(["--encrypted_param_keywords", "lora_latent.,classifier.,score."])
    if not _contains_cli_arg(patched, "--reuse_mode"):
        patched.extend(["--reuse_mode", "SHARED_LEFT"])
    if not _contains_cli_arg(patched, "--shared_left_min_fanout"):
        patched.extend(["--shared_left_min_fanout", "3"])
    if not _contains_cli_arg(patched, "--shared_left_log_groups"):
        patched.extend(["--shared_left_log_groups", "12"])
    if not _contains_cli_arg(patched, "--experimental_reuse_mask"):
        patched.append("--experimental_reuse_mask")

    return patched


def _patched_parse_args():
    original_argv = list(sys.argv)
    sys.argv = [original_argv[0]] + _inject_default_argv(original_argv[1:])
    try:
        return _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv


def _set_loraxs_trainable(model, train_classifier_head=True):
    del train_classifier_head

    trainable = []
    for name, param in model.named_parameters():
        is_loraxs = "lora_latent." in name
        is_classifier = name.startswith("classifier.") or name.startswith("score.")
        param.requires_grad = is_loraxs or is_classifier
        if param.requires_grad:
            trainable.append(name)
    return trainable


base.LoRALinear = LoRAXSPublicLinear
base._inject_lora_layers = inject_shared_u_qkv_loraxs_layers
base._set_lora_trainable = _set_loraxs_trainable
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
            launcher = base.MultiProcessLauncher(2, base.main)
            launcher.start()
            launcher.join()
            launcher.terminate()


if __name__ == "__main__":
    main()
