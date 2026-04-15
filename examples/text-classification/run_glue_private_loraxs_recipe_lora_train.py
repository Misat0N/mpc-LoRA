"""
Standard LoRA private finetuning entrypoint aligned with the original LoRA-XS
GLUE recipe so it can serve as a fairer baseline against LoRA-XS.

This changes the experiment setup to:
- target_modules = query,value,attention.output.dense,output.dense
- classifier head uses a separate learning rate by default
- backbone weights remain public / frozen
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


def _inject_default_argv(argv):
    patched = list(argv)

    if not _contains_cli_arg(patched, "--model_name_or_path"):
        patched.extend(["--model_name_or_path", "bert-base-uncased"])
    if not _contains_cli_arg(patched, "--task_name"):
        patched.extend(["--task_name", "sst2"])
    if not _contains_cli_arg(patched, "--adapter_type_label"):
        patched.extend(["--adapter_type_label", "lora"])
    if not _contains_cli_arg(patched, "--max_length"):
        patched.extend(["--max_length", "128"])
    if not _contains_cli_arg(patched, "--lora_target_modules"):
        patched.extend(
            ["--lora_target_modules", "query,value,attention.output.dense,output.dense"]
        )
    if not _contains_cli_arg(patched, "--lora_dropout"):
        patched.extend(["--lora_dropout", "0.0"])
    if not _contains_cli_arg(patched, "--crypten_native_lora"):
        patched.append("--crypten_native_lora")
    if not _contains_cli_arg(patched, "--public_non_lora_weights"):
        patched.append("--public_non_lora_weights")
    if not _contains_cli_arg(patched, "--encrypted_param_keywords"):
        patched.extend(["--encrypted_param_keywords", "lora_A.,lora_B.,classifier.,score."])
    if not _contains_cli_arg(patched, "--classifier_learning_rate"):
        patched.extend(["--classifier_learning_rate", "2e-4"])
    if not _contains_cli_arg(patched, "--lr_scheduler_type"):
        patched.extend(["--lr_scheduler_type", "linear"])
    if not _contains_cli_arg(patched, "--warmup_ratio"):
        patched.extend(["--warmup_ratio", "0.06"])

    return patched


def _patched_parse_args():
    original_argv = list(sys.argv)
    sys.argv = [original_argv[0]] + _inject_default_argv(original_argv[1:])
    try:
        return _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv


def _set_lora_and_classifier_trainable(model, train_classifier_head=True):
    del train_classifier_head

    trainable = []
    for name, param in model.named_parameters():
        is_lora = ("lora_A." in name) or ("lora_B." in name)
        is_classifier = name.startswith("classifier.") or name.startswith("score.")
        param.requires_grad = is_lora or is_classifier
        if param.requires_grad:
            trainable.append(name)
    return trainable


base.parse_args = _patched_parse_args
base._set_lora_trainable = _set_lora_and_classifier_trainable


def main():
    args = base.parse_args()
    if args.comp:
        with base.cfg.temp_override({"cost.estimate_cost": True, "cost.estimate_mode": "comp"}):
            base.main()
    elif args.acc:
        with base.cfg.temp_override({"cost.estimate_cost": False}):
            base.main()
    else:
        with base.cfg.temp_override(
            {"cost.estimate_cost": args.print_comm_cost, "cost.estimate_mode": "comm"}
        ):
            launcher = base.MultiProcessLauncher(2, base.main)
            launcher.start()
            launcher.join()
            launcher.terminate()


if __name__ == "__main__":
    main()
