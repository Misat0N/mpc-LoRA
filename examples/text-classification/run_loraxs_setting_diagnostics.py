#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def _load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_group(group_name, command, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "console.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=str(THIS_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"group {group_name} failed, see {log_path}")
    return log_path


def _summary_row(name, script_name, setting, summary_path, log_path):
    summary = _load_json(summary_path) or {}
    return {
        "group": name,
        "script": script_name,
        "setting": setting,
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "train_steps": summary.get("train_steps"),
        "private_eval_metric": summary.get("private_eval_metric"),
        "plain_eval_metric": summary.get("plain_eval_metric"),
        "learning_rate": summary.get("learning_rate"),
        "classifier_learning_rate": summary.get("classifier_learning_rate"),
        "optimizer_summary": summary.get("optimizer_summary"),
        "public_non_lora_weights": summary.get("public_non_lora_weights"),
        "encrypted_param_keywords": summary.get("encrypted_param_keywords"),
        "reuse_mode": summary.get("reuse_mode"),
        "final_comm_stats": summary.get("final_comm_stats"),
        "reuse_profile_summary": summary.get("reuse_profile_summary"),
        "basis_source": "weight_svd" if "loraxs" in name else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a small-step diagnostics matrix for LoRA / LoRA-XS settings. "
            "This isolates target modules and grouped learning-rate effects while "
            "keeping LoRA-XS on the current weight-SVD basis."
        )
    )
    parser.add_argument("--task_name", type=str, default="sst2")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--len_data", type=int, default=64)
    parser.add_argument("--max_train_steps", type=int, default=120)
    parser.add_argument("--log_every_steps", type=int, default=5)
    parser.add_argument("--eval_max_steps", type=int, default=128)
    parser.add_argument("--train_max_samples", type=int, default=-1)
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--classifier_learning_rate", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--output_root", type=str, default=None)
    args = parser.parse_args()

    output_root = (
        Path(args.output_root)
        if args.output_root
        else THIS_DIR / "eval_private" / args.task_name / "loraxs_setting_diagnostics"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    common = [
        "--model_name_or_path",
        args.model_name_or_path,
        "--task_name",
        args.task_name,
        "--gpu_ids",
        args.gpu_ids,
        "--seed",
        str(args.seed),
        "--pad_to_max_length",
        "--len_data",
        str(args.len_data),
        "--max_length",
        str(args.max_length),
        "--train_max_samples",
        str(args.train_max_samples),
        "--eval_max_samples",
        str(args.eval_max_samples),
        "--max_train_steps",
        str(args.max_train_steps),
        "--log_every_steps",
        str(args.log_every_steps),
        "--eval_max_steps",
        str(args.eval_max_steps),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--learning_rate",
        str(args.learning_rate),
        "--momentum",
        str(args.momentum),
        "--reuse_profile",
    ]

    groups = [
        {
            "name": "lora_current_qkv_singlelr",
            "script": "run_glue_private_loraxs_lora_train.py",
            "setting": {
                "target_modules": "query,key,value",
                "classifier_learning_rate": None,
                "basis_source": None,
            },
            "extra_args": [],
        },
        {
            "name": "loraxs_current_qkv_singlelr",
            "script": "run_glue_private_loraxs_train.py",
            "setting": {
                "target_modules": "query,key,value",
                "classifier_learning_rate": None,
                "basis_source": "weight_svd",
            },
            "extra_args": [],
        },
        {
            "name": "loraxs_repo_targets_singlelr",
            "script": "run_glue_private_loraxs_recipe_train.py",
            "setting": {
                "target_modules": "query,value,attention.output.dense,output.dense",
                "classifier_learning_rate": None,
                "basis_source": "weight_svd",
            },
            "extra_args": ["--classifier_learning_rate", "-1"],
        },
        {
            "name": "loraxs_current_qkv_groupedlr",
            "script": "run_glue_private_loraxs_train.py",
            "setting": {
                "target_modules": "query,key,value",
                "classifier_learning_rate": args.classifier_learning_rate,
                "basis_source": "weight_svd",
            },
            "extra_args": ["--classifier_learning_rate", str(args.classifier_learning_rate)],
        },
        {
            "name": "loraxs_repo_targets_groupedlr",
            "script": "run_glue_private_loraxs_recipe_train.py",
            "setting": {
                "target_modules": "query,value,attention.output.dense,output.dense",
                "classifier_learning_rate": args.classifier_learning_rate,
                "basis_source": "weight_svd",
            },
            "extra_args": ["--classifier_learning_rate", str(args.classifier_learning_rate)],
        },
        {
            "name": "lora_repo_targets_groupedlr",
            "script": "run_glue_private_loraxs_recipe_lora_train.py",
            "setting": {
                "target_modules": "query,value,attention.output.dense,output.dense",
                "classifier_learning_rate": args.classifier_learning_rate,
                "basis_source": None,
            },
            "extra_args": ["--classifier_learning_rate", str(args.classifier_learning_rate)],
        },
    ]

    rows = []
    for group in groups:
        group_out_dir = output_root / group["name"]
        command = [
            sys.executable,
            group["script"],
            *common,
            *group["extra_args"],
            "--output_dir",
            str(group_out_dir),
        ]
        log_path = _run_group(group["name"], command, group_out_dir)
        rows.append(
            _summary_row(
                name=group["name"],
                script_name=group["script"],
                setting=group["setting"],
                summary_path=group_out_dir / "train_eval_summary.json",
                log_path=log_path,
            )
        )

    summary_json_path = output_root / "setting_diagnostics_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"saved: {summary_json_path}")


if __name__ == "__main__":
    main()
