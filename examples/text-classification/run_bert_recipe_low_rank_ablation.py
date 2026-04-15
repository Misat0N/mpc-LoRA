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


def main():
    parser = argparse.ArgumentParser(description="Run the recipe-aligned 4-group LoRA / LoRA-XS ablation.")
    parser.add_argument("--task_name", type=str, default="sst2")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--len_data", type=int, default=128)
    parser.add_argument("--max_train_steps", type=int, default=600)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--eval_max_steps", type=int, default=256)
    parser.add_argument("--train_max_samples", type=int, default=-1)
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--classifier_learning_rate", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["none", "linear"])
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--output_root", type=str, default=None)
    args = parser.parse_args()

    output_root = (
        Path(args.output_root)
        if args.output_root
        else THIS_DIR / "eval_compare" / args.task_name / "recipe_low_rank_ablation"
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
        "--classifier_learning_rate",
        str(args.classifier_learning_rate),
        "--momentum",
        str(args.momentum),
        "--weight_decay",
        str(args.weight_decay),
        "--lr_scheduler_type",
        args.lr_scheduler_type,
        "--num_warmup_steps",
        str(args.num_warmup_steps),
        "--warmup_ratio",
        str(args.warmup_ratio),
    ]

    groups = [
        ("plain_lora_recipe", [sys.executable, "run_glue_plain_lora_recipe_train.py", *common]),
        ("plain_loraxs_recipe", [sys.executable, "run_glue_plain_loraxs_recipe_train.py", *common]),
        (
            "mpc_lora_recipe",
            [sys.executable, "run_glue_private_loraxs_recipe_lora_train.py", *common, "--reuse_profile"],
        ),
        (
            "mpc_loraxs_recipe",
            [sys.executable, "run_glue_private_loraxs_recipe_train.py", *common, "--reuse_profile"],
        ),
    ]

    rows = []
    for group_name, command in groups:
        group_out_dir = output_root / group_name
        log_path = _run_group(group_name, [*command, "--output_dir", str(group_out_dir)], group_out_dir)
        summary = _load_json(group_out_dir / "train_eval_summary.json") or {}
        rows.append(
            {
                "group": group_name,
                "output_dir": str(group_out_dir),
                "log_path": str(log_path),
                "train_steps": summary.get("train_steps"),
                "adapter_type": summary.get("adapter_type"),
                "eval_metric": summary.get("eval_metric"),
                "private_eval_metric": summary.get("private_eval_metric"),
                "plain_eval_metric": summary.get("plain_eval_metric"),
                "optimizer_summary": summary.get("optimizer_summary"),
                "lr_scheduler_summary": summary.get("lr_scheduler_summary"),
                "trainable_param_summary": summary.get("trainable_param_summary"),
                "reuse_profile_summary": summary.get("reuse_profile_summary"),
                "final_comm_stats": summary.get("final_comm_stats"),
                "total_elapsed_s": summary.get("total_elapsed_s"),
            }
        )

    summary_json_path = output_root / "recipe_ablation_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    summary_tsv_path = output_root / "recipe_ablation_summary.tsv"
    header = [
        "group",
        "train_steps",
        "eval_metric",
        "private_eval_metric",
        "plain_eval_metric",
        "total_elapsed_s",
        "optimizer_summary",
        "lr_scheduler_summary",
        "trainable_param_summary",
        "final_comm_stats",
        "reuse_profile_summary",
        "output_dir",
    ]
    with summary_tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            serialized = []
            for col in header:
                value = row.get(col)
                if isinstance(value, str):
                    serialized.append(value)
                else:
                    serialized.append(json.dumps(value, ensure_ascii=False))
            f.write("\t".join(serialized) + "\n")

    print(f"saved: {summary_json_path}")
    print(f"saved: {summary_tsv_path}")


if __name__ == "__main__":
    main()
