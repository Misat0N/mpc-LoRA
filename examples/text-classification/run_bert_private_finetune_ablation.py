#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def _extract_total_elapsed_s(log_path: Path):
    if not log_path.exists():
        return None
    pattern = re.compile(r"total elapsed=([0-9.]+)s")
    elapsed = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match is not None:
            elapsed = float(match.group(1))
    return elapsed


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
    parser = argparse.ArgumentParser(description="Run private BERT finetuning ablation groups.")
    parser.add_argument("--task_name", type=str, default="sst2")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--len_data", type=int, default=64)
    parser.add_argument("--max_train_steps", type=int, default=300)
    parser.add_argument("--log_every_steps", type=int, default=5)
    parser.add_argument("--eval_max_steps", type=int, default=256)
    parser.add_argument("--train_max_samples", type=int, default=-1)
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--output_root", type=str, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else THIS_DIR / "eval_private" / args.task_name / "ablation_matrix"
    output_root.mkdir(parents=True, exist_ok=True)

    common_dense = [
        "--model_name_or_path", args.model_name_or_path,
        "--task_name", args.task_name,
        "--gpu_ids", args.gpu_ids,
        "--seed", str(args.seed),
        "--pad_to_max_length",
        "--len_data", str(args.len_data),
        "--max_length", str(args.max_length),
        "--train_max_samples", str(args.train_max_samples),
        "--eval_max_samples", str(args.eval_max_samples),
        "--max_train_steps", str(args.max_train_steps),
        "--log_every_steps", str(args.log_every_steps),
        "--eval_max_steps", str(args.eval_max_steps),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
    ]
    common_low_rank = common_dense + [
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", "0.0",
        "--lora_target_modules", "query,key,value",
        "--learning_rate", str(args.learning_rate),
        "--momentum", str(args.momentum),
    ]

    groups = [
        (
            "full_private_lora",
            [
                sys.executable,
                "run_glue_private_full_private_lora_train.py",
                *common_low_rank,
                "--reuse_profile",
            ],
        ),
        (
            "public_backbone_lora",
            [
                sys.executable,
                "run_glue_private_loraxs_lora_train.py",
                *common_low_rank,
                "--public_non_lora_weights",
                "--encrypted_param_keywords",
                "lora_A.,lora_B.,classifier.,score.",
                "--reuse_profile",
            ],
        ),
        (
            "public_backbone_loraxs",
            [
                sys.executable,
                "run_glue_private_loraxs_train.py",
                *common_low_rank,
                "--public_non_lora_weights",
                "--encrypted_param_keywords",
                "lora_latent.,classifier.,score.",
                "--reuse_profile",
            ],
        ),
        (
            "public_backbone_loraxs_shared_left",
            [
                sys.executable,
                "run_glue_private_loraxs_shared_left_train.py",
                *common_low_rank,
                "--public_non_lora_weights",
                "--encrypted_param_keywords",
                "lora_latent.,classifier.,score.",
                "--reuse_profile",
            ],
        ),
    ]

    rows = []
    for group_name, command in groups:
        group_out_dir = output_root / group_name
        command_with_output = [*command, "--output_dir", str(group_out_dir)]
        log_path = _run_group(group_name, command_with_output, group_out_dir)
        summary = _load_json(group_out_dir / "train_eval_summary.json") or {}
        rows.append(
            {
                "group": group_name,
                "output_dir": str(group_out_dir),
                "log_path": str(log_path),
                "total_elapsed_s": _extract_total_elapsed_s(log_path),
                "train_steps": summary.get("train_steps"),
                "private_eval_metric": summary.get("private_eval_metric"),
                "plain_eval_metric": summary.get("plain_eval_metric"),
                "public_non_lora_weights": summary.get("public_non_lora_weights"),
                "encrypted_param_keywords": summary.get("encrypted_param_keywords"),
                "reuse_mode": summary.get("reuse_mode"),
                "shared_left_groups": (summary.get("shared_left_group_summary") or {}).get("num_groups"),
                "reuse_profile_summary": summary.get("reuse_profile_summary"),
                "final_comm_stats": summary.get("final_comm_stats"),
            }
        )

    summary_json_path = output_root / "ablation_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    summary_tsv_path = output_root / "ablation_summary.tsv"
    header = [
        "group",
        "total_elapsed_s",
        "train_steps",
        "private_eval_metric",
        "plain_eval_metric",
        "public_non_lora_weights",
        "encrypted_param_keywords",
        "reuse_mode",
        "shared_left_groups",
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
