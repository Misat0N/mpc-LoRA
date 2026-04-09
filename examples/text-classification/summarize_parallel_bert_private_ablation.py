#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


GROUPS = [
    "full_private_lora",
    "public_backbone_lora",
    "public_backbone_loraxs",
    "public_backbone_loraxs_shared_left",
]


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


def main():
    parser = argparse.ArgumentParser(description="Summarize a parallel 4-GPU private BERT ablation run.")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    log_dir = Path(args.log_dir)

    rows = []
    for group_name in GROUPS:
        group_out_dir = output_root / group_name
        log_path = log_dir / f"{group_name}.log"
        summary = _load_json(group_out_dir / "train_eval_summary.json") or {}
        rows.append(
            {
                "group": group_name,
                "output_dir": str(group_out_dir),
                "log_path": str(log_path),
                "total_elapsed_s": summary.get("total_elapsed_s") or _extract_total_elapsed_s(log_path),
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
        "log_path",
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
