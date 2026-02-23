#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def extract_accuracy(metric):
    if isinstance(metric, dict) and "accuracy" in metric:
        try:
            return float(metric["accuracy"])
        except (TypeError, ValueError):
            return None
    return None


def load_step_summary(base_dir: Path, step: int):
    summary_path = base_dir / f"steps_{step}" / "train_eval_summary.json"
    if not summary_path.exists():
        return {
            "step": step,
            "status": "missing",
            "summary_path": str(summary_path),
            "train_steps": None,
            "private_acc": None,
            "plain_acc": None,
        }

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {
            "step": step,
            "status": "invalid_json",
            "summary_path": str(summary_path),
            "train_steps": None,
            "private_acc": None,
            "plain_acc": None,
        }

    private_acc = extract_accuracy(data.get("private_eval_metric"))
    plain_acc = extract_accuracy(data.get("plain_eval_metric"))

    return {
        "step": step,
        "status": "ok",
        "summary_path": str(summary_path),
        "train_steps": data.get("train_steps"),
        "private_acc": private_acc,
        "plain_acc": plain_acc,
    }


def evaluate_trend(rows, tol=1e-6, plateau_eps=2e-3):
    valid = [r for r in rows if r["status"] == "ok" and r["private_acc"] is not None]
    if len(valid) < 2:
        return {
            "trend": "insufficient_data",
            "convergence": "unknown",
            "reason": "Need at least 2 valid private accuracy points.",
        }

    accs = [r["private_acc"] for r in valid]
    deltas = [accs[i] - accs[i - 1] for i in range(1, len(accs))]
    monotonic_non_decreasing = all(d >= -tol for d in deltas)
    last_delta = deltas[-1]

    if monotonic_non_decreasing and abs(last_delta) <= plateau_eps:
        trend = "non_decreasing_plateau"
        convergence = "likely_converging"
        reason = "Accuracy non-decreasing and latest improvement is very small."
    elif monotonic_non_decreasing:
        trend = "non_decreasing"
        convergence = "still_improving"
        reason = "Accuracy keeps increasing; no clear plateau yet."
    else:
        trend = "non_monotonic"
        convergence = "unstable_or_noisy"
        reason = "Accuracy is not monotonic; run longer or average across seeds."

    return {
        "trend": trend,
        "convergence": convergence,
        "reason": reason,
        "deltas": deltas,
    }


def print_table(rows):
    print("\nConvergence Sweep Results")
    print("step\ttrain_steps\tprivate_acc\tplain_acc\tstatus")
    for r in rows:
        p = "-" if r["private_acc"] is None else f"{r['private_acc']:.6f}"
        q = "-" if r["plain_acc"] is None else f"{r['plain_acc']:.6f}"
        print(f"{r['step']}\t{r['train_steps']}\t{p}\t{q}\t{r['status']}")


def main():
    parser = argparse.ArgumentParser(description="Summarize convergence from train_eval_summary.json files.")
    parser.add_argument("--base_dir", type=Path, required=True, help="Base directory containing steps_* outputs.")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="Step list, e.g. 10 30 50")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON file path.")
    args = parser.parse_args()

    rows = [load_step_summary(args.base_dir, s) for s in args.steps]
    trend = evaluate_trend(rows)

    print_table(rows)
    print("\nConvergence Judgment")
    print(f"trend: {trend['trend']}")
    print(f"convergence: {trend['convergence']}")
    print(f"reason: {trend['reason']}")

    report = {
        "base_dir": str(args.base_dir),
        "rows": rows,
        "judgment": trend,
    }

    output_path = args.output
    if output_path is None:
        output_path = args.base_dir / "convergence_summary.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"saved summary: {output_path}")


if __name__ == "__main__":
    main()
