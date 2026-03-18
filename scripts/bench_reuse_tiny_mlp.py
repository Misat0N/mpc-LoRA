#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
import time

import crypten
import torch
from crypten.common.reuse_context import clear_current_reuse_step, set_current_reuse_step
from crypten.config import cfg
from crypten.mpc.primitives import beaver


PRESET_CASES = {
    "tiny": {
        "batch_size": 32,
        "in_features": 64,
        "hidden_features": 64,
        "out_features": 16,
        "hidden_layers": 1,
    },
    "base": {
        "batch_size": 64,
        "in_features": 256,
        "hidden_features": 256,
        "out_features": 64,
        "hidden_layers": 2,
    },
    "wide": {
        "batch_size": 128,
        "in_features": 512,
        "hidden_features": 1024,
        "out_features": 128,
        "hidden_layers": 2,
    },
    "tall": {
        "batch_size": 64,
        "in_features": 256,
        "hidden_features": 256,
        "out_features": 64,
        "hidden_layers": 4,
    },
}


class TinyMLP(crypten.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_layers=1,
        activation="relu",
        bias=True,
    ):
        super().__init__()
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be >= 0")

        self.hidden = crypten.nn.ModuleList()
        input_dim = in_features
        for _ in range(hidden_layers):
            self.hidden.append(crypten.nn.Linear(input_dim, hidden_features, bias=bias))
            input_dim = hidden_features
        self.output = crypten.nn.Linear(input_dim, out_features, bias=bias)
        self.activation = self._build_activation(activation)
        self.activation_name = activation

    @staticmethod
    def _build_activation(name):
        name = str(name).lower()
        if name == "relu":
            return crypten.nn.ReLU()
        if name == "tanh":
            return crypten.nn.Tanh()
        if name == "sigmoid":
            return crypten.nn.Sigmoid()
        raise ValueError(f"Unsupported activation `{name}`")

    def forward(self, x):
        for layer in self.hidden:
            x = self.activation(layer(x))
        x = self.output(x)
        return x


def _mean(values):
    if len(values) == 0:
        return 0.0
    return float(statistics.mean(values))


def _delta_dict(after, before):
    keys = set(before.keys()) | set(after.keys())
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def _set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)


def _safe_ratio(numerator, denominator, default=0.0):
    if denominator == 0:
        return default
    return float(numerator / denominator)


def _relative_change(current, baseline):
    if baseline == 0:
        return 0.0
    return float((current - baseline) / baseline)


def _synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configure_mode(experimental_reuse_mask, reuse_mode):
    cfg.mpc.experimental_reuse_mask = experimental_reuse_mask
    cfg.mpc.reuse_mode = reuse_mode
    cfg.mpc.reuse_scope = "STEP"
    cfg.mpc.reuse_op_types = ["matmul"]
    cfg.mpc.reuse_tagging = True


def _build_cases(args):
    case_names = []
    if args.preset_cases.strip():
        case_names = [item.strip() for item in args.preset_cases.split(",") if item.strip()]

    if len(case_names) == 0:
        return [
            {
                "name": args.case_name,
                "batch_size": args.batch_size,
                "in_features": args.in_features,
                "hidden_features": args.hidden_features,
                "out_features": args.out_features,
                "hidden_layers": args.hidden_layers,
                "activation": args.activation,
                "bias": not args.no_bias,
            }
        ]

    cases = []
    for case_name in case_names:
        if case_name not in PRESET_CASES:
            raise ValueError(
                f"Unknown preset case `{case_name}`. Available: {sorted(PRESET_CASES.keys())}"
            )
        case = dict(PRESET_CASES[case_name])
        case["name"] = case_name
        case["activation"] = args.activation
        case["bias"] = not args.no_bias
        cases.append(case)
    return cases


def _create_model(case, device):
    model = TinyMLP(
        in_features=case["in_features"],
        hidden_features=case["hidden_features"],
        out_features=case["out_features"],
        hidden_layers=case["hidden_layers"],
        activation=case["activation"],
        bias=case["bias"],
    )
    if device.type == "cuda":
        model = model.cuda(device=device)
    model.encrypt(src=0)
    model.train(True)
    return model


def _prepare_batches(case, args, device):
    batches = []
    total_steps = args.steps + args.warmup
    for _ in range(total_steps):
        x_plain = torch.randn(case["batch_size"], case["in_features"], device=device)
        y_plain = torch.randn(case["batch_size"], case["out_features"], device=device)
        batches.append((x_plain, y_plain))
    return batches


def _empty_metric_lists():
    return {
        "prep_time_s": [],
        "forward_time_s": [],
        "backward_time_s": [],
        "optim_time_s": [],
        "step_time_s": [],
        "comm_rounds": [],
        "comm_bytes": [],
        "comm_time_s": [],
        "beaver_reveal_calls": [],
        "beaver_revealed_tensors": [],
        "triple_generate_calls": [],
        "a_cache_hit": [],
        "a_cache_miss": [],
        "a_base_cache_hit": [],
        "a_base_cache_miss": [],
        "a_derived_cache_hit": [],
        "a_derived_generated": [],
        "b_cache_hit": [],
        "b_cache_miss": [],
        "b_base_cache_hit": [],
        "b_base_cache_miss": [],
        "b_derived_cache_hit": [],
        "b_derived_generated": [],
        "b_fresh_generated": [],
        "c_cache_hit": [],
        "c_cache_miss": [],
        "c_cache_probe_hit": [],
        "c_cache_probe_miss": [],
        "c_cache_bypassed": [],
        "c_fresh_generated": [],
        "residual_anchor_hit": [],
        "residual_anchor_miss": [],
    }


def _run_mode(args, case, mode_name, experimental_reuse_mask, reuse_mode, device, batches):
    _configure_mode(experimental_reuse_mask, reuse_mode)
    beaver.reset_reuse_stats(reset_cache=True)
    crypten.reset_communication_stats()

    model = _create_model(case, device)
    optimizer = crypten.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = crypten.nn.MSELoss()
    metrics = _empty_metric_lists()

    for step_idx, (x_plain, y_plain) in enumerate(batches):
        set_current_reuse_step(step_idx)
        beaver.begin_reuse_step(step_idx)
        comm_before = crypten.get_communication_stats()
        beaver_before = beaver.get_reuse_stats()

        try:
            _synchronize_device(device)
            step_start = time.perf_counter()
            prep_start = step_start
            x_enc = crypten.cryptensor(x_plain, src=0, requires_grad=True)
            y_enc = crypten.cryptensor(y_plain, src=0, requires_grad=False)
            optimizer.zero_grad()
            _synchronize_device(device)
            prep_end = time.perf_counter()

            forward_start = prep_end
            output = model(x_enc)
            loss = criterion(output, y_enc)
            _synchronize_device(device)
            forward_end = time.perf_counter()

            backward_start = forward_end
            loss.backward()
            _synchronize_device(device)
            backward_end = time.perf_counter()

            optim_start = backward_end
            optimizer.step()
            _synchronize_device(device)
            step_end = time.perf_counter()

            comm_after = crypten.get_communication_stats()
            beaver_after = beaver.get_reuse_stats()
        finally:
            beaver.end_reuse_step(step_idx)
            clear_current_reuse_step()

        if step_idx < args.warmup:
            continue

        comm_delta = _delta_dict(comm_after, comm_before)
        beaver_delta = _delta_dict(beaver_after, beaver_before)
        metrics["prep_time_s"].append(prep_end - prep_start)
        metrics["forward_time_s"].append(forward_end - forward_start)
        metrics["backward_time_s"].append(backward_end - backward_start)
        metrics["optim_time_s"].append(step_end - optim_start)
        metrics["step_time_s"].append(step_end - step_start)
        metrics["comm_rounds"].append(comm_delta.get("rounds", 0))
        metrics["comm_bytes"].append(comm_delta.get("bytes", 0))
        metrics["comm_time_s"].append(comm_delta.get("time", 0.0))
        metrics["beaver_reveal_calls"].append(beaver_delta.get("beaver_reveal_calls", 0))
        metrics["beaver_revealed_tensors"].append(
            beaver_delta.get("beaver_revealed_tensors", 0)
        )
        metrics["triple_generate_calls"].append(beaver_delta.get("triple_generate_calls", 0))
        metrics["a_cache_hit"].append(beaver_delta.get("a_cache_hit", 0))
        metrics["a_cache_miss"].append(beaver_delta.get("a_cache_miss", 0))
        metrics["a_base_cache_hit"].append(beaver_delta.get("a_base_cache_hit", 0))
        metrics["a_base_cache_miss"].append(beaver_delta.get("a_base_cache_miss", 0))
        metrics["a_derived_cache_hit"].append(beaver_delta.get("a_derived_cache_hit", 0))
        metrics["a_derived_generated"].append(beaver_delta.get("a_derived_generated", 0))
        metrics["b_cache_hit"].append(beaver_delta.get("b_cache_hit", 0))
        metrics["b_cache_miss"].append(beaver_delta.get("b_cache_miss", 0))
        metrics["b_base_cache_hit"].append(beaver_delta.get("b_base_cache_hit", 0))
        metrics["b_base_cache_miss"].append(beaver_delta.get("b_base_cache_miss", 0))
        metrics["b_derived_cache_hit"].append(beaver_delta.get("b_derived_cache_hit", 0))
        metrics["b_derived_generated"].append(beaver_delta.get("b_derived_generated", 0))
        metrics["b_fresh_generated"].append(beaver_delta.get("b_fresh_generated", 0))
        metrics["c_cache_hit"].append(beaver_delta.get("c_cache_hit", 0))
        metrics["c_cache_miss"].append(beaver_delta.get("c_cache_miss", 0))
        metrics["c_cache_probe_hit"].append(beaver_delta.get("c_cache_probe_hit", 0))
        metrics["c_cache_probe_miss"].append(beaver_delta.get("c_cache_probe_miss", 0))
        metrics["c_cache_bypassed"].append(beaver_delta.get("c_cache_bypassed", 0))
        metrics["c_fresh_generated"].append(beaver_delta.get("c_fresh_generated", 0))
        metrics["residual_anchor_hit"].append(beaver_delta.get("residual_anchor_hit", 0))
        metrics["residual_anchor_miss"].append(beaver_delta.get("residual_anchor_miss", 0))

    result = {
        "mode": mode_name,
        "reuse_enabled": experimental_reuse_mask,
        "reuse_mode": reuse_mode,
        "config": dict(case),
        "extra_counters": beaver.get_reuse_stats(),
    }
    for key, values in metrics.items():
        result[key] = _mean(values)
    return result


def _average_mode_runs(mode_runs):
    if len(mode_runs) == 0:
        return {}

    summary = {
        "mode": mode_runs[0]["mode"],
        "reuse_enabled": mode_runs[0]["reuse_enabled"],
        "reuse_mode": mode_runs[0]["reuse_mode"],
        "config": dict(mode_runs[0]["config"]),
        "num_repeats": len(mode_runs),
    }
    numeric_keys = [
        "prep_time_s",
        "forward_time_s",
        "backward_time_s",
        "optim_time_s",
        "step_time_s",
        "comm_rounds",
        "comm_bytes",
        "comm_time_s",
        "beaver_reveal_calls",
        "beaver_revealed_tensors",
        "triple_generate_calls",
        "a_cache_hit",
        "a_cache_miss",
        "a_base_cache_hit",
        "a_base_cache_miss",
        "a_derived_cache_hit",
        "a_derived_generated",
        "b_cache_hit",
        "b_cache_miss",
        "b_base_cache_hit",
        "b_base_cache_miss",
        "b_derived_cache_hit",
        "b_derived_generated",
        "b_fresh_generated",
        "c_cache_hit",
        "c_cache_miss",
        "c_cache_probe_hit",
        "c_cache_probe_miss",
        "c_cache_bypassed",
        "c_fresh_generated",
        "residual_anchor_hit",
        "residual_anchor_miss",
    ]
    for key in numeric_keys:
        summary[key] = _mean([run[key] for run in mode_runs])
    return summary


def _build_mode_comparisons(summary_by_mode):
    if "baseline" not in summary_by_mode:
        return {}
    baseline = summary_by_mode["baseline"]
    comparisons = {}
    for mode_name, item in summary_by_mode.items():
        step_speedup = _safe_ratio(baseline["step_time_s"], item["step_time_s"], default=1.0)
        comparisons[mode_name] = {
            "speedup_vs_baseline": step_speedup,
            "step_time_delta_pct": 100.0
            * _safe_ratio(
                baseline["step_time_s"] - item["step_time_s"],
                baseline["step_time_s"],
            ),
            "forward_time_reduction_pct": 100.0
            * _safe_ratio(
                baseline["forward_time_s"] - item["forward_time_s"],
                baseline["forward_time_s"],
            ),
            "backward_time_reduction_pct": 100.0
            * _safe_ratio(
                baseline["backward_time_s"] - item["backward_time_s"],
                baseline["backward_time_s"],
            ),
            "triple_reduction_pct": 100.0
            * _safe_ratio(
                baseline["triple_generate_calls"] - item["triple_generate_calls"],
                baseline["triple_generate_calls"],
            ),
            "reveal_tensor_reduction_pct": 100.0
            * _safe_ratio(
                baseline["beaver_revealed_tensors"] - item["beaver_revealed_tensors"],
                baseline["beaver_revealed_tensors"],
            ),
        }
    return comparisons


def _run_case(args, case, device, mode_specs, case_seed_base):
    mode_runs = {mode_name: [] for mode_name, _, _ in mode_specs}
    repeat_summaries = []
    for repeat_idx in range(args.repeats):
        repeat_seed = case_seed_base + repeat_idx if case_seed_base is not None else None
        _set_seed(repeat_seed)
        batches = _prepare_batches(case, args, device=device)
        repeat_record = {"repeat_idx": repeat_idx, "seed": repeat_seed, "results": {}}
        for mode_name, enabled, reuse_mode in mode_specs:
            result = _run_mode(
                args=args,
                case=case,
                mode_name=mode_name,
                experimental_reuse_mask=enabled,
                reuse_mode=reuse_mode,
                device=device,
                batches=batches,
            )
            mode_runs[mode_name].append(result)
            repeat_record["results"][mode_name] = result
        repeat_summaries.append(repeat_record)

    summary = {
        mode_name: _average_mode_runs(mode_runs[mode_name]) for mode_name in mode_runs.keys()
    }
    return {
        "case_name": case["name"],
        "config": dict(case),
        "summary": summary,
        "comparisons": _build_mode_comparisons(summary),
        "repeats": repeat_summaries,
    }


def _worker(args):
    _set_seed(args.seed)
    cfg.communicator.verbose = args.verbose_comm
    cfg.mpc.provider = args.provider
    device = torch.device(args.device)
    cases = _build_cases(args)

    mode_specs = [
        ("baseline", False, "FIX_A"),
        ("reuse_fix_a", True, "FIX_A"),
    ]
    if args.run_fix_ab:
        mode_specs.append(("reuse_fix_ab", True, "FIX_AB"))

    case_results = []
    for case_idx, case in enumerate(cases):
        case_seed_base = None if args.seed is None else args.seed + case_idx * 1000
        case_results.append(
            _run_case(
                args=args,
                case=case,
                device=device,
                mode_specs=mode_specs,
                case_seed_base=case_seed_base,
            )
        )

    return {
        "rank": crypten.communicator.get().get_rank(),
        "provider": cfg.mpc.provider,
        "cases": case_results,
    }


def _build_runner(world_size):
    @crypten.mpc.context.run_multiprocess(world_size)
    def _runner(args):
        return _worker(args)

    return _runner


def _print_case_summary(case_result):
    cfg_desc = case_result["config"]
    print("")
    print(
        "Case: "
        f"{case_result['case_name']} "
        f"[B={cfg_desc['batch_size']}, in={cfg_desc['in_features']}, hidden={cfg_desc['hidden_features']}, "
        f"out={cfg_desc['out_features']}, hidden_layers={cfg_desc['hidden_layers']}, act={cfg_desc['activation']}]"
    )
    print(
        "Columns: mode | prep(s) | fwd(s) | bwd(s) | opt(s) | step(s) | rounds | bytes | reveal_tensors | triple_gen | speedup"
    )
    mode_order = ["baseline", "reuse_fix_a", "reuse_fix_ab"]
    for mode in mode_order:
        if mode not in case_result["summary"]:
            continue
        item = case_result["summary"][mode]
        comparison = case_result["comparisons"].get(mode, {})
        print(
            f"{mode:>12} | "
            f"{item['prep_time_s']:.6f} | "
            f"{item['forward_time_s']:.6f} | "
            f"{item['backward_time_s']:.6f} | "
            f"{item['optim_time_s']:.6f} | "
            f"{item['step_time_s']:.6f} | "
            f"{item['comm_rounds']:.2f} | "
            f"{item['comm_bytes']:.2f} | "
            f"{item['beaver_revealed_tensors']:.2f} | "
            f"{item['triple_generate_calls']:.2f} | "
            f"{comparison.get('speedup_vs_baseline', 1.0):.3f}x"
        )

    print("Delta vs baseline:")
    for mode in ["reuse_fix_a", "reuse_fix_ab"]:
        if mode not in case_result["comparisons"]:
            continue
        comparison = case_result["comparisons"][mode]
        print(
            f"{mode:>12} | "
            f"step={comparison['step_time_delta_pct']:.2f}% | "
            f"fwd={comparison['forward_time_reduction_pct']:.2f}% | "
            f"bwd={comparison['backward_time_reduction_pct']:.2f}% | "
            f"triple={comparison['triple_reduction_pct']:.2f}% | "
            f"reveal_tensors={comparison['reveal_tensor_reduction_pct']:.2f}%"
        )
    print("Mask counters per step:")
    print(
        "        mode | a_base_hit/miss | a_der_hit/new | b_base_hit/miss | b_der_hit/new | b_fresh"
    )
    for mode in mode_order:
        if mode not in case_result["summary"]:
            continue
        item = case_result["summary"][mode]
        print(
            f"{mode:>12} | "
            f"{item['a_base_cache_hit']:.2f}/{item['a_base_cache_miss']:.2f} | "
            f"{item['a_derived_cache_hit']:.2f}/{item['a_derived_generated']:.2f} | "
            f"{item['b_base_cache_hit']:.2f}/{item['b_base_cache_miss']:.2f} | "
            f"{item['b_derived_cache_hit']:.2f}/{item['b_derived_generated']:.2f} | "
            f"{item['b_fresh_generated']:.2f}"
        )
    print("C/residual counters per step:")
    print("        mode | c_hit/miss | c_bypass/fresh | anchor_hit/miss")
    for mode in mode_order:
        if mode not in case_result["summary"]:
            continue
        item = case_result["summary"][mode]
        print(
            f"{mode:>12} | "
            f"{item['c_cache_probe_hit']:.2f}/{item['c_cache_probe_miss']:.2f} | "
            f"{item['c_cache_bypassed']:.2f}/{item['c_fresh_generated']:.2f} | "
            f"{item['residual_anchor_hit']:.2f}/{item['residual_anchor_miss']:.2f}"
        )


def _build_csv_rows(rank0_payload):
    rows = []
    for case_result in rank0_payload["cases"]:
        comparisons = case_result["comparisons"]
        for mode_name, item in case_result["summary"].items():
            row = {
                "case_name": case_result["case_name"],
                "provider": rank0_payload["provider"],
                "batch_size": item["config"]["batch_size"],
                "in_features": item["config"]["in_features"],
                "hidden_features": item["config"]["hidden_features"],
                "out_features": item["config"]["out_features"],
                "hidden_layers": item["config"]["hidden_layers"],
                "activation": item["config"]["activation"],
                "mode": mode_name,
                "prep_time_s": item["prep_time_s"],
                "forward_time_s": item["forward_time_s"],
                "backward_time_s": item["backward_time_s"],
                "optim_time_s": item["optim_time_s"],
                "step_time_s": item["step_time_s"],
                "comm_rounds": item["comm_rounds"],
                "comm_bytes": item["comm_bytes"],
                "comm_time_s": item["comm_time_s"],
                "beaver_reveal_calls": item["beaver_reveal_calls"],
                "beaver_revealed_tensors": item["beaver_revealed_tensors"],
                "triple_generate_calls": item["triple_generate_calls"],
                "a_cache_hit": item["a_cache_hit"],
                "a_cache_miss": item["a_cache_miss"],
                "a_base_cache_hit": item["a_base_cache_hit"],
                "a_base_cache_miss": item["a_base_cache_miss"],
                "a_derived_cache_hit": item["a_derived_cache_hit"],
                "a_derived_generated": item["a_derived_generated"],
                "b_cache_hit": item["b_cache_hit"],
                "b_cache_miss": item["b_cache_miss"],
                "b_base_cache_hit": item["b_base_cache_hit"],
                "b_base_cache_miss": item["b_base_cache_miss"],
                "b_derived_cache_hit": item["b_derived_cache_hit"],
                "b_derived_generated": item["b_derived_generated"],
                "b_fresh_generated": item["b_fresh_generated"],
                "c_cache_hit": item["c_cache_hit"],
                "c_cache_miss": item["c_cache_miss"],
                "c_cache_probe_hit": item["c_cache_probe_hit"],
                "c_cache_probe_miss": item["c_cache_probe_miss"],
                "c_cache_bypassed": item["c_cache_bypassed"],
                "c_fresh_generated": item["c_fresh_generated"],
                "residual_anchor_hit": item["residual_anchor_hit"],
                "residual_anchor_miss": item["residual_anchor_miss"],
                "speedup_vs_baseline": comparisons.get(mode_name, {}).get(
                    "speedup_vs_baseline", 1.0
                ),
                "step_time_delta_pct": comparisons.get(mode_name, {}).get(
                    "step_time_delta_pct", 0.0
                ),
                "triple_reduction_pct": comparisons.get(mode_name, {}).get(
                    "triple_reduction_pct", 0.0
                ),
                "reveal_tensor_reduction_pct": comparisons.get(mode_name, {}).get(
                    "reveal_tensor_reduction_pct", 0.0
                ),
            }
            rows.append(row)
    return rows


def _save_json(path, args, rank0_payload):
    payload = {
        "args": vars(args),
        "provider": rank0_payload["provider"],
        "cases": rank0_payload["cases"],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_csv(path, rank0_payload):
    rows = _build_csv_rows(rank0_payload)
    if len(rows) == 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rank0_payload, args):
    print("")
    print(f"Provider: {rank0_payload['provider']}")
    print(
        "Benchmark: TinyMLP train step = encrypt inputs + forward + MSE loss + backward + SGD"
    )
    print(
        f"Setup: repeats={args.repeats}, steps={args.steps}, warmup={args.warmup}, device={args.device}, world_size={args.world_size}"
    )
    for case_result in rank0_payload["cases"]:
        _print_case_summary(case_result)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight TinyMLP benchmark for Beaver mask reuse."
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--provider", type=str, default="TFP", choices=["TFP", "TTP"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--case-name", type=str, default="custom")
    parser.add_argument("--preset-cases", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=64)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--out-features", type=int, default=16)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "sigmoid"],
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-fix-ab", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--verbose-comm", action="store_true")
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--save-csv", type=str, default="")
    args = parser.parse_args()
    if args.hidden_layers < 0:
        raise ValueError("--hidden-layers must be >= 0")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    return args


def main():
    args = parse_args()
    runner = _build_runner(args.world_size)
    outputs = runner(args)
    if outputs is None:
        raise RuntimeError("Multiprocess benchmark failed. Check previous logs.")

    worker_outputs = [item for item in outputs if isinstance(item, dict) and "rank" in item]
    if len(worker_outputs) == 0:
        raise RuntimeError("No worker payload was returned from multiprocess benchmark.")

    payload_by_rank = {item["rank"]: item for item in worker_outputs}
    rank0_payload = payload_by_rank[min(payload_by_rank.keys())]
    _print_summary(rank0_payload, args)

    if args.save_json:
        _save_json(args.save_json, args, rank0_payload)
    if args.save_csv:
        _save_csv(args.save_csv, rank0_payload)


if __name__ == "__main__":
    main()
