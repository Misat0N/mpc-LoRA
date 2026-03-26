#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
import time

import crypten
import torch
from crypten.common.reuse_context import (
    clear_current_reuse_step,
    set_current_reuse_step,
    use_a_group,
)
from crypten.config import cfg
from crypten.mpc.primitives import beaver


PRESET_CASES = {
    "tiny": {
        "batch_size": 32,
        "in_features": 64,
        "hidden_features": 64,
        "out_features": 16,
        "fanout_heads": 3,
    },
    "base": {
        "batch_size": 64,
        "in_features": 256,
        "hidden_features": 256,
        "out_features": 64,
        "fanout_heads": 3,
    },
    "wide": {
        "batch_size": 128,
        "in_features": 512,
        "hidden_features": 1024,
        "out_features": 128,
        "fanout_heads": 4,
    },
}


class SharedLeftFanoutNet(crypten.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        fanout_heads=3,
        activation="relu",
        bias=True,
        shared_left_group=False,
    ):
        super().__init__()
        if fanout_heads < 2:
            raise ValueError("fanout_heads must be >= 2")
        self.shared_left_group = shared_left_group
        self.group_tag = f"fanout_heads:{id(self)}"
        self.stem = crypten.nn.Linear(in_features, hidden_features, bias=bias)
        self.heads = crypten.nn.ModuleList(
            [
                crypten.nn.Linear(hidden_features, out_features, bias=bias)
                for _ in range(fanout_heads)
            ]
        )
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
        x = self.activation(self.stem(x))
        if not self.shared_left_group:
            return [head(x) for head in self.heads]
        with use_a_group(self.group_tag):
            return [head(x) for head in self.heads]


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
    case_names = [item.strip() for item in args.preset_cases.split(",") if item.strip()]
    if len(case_names) == 0:
        return [
            {
                "name": args.case_name,
                "batch_size": args.batch_size,
                "in_features": args.in_features,
                "hidden_features": args.hidden_features,
                "out_features": args.out_features,
                "fanout_heads": args.fanout_heads,
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


def _select_device(args):
    if str(args.device).lower() != "cuda":
        return torch.device(args.device)

    rank = crypten.communicator.get().get_rank()
    if args.gpu_ids.strip():
        gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    if len(gpu_ids) == 0:
        raise RuntimeError("CUDA requested but no visible GPU is available.")
    local_gpu_id = gpu_ids[rank % len(gpu_ids)]
    torch.cuda.set_device(local_gpu_id)
    return torch.device(f"cuda:{local_gpu_id}")


def _create_model(case, device, shared_left_group):
    model = SharedLeftFanoutNet(
        in_features=case["in_features"],
        hidden_features=case["hidden_features"],
        out_features=case["out_features"],
        fanout_heads=case["fanout_heads"],
        activation=case["activation"],
        bias=case["bias"],
        shared_left_group=shared_left_group,
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
        targets = [
            torch.randn(case["batch_size"], case["out_features"], device=device)
            for _ in range(case["fanout_heads"])
        ]
        batches.append((x_plain, targets))
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
        "beaver_revealed_tensors": [],
        "triple_generate_calls": [],
        "a_base_cache_hit": [],
        "a_base_cache_miss": [],
        "residual_cache_hit": [],
        "residual_cache_miss": [],
        "residual_anchor_hit": [],
        "residual_anchor_miss": [],
    }


def _run_mode(
    args,
    case,
    mode_name,
    experimental_reuse_mask,
    reuse_mode,
    shared_left_group,
    device,
    batches,
):
    _configure_mode(experimental_reuse_mask, reuse_mode)
    beaver.reset_reuse_stats(reset_cache=True)
    crypten.reset_communication_stats()

    model = _create_model(case, device=device, shared_left_group=shared_left_group)
    optimizer = crypten.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = crypten.nn.MSELoss()
    metrics = _empty_metric_lists()

    for step_idx, (x_plain, target_plain_list) in enumerate(batches):
        set_current_reuse_step(step_idx)
        beaver.begin_reuse_step(step_idx)
        comm_before = crypten.get_communication_stats()
        beaver_before = beaver.get_reuse_stats()

        try:
            _synchronize_device(device)
            step_start = time.perf_counter()
            prep_start = step_start
            x_enc = crypten.cryptensor(x_plain, src=0, requires_grad=True)
            targets_enc = [
                crypten.cryptensor(target_plain, src=0, requires_grad=False)
                for target_plain in target_plain_list
            ]
            optimizer.zero_grad()
            _synchronize_device(device)
            prep_end = time.perf_counter()

            forward_start = prep_end
            outputs = model(x_enc)
            loss = criterion(outputs[0], targets_enc[0])
            for output, target in zip(outputs[1:], targets_enc[1:]):
                loss = loss + criterion(output, target)
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
        metrics["beaver_revealed_tensors"].append(
            beaver_delta.get("beaver_revealed_tensors", 0)
        )
        metrics["triple_generate_calls"].append(beaver_delta.get("triple_generate_calls", 0))
        metrics["a_base_cache_hit"].append(beaver_delta.get("a_base_cache_hit", 0))
        metrics["a_base_cache_miss"].append(beaver_delta.get("a_base_cache_miss", 0))
        metrics["residual_cache_hit"].append(beaver_delta.get("residual_cache_hit", 0))
        metrics["residual_cache_miss"].append(beaver_delta.get("residual_cache_miss", 0))
        metrics["residual_anchor_hit"].append(beaver_delta.get("residual_anchor_hit", 0))
        metrics["residual_anchor_miss"].append(beaver_delta.get("residual_anchor_miss", 0))

    result = {
        "mode": mode_name,
        "reuse_enabled": experimental_reuse_mask,
        "reuse_mode": reuse_mode,
        "shared_left_group": shared_left_group,
        "config": dict(case),
    }
    for key, values in metrics.items():
        result[key] = _mean(values)
    return result


def _average_mode_runs(mode_runs):
    summary = {
        "mode": mode_runs[0]["mode"],
        "reuse_enabled": mode_runs[0]["reuse_enabled"],
        "reuse_mode": mode_runs[0]["reuse_mode"],
        "shared_left_group": mode_runs[0]["shared_left_group"],
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
        "beaver_revealed_tensors",
        "triple_generate_calls",
        "a_base_cache_hit",
        "a_base_cache_miss",
        "residual_cache_hit",
        "residual_cache_miss",
        "residual_anchor_hit",
        "residual_anchor_miss",
    ]
    for key in numeric_keys:
        summary[key] = _mean([run[key] for run in mode_runs])
    return summary


def _build_comparisons(summary_by_mode):
    baseline = summary_by_mode["baseline"]
    comparisons = {}
    for mode_name, item in summary_by_mode.items():
        comparisons[mode_name] = {
            "speedup_vs_baseline": _safe_ratio(
                baseline["step_time_s"], item["step_time_s"], default=1.0
            ),
            "step_time_delta_pct": 100.0
            * _safe_ratio(
                baseline["step_time_s"] - item["step_time_s"],
                baseline["step_time_s"],
            ),
            "reveal_tensor_reduction_pct": 100.0
            * _safe_ratio(
                baseline["beaver_revealed_tensors"] - item["beaver_revealed_tensors"],
                baseline["beaver_revealed_tensors"],
            ),
            "triple_reduction_pct": 100.0
            * _safe_ratio(
                baseline["triple_generate_calls"] - item["triple_generate_calls"],
                baseline["triple_generate_calls"],
            ),
        }
    return comparisons


def _run_case(args, case, device):
    mode_specs = [
        ("baseline", False, "SHARED_LEFT", False),
        ("shared_left", True, "SHARED_LEFT", True),
    ]

    mode_runs = {mode_name: [] for mode_name, _, _, _ in mode_specs}
    for repeat_idx in range(args.repeats):
        repeat_seed = None if args.seed is None else args.seed + repeat_idx
        _set_seed(repeat_seed)
        batches = _prepare_batches(case, args, device=device)
        for mode_name, enabled, reuse_mode, shared_left_group in mode_specs:
            mode_runs[mode_name].append(
                _run_mode(
                    args=args,
                    case=case,
                    mode_name=mode_name,
                    experimental_reuse_mask=enabled,
                    reuse_mode=reuse_mode,
                    shared_left_group=shared_left_group,
                    device=device,
                    batches=batches,
                )
            )

    summary = {
        mode_name: _average_mode_runs(mode_runs[mode_name]) for mode_name in mode_runs.keys()
    }
    return {
        "case_name": case["name"],
        "config": dict(case),
        "summary": summary,
        "comparisons": _build_comparisons(summary),
    }


def _print_case_summary(case_result):
    cfg_desc = case_result["config"]
    print("")
    print(
        "Case: "
        f"{case_result['case_name']} "
        f"[B={cfg_desc['batch_size']}, in={cfg_desc['in_features']}, hidden={cfg_desc['hidden_features']}, "
        f"out={cfg_desc['out_features']}, heads={cfg_desc['fanout_heads']}, act={cfg_desc['activation']}]"
    )
    print(
        "Columns: mode | step(s) | rounds | bytes | reveal_tensors | triple_gen | a_base_hit/miss | residual_hit/miss | speedup"
    )
    for mode_name in ["baseline", "shared_left"]:
        if mode_name not in case_result["summary"]:
            continue
        item = case_result["summary"][mode_name]
        comparison = case_result["comparisons"].get(mode_name, {})
        print(
            f"{mode_name:>18} | "
            f"{item['step_time_s']:.6f} | "
            f"{item['comm_rounds']:.2f} | "
            f"{item['comm_bytes']:.2f} | "
            f"{item['beaver_revealed_tensors']:.2f} | "
            f"{item['triple_generate_calls']:.2f} | "
            f"{item['a_base_cache_hit']:.2f}/{item['a_base_cache_miss']:.2f} | "
            f"{item['residual_cache_hit']:.2f}/{item['residual_cache_miss']:.2f} | "
            f"{comparison.get('speedup_vs_baseline', 1.0):.3f}x"
        )

    print("Delta vs baseline:")
    for mode_name in ["shared_left"]:
        if mode_name not in case_result["comparisons"]:
            continue
        comparison = case_result["comparisons"][mode_name]
        print(
            f"{mode_name:>18} | "
            f"step={comparison['step_time_delta_pct']:.2f}% | "
            f"triple={comparison['triple_reduction_pct']:.2f}% | "
            f"reveal_tensors={comparison['reveal_tensor_reduction_pct']:.2f}%"
        )


def _build_csv_rows(rank0_payload):
    rows = []
    for case_result in rank0_payload["cases"]:
        for mode_name, item in case_result["summary"].items():
            row = {
                "case_name": case_result["case_name"],
                "provider": rank0_payload["provider"],
                "mode": mode_name,
                "batch_size": item["config"]["batch_size"],
                "in_features": item["config"]["in_features"],
                "hidden_features": item["config"]["hidden_features"],
                "out_features": item["config"]["out_features"],
                "fanout_heads": item["config"]["fanout_heads"],
                "step_time_s": item["step_time_s"],
                "comm_rounds": item["comm_rounds"],
                "comm_bytes": item["comm_bytes"],
                "beaver_revealed_tensors": item["beaver_revealed_tensors"],
                "triple_generate_calls": item["triple_generate_calls"],
                "a_base_cache_hit": item["a_base_cache_hit"],
                "a_base_cache_miss": item["a_base_cache_miss"],
                "residual_cache_hit": item["residual_cache_hit"],
                "residual_cache_miss": item["residual_cache_miss"],
                "residual_anchor_hit": item["residual_anchor_hit"],
                "residual_anchor_miss": item["residual_anchor_miss"],
                "speedup_vs_baseline": case_result["comparisons"].get(mode_name, {}).get(
                    "speedup_vs_baseline", 1.0
                ),
            }
            rows.append(row)
    return rows


def _save_json(path, args, rank0_payload):
    payload = {"args": vars(args), "provider": rank0_payload["provider"], "cases": rank0_payload["cases"]}
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


def _worker(args):
    _set_seed(args.seed)
    cfg.communicator.verbose = args.verbose_comm
    cfg.mpc.provider = args.provider
    device = _select_device(args)
    cases = _build_cases(args)
    return {
        "rank": crypten.communicator.get().get_rank(),
        "provider": cfg.mpc.provider,
        "device": str(device),
        "cases": [_run_case(args, case, device=device) for case in cases],
    }


def _build_runner(world_size):
    @crypten.mpc.context.run_multiprocess(world_size)
    def _runner(args):
        return _worker(args)

    return _runner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark shared-left Beaver reuse for A@B, A@C, ... fan-out matmuls."
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--provider", type=str, default="TFP", choices=["TFP", "TTP"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--case-name", type=str, default="fanout_custom")
    parser.add_argument("--preset-cases", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--hidden-features", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=64)
    parser.add_argument("--fanout-heads", type=int, default=3)
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

    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--verbose-comm", action="store_true")
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--save-csv", type=str, default="")
    args = parser.parse_args()
    if args.fanout_heads < 2:
        raise ValueError("--fanout-heads must be >= 2")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    return args


def main():
    args = parse_args()
    outputs = _build_runner(args.world_size)(args)
    if outputs is None:
        raise RuntimeError("Multiprocess benchmark failed. Check previous logs.")

    worker_outputs = [item for item in outputs if isinstance(item, dict) and "rank" in item]
    if len(worker_outputs) == 0:
        raise RuntimeError("No worker payload was returned from multiprocess benchmark.")

    payload_by_rank = {item["rank"]: item for item in worker_outputs}
    rank0_payload = payload_by_rank[min(payload_by_rank.keys())]
    print("")
    print(f"Provider: {rank0_payload['provider']}")
    print(
        "Benchmark: Shared-left fan-out train step = shared stem + multi-head Linear fan-out + MSE + backward + SGD"
    )
    print(
        f"Setup: repeats={args.repeats}, steps={args.steps}, warmup={args.warmup}, device={rank0_payload['device']}, world_size={args.world_size}"
    )
    for case_result in rank0_payload["cases"]:
        _print_case_summary(case_result)

    if args.save_json:
        _save_json(args.save_json, args, rank0_payload)
    if args.save_csv:
        _save_csv(args.save_csv, rank0_payload)


if __name__ == "__main__":
    main()

