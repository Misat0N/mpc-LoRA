#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_LIB = REPO_ROOT / "build" / "lib"
if str(BUILD_LIB) not in sys.path:
    sys.path.insert(0, str(BUILD_LIB))

import crypten
import torch
from crypten import communicator as comm
from crypten.gradients import (
    AutogradFunction,
    FUNCTION_REGISTRY,
    _grouped_shared_left_matmuls,
    get_structural_reuse_stats,
    structural_reuse_capability,
)

from multiprocess_launcher import MultiProcessLauncher


class GroupedNoTokenLoRALinear(AutogradFunction):
    """Old LoRA hot path: grouped backward, no forward/backward token."""

    @staticmethod
    def forward(ctx, input, weight, lora_A, lora_B, scaling, bias=None):
        non_differentiable = [weight]
        if bias is not None:
            non_differentiable.append(bias)
        ctx.mark_non_differentiable(non_differentiable)

        output = input.matmul(weight.t())
        lora_hidden = input.matmul(lora_A.t())
        ctx.save_multiple_for_backward(
            [input, weight, lora_A, lora_B, bias, scaling, lora_hidden]
        )

        output = output.add(lora_hidden.matmul(lora_B.t()).mul(scaling))
        if bias is not None:
            output = output.add(bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, lora_A, lora_B, bias, scaling, lora_hidden = ctx.saved_tensors

        input_requires_grad = getattr(input, "requires_grad", False)
        weight_requires_grad = getattr(weight, "requires_grad", False)
        lora_A_requires_grad = getattr(lora_A, "requires_grad", False)
        lora_B_requires_grad = getattr(lora_B, "requires_grad", False)
        bias_requires_grad = bias is not None and getattr(bias, "requires_grad", False)

        input_size = input.size()
        in_features = input_size[-1]
        out_features = weight.size(0)

        input_2d = input.reshape(-1, in_features)
        grad_output_2d = grad_output.reshape(-1, out_features)
        lora_hidden_2d = lora_hidden.reshape(-1, lora_hidden.size(-1))

        grouped_specs = []
        grouped_keys = []
        if input_requires_grad:
            grouped_specs.append((weight, False))
            grouped_keys.append("grad_input_base_2d")
        if input_requires_grad or lora_A_requires_grad:
            grouped_specs.append((lora_B, False))
            grouped_keys.append("grad_lora_hidden_base_2d")
        if weight_requires_grad:
            grouped_specs.append((input_2d, True))
            grouped_keys.append("grad_weight")
        if lora_B_requires_grad:
            grouped_specs.append((lora_hidden_2d, True))
            grouped_keys.append("grad_lora_B")

        grouped_results = {}
        if grouped_specs:
            grouped_values = _grouped_shared_left_matmuls(grad_output_2d, grouped_specs)
            grouped_results = dict(zip(grouped_keys, grouped_values))

        grad_input = None
        grad_lora_A = None
        grad_lora_B = None
        grad_weight = None
        grad_bias = None

        grad_lora_hidden_base_2d = grouped_results.get("grad_lora_hidden_base_2d")
        grad_lora_hidden_2d = (
            grad_lora_hidden_base_2d.mul(scaling)
            if grad_lora_hidden_base_2d is not None
            else None
        )

        if input_requires_grad:
            grad_input_base_2d = grouped_results["grad_input_base_2d"]
            grad_input = grad_input_base_2d.reshape(input_size)
            if grad_lora_hidden_2d is not None:
                grad_input = grad_input.add(
                    grad_lora_hidden_2d.matmul(lora_A).reshape(input_size)
                )

        if lora_A_requires_grad:
            if grad_lora_hidden_2d is None:
                raise RuntimeError("grad_lora_hidden_2d is required for grad_lora_A")
            grad_lora_A = grad_lora_hidden_2d.t().matmul(input_2d)

        if lora_B_requires_grad:
            grad_lora_B = grouped_results["grad_lora_B"].mul(scaling)

        if weight_requires_grad:
            grad_weight = grouped_results["grad_weight"]

        if bias_requires_grad:
            grad_bias = grad_output_2d.sum(0)

        grad_outputs = [grad_input]
        if weight_requires_grad:
            grad_outputs.append(grad_weight)
        if lora_A_requires_grad:
            grad_outputs.append(grad_lora_A)
        if lora_B_requires_grad:
            grad_outputs.append(grad_lora_B)
        if bias_requires_grad:
            grad_outputs.append(grad_bias)

        return tuple(grad_outputs)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def _clear_grad(tensor):
    if hasattr(tensor, "grad"):
        tensor.grad = None


def _make_cryptensor(shape, device, requires_grad, seed, src=0):
    rank = comm.get().get_rank()
    torch.manual_seed(seed)
    plain = torch.randn(shape, device=device) if rank == src else torch.empty(shape, device=device)
    return crypten.cryptensor(plain, src=src, requires_grad=requires_grad)


def _run_one_mode(args, mode, device):
    original_lora_fn = FUNCTION_REGISTRY["lora_linear"]
    if mode == "grouped_no_token":
        FUNCTION_REGISTRY["lora_linear"] = GroupedNoTokenLoRALinear
    elif mode != "token":
        raise ValueError(f"unknown mode: {mode}")

    try:
        leaves = []
        modules = []
        rows = args.batch_tokens
        scaling = float(args.lora_alpha) / float(args.lora_rank)

        input_tensor = _make_cryptensor(
            (rows, args.in_features),
            device,
            args.input_requires_grad,
            args.seed + 1,
        )
        leaves.append(input_tensor)

        for idx in range(args.modules):
            weight = _make_cryptensor(
                (args.out_features, args.in_features),
                device,
                False,
                args.seed + 100 + idx,
            )
            lora_a = _make_cryptensor(
                (args.lora_rank, args.in_features),
                device,
                True,
                args.seed + 200 + idx,
            )
            lora_b = _make_cryptensor(
                (args.out_features, args.lora_rank),
                device,
                True,
                args.seed + 300 + idx,
            )
            bias = None
            if args.bias:
                bias = _make_cryptensor(
                    (args.out_features,),
                    device,
                    False,
                    args.seed + 400 + idx,
                )
            modules.append((weight, lora_a, lora_b, bias))
            leaves.extend([weight, lora_a, lora_b])
            if bias is not None:
                leaves.append(bias)

        def run_iteration():
            total = None
            for weight, lora_a, lora_b, bias in modules:
                out = input_tensor.lora_linear(weight, lora_a, lora_b, scaling, bias)
                value = out.sum()
                total = value if total is None else total.add(value)
            if args.backward:
                total.backward()
            else:
                total.get_plain_text()
            for tensor in leaves:
                _clear_grad(tensor)

        for _ in range(args.warmup):
            run_iteration()

        crypten.reset_communication_stats()
        get_structural_reuse_stats(reset=True)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        for _ in range(args.steps):
            run_iteration()

        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        comm_stats = crypten.get_communication_stats()
        reuse_stats = get_structural_reuse_stats(reset=True)

        return {
            "mode": mode,
            "steps": args.steps,
            "warmup": args.warmup,
            "modules": args.modules,
            "batch_tokens": args.batch_tokens,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "lora_rank": args.lora_rank,
            "input_requires_grad": args.input_requires_grad,
            "backward": args.backward,
            "elapsed_seconds": elapsed,
            "seconds_per_step": elapsed / max(1, args.steps),
            "seconds_per_module": elapsed / max(1, args.steps * args.modules),
            "comm_rounds": comm_stats["rounds"],
            "comm_bytes": comm_stats["bytes"],
            "comm_gib": comm_stats["bytes"] / (1024 ** 3),
            "comm_time_seconds": comm_stats["time"],
            "structural_reuse": reuse_stats,
        }
    finally:
        FUNCTION_REGISTRY["lora_linear"] = original_lora_fn


def worker(args):
    rank = comm.get().get_rank()
    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    if args.device == "cuda":
        torch.cuda.set_device(gpu_ids[rank % len(gpu_ids)])
        device = torch.device(f"cuda:{gpu_ids[rank % len(gpu_ids)]}")
    else:
        device = torch.device("cpu")

    modes = ["token", "grouped_no_token"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        results.append(_run_one_mode(args, mode, device))

    if rank == 0:
        output = {
            "device": str(device),
            "capability": structural_reuse_capability(),
            "results": results,
        }
        print(json.dumps(output, indent=2), flush=True)
        if args.output_json:
            with open(args.output_json, "w") as file_obj:
                json.dump(output, file_obj, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CrypTen LoRA token reuse overhead.")
    parser.add_argument("--mode", choices=["token", "grouped_no_token", "both"], default="both")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--modules", type=int, default=4)
    parser.add_argument("--batch_tokens", type=int, default=512)
    parser.add_argument("--in_features", type=int, default=128)
    parser.add_argument("--out_features", type=int, default=128)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--input_requires_grad", type=_str2bool, default=True)
    parser.add_argument("--backward", type=_str2bool, default=True)
    parser.add_argument("--bias", type=_str2bool, default=False)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    launcher = MultiProcessLauncher(parsed_args.world_size, worker, parsed_args)
    launcher.start()
    launcher.join()
    launcher.terminate()
