#!/usr/bin/env python3

import argparse
import statistics
import time

import crypten
import torch
from crypten.common.reuse_context import clear_current_reuse_step, set_current_reuse_step
from crypten.config import cfg
from crypten.mpc.primitives import beaver


class TinyMLP(crypten.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super().__init__()
        self.fc1 = crypten.nn.Linear(in_features, hidden_features, bias=bias)
        self.act = crypten.nn.ReLU()
        self.fc2 = crypten.nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def _mean(values):
    if len(values) == 0:
        return 0.0
    return float(statistics.mean(values))


def _delta_dict(after, before):
    keys = set(before.keys()) | set(after.keys())
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def _configure_mode(experimental_reuse_mask, reuse_mode):
    cfg.mpc.experimental_reuse_mask = experimental_reuse_mask
    cfg.mpc.reuse_mode = reuse_mode
    cfg.mpc.reuse_scope = "STEP"
    cfg.mpc.reuse_op_types = ["matmul"]
    cfg.mpc.reuse_tagging = True


def _create_model(args, device):
    model = TinyMLP(
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        out_features=args.out_features,
        bias=not args.no_bias,
    )
    if device.type == "cuda":
        model = model.cuda(device=device)
    model.encrypt(src=0)
    model.train(True)
    return model


def _prepare_batches(args, device):
    batches = []
    for _ in range(args.steps + args.warmup):
        x_plain = torch.randn(args.batch_size, args.in_features, device=device)
        y_plain = torch.randn(args.batch_size, args.out_features, device=device)
        batches.append((x_plain, y_plain))
    return batches


def _run_mode(args, mode_name, experimental_reuse_mask, reuse_mode, device, batches):
    _configure_mode(experimental_reuse_mask, reuse_mode)
    beaver.reset_reuse_stats(reset_cache=True)
    crypten.reset_communication_stats()

    model = _create_model(args, device)
    optimizer = crypten.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = crypten.nn.MSELoss()

    prep_times = []
    forward_times = []
    backward_times = []
    optim_times = []
    step_times = []
    comm_rounds = []
    reveal_calls = []
    reveal_tensors = []
    triple_calls = []

    for step_idx, (x_plain, y_plain) in enumerate(batches):
        set_current_reuse_step(step_idx)
        beaver.begin_reuse_step(step_idx)
        comm_before = crypten.get_communication_stats()
        beaver_before = beaver.get_reuse_stats()

        try:
            step_start = time.perf_counter()
            prep_start = step_start
            x_enc = crypten.cryptensor(x_plain, src=0, requires_grad=True)
            y_enc = crypten.cryptensor(y_plain, src=0, requires_grad=False)
            optimizer.zero_grad()
            prep_end = time.perf_counter()

            forward_start = prep_end
            output = model(x_enc)
            loss = criterion(output, y_enc)
            forward_end = time.perf_counter()

            backward_start = forward_end
            loss.backward()
            backward_end = time.perf_counter()

            optim_start = backward_end
            optimizer.step()
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

        prep_times.append(prep_end - prep_start)
        forward_times.append(forward_end - forward_start)
        backward_times.append(backward_end - backward_start)
        optim_times.append(step_end - optim_start)
        step_times.append(step_end - step_start)
        comm_rounds.append(comm_delta.get("rounds", 0))
        reveal_calls.append(beaver_delta.get("beaver_reveal_calls", 0))
        reveal_tensors.append(beaver_delta.get("beaver_revealed_tensors", 0))
        triple_calls.append(beaver_delta.get("triple_generate_calls", 0))

    return {
        "mode": mode_name,
        "reuse_enabled": experimental_reuse_mask,
        "reuse_mode": reuse_mode,
        "prep_time_s": _mean(prep_times),
        "forward_time_s": _mean(forward_times),
        "backward_time_s": _mean(backward_times),
        "optim_time_s": _mean(optim_times),
        "step_time_s": _mean(step_times),
        "comm_rounds": _mean(comm_rounds),
        "beaver_reveal_calls": _mean(reveal_calls),
        "beaver_revealed_tensors": _mean(reveal_tensors),
        "triple_generate_calls": _mean(triple_calls),
        "extra_counters": beaver.get_reuse_stats(),
    }


def _worker(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.random.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.random.manual_seed(args.seed)

    cfg.communicator.verbose = args.verbose_comm
    cfg.mpc.provider = args.provider
    device = torch.device(args.device)
    batches = _prepare_batches(args, device=device)

    mode_specs = [
        ("baseline", False, "FIX_A"),
        ("reuse_fix_a", True, "FIX_A"),
    ]
    if args.run_fix_ab:
        mode_specs.append(("reuse_fix_ab", True, "FIX_AB"))

    results = {}
    for mode_name, enabled, reuse_mode in mode_specs:
        results[mode_name] = _run_mode(
            args=args,
            mode_name=mode_name,
            experimental_reuse_mask=enabled,
            reuse_mode=reuse_mode,
            device=device,
            batches=batches,
        )

    return {
        "rank": crypten.communicator.get().get_rank(),
        "provider": cfg.mpc.provider,
        "results": results,
    }


def _build_runner(world_size):
    @crypten.mpc.context.run_multiprocess(world_size)
    def _runner(args):
        return _worker(args)

    return _runner


def _print_summary(rank0_payload, mode_order):
    provider = rank0_payload["provider"]
    print("")
    print(f"Provider: {provider}")
    print("Benchmark: TinyMLP train step = encrypt inputs + forward + MSE loss + backward + SGD")
    print(
        "Columns: mode | prep(s) | fwd(s) | bwd(s) | optim(s) | step(s) | rounds | reveal_calls | reveal_tensors | triple_gen"
    )
    for mode in mode_order:
        if mode not in rank0_payload["results"]:
            continue
        item = rank0_payload["results"][mode]
        print(
            f"{mode:>12} | "
            f"{item['prep_time_s']:.6f} | "
            f"{item['forward_time_s']:.6f} | "
            f"{item['backward_time_s']:.6f} | "
            f"{item['optim_time_s']:.6f} | "
            f"{item['step_time_s']:.6f} | "
            f"{item['comm_rounds']:.2f} | "
            f"{item['beaver_reveal_calls']:.2f} | "
            f"{item['beaver_revealed_tensors']:.2f} | "
            f"{item['triple_generate_calls']:.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight TinyMLP benchmark for Beaver mask reuse."
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--provider", type=str, default="TFP", choices=["TFP", "TTP"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=64)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--out-features", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-fix-ab", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--verbose-comm", action="store_true")
    return parser.parse_args()


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
    _print_summary(rank0_payload, mode_order=["baseline", "reuse_fix_a", "reuse_fix_ab"])


if __name__ == "__main__":
    main()
