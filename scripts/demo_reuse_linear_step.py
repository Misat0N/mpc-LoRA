#!/usr/bin/env python3

import argparse

import crypten
import torch
from crypten.common.reuse_context import clear_current_reuse_step, set_current_reuse_step
from crypten.config import cfg
from crypten.mpc.primitives import beaver


def _configure(experimental_reuse_mask, reuse_mode):
    cfg.mpc.experimental_reuse_mask = experimental_reuse_mask
    cfg.mpc.reuse_mode = reuse_mode
    cfg.mpc.reuse_scope = "STEP"
    cfg.mpc.reuse_op_types = ["matmul"]
    cfg.mpc.reuse_tagging = True


def _run_step(args):
    cfg.mpc.provider = args.provider
    cfg.communicator.verbose = args.verbose_comm
    _configure(args.experimental_reuse_mask, args.reuse_mode)
    beaver.reset_reuse_stats(reset_cache=True)
    crypten.reset_communication_stats()

    device = torch.device(args.device)
    layer = crypten.nn.Linear(args.in_features, args.out_features, bias=not args.no_bias)
    if device.type == "cuda":
        layer = layer.cuda(device=device)
    layer.encrypt(src=0)

    x = torch.randn(args.batch_size, args.in_features, device=device)
    set_current_reuse_step(0)
    beaver.begin_reuse_step(0)
    try:
        x_enc = crypten.cryptensor(x, src=0, requires_grad=True)
        y = layer(x_enc)
        loss = y.sum()
        loss.backward()
    finally:
        beaver.end_reuse_step(0)
        clear_current_reuse_step()

    return {
        "rank": crypten.communicator.get().get_rank(),
        "comm_stats": crypten.get_communication_stats(),
        "reuse_stats": beaver.get_reuse_stats(),
    }


def _build_runner(world_size):
    @crypten.mpc.context.run_multiprocess(world_size)
    def _runner(args):
        return _run_step(args)

    return _runner


def parse_args():
    parser = argparse.ArgumentParser(description="One-step Linear reuse demo.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--provider", type=str, default="TFP", choices=["TFP", "TTP"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--out-features", type=int, default=16)
    parser.add_argument("--experimental-reuse-mask", action="store_true")
    parser.add_argument("--reuse-mode", type=str, default="FIX_A", choices=["FIX_A", "FIX_AB"])
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--verbose-comm", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    runner = _build_runner(args.world_size)
    outputs = runner(args)
    if outputs is None:
        raise RuntimeError("Multiprocess run failed.")

    worker_outputs = [item for item in outputs if isinstance(item, dict) and "rank" in item]
    if len(worker_outputs) == 0:
        raise RuntimeError("No worker output was returned.")
    rank0 = sorted(worker_outputs, key=lambda item: item["rank"])[0]

    print("")
    print("One-step demo (rank0):")
    print(f"comm_stats={rank0['comm_stats']}")
    print(f"reuse_stats={rank0['reuse_stats']}")


if __name__ == "__main__":
    main()
