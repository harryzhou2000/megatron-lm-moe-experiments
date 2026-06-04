#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Benchmark Hybrid-EP dense top-k metadata preprocessing only."""

import argparse
import os
import sys

import torch
import torch.distributed as dist


def _add_deepep_paths(deepep_path: str) -> None:
    sys.path.insert(0, deepep_path)
    sys.path.insert(0, os.path.join(deepep_path, "tests"))


def _bench(fn, warmups: int, tests: int) -> float:
    torch.cuda.synchronize()
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(tests)]
    for i in range(tests):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    return sum(times[1:]) / max(len(times) - 1, 1)


def _run(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    _add_deepep_paths(args.deepep_path)

    import deep_ep
    from utils import init_dist

    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    assert world_size == args.num_processes
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    routing_rows = args.num_tokens * (args.fake_ranks_per_node or 1)
    topk_idx = torch.empty((routing_rows, args.topk), dtype=torch.int16, device="cuda")
    for i in range(routing_rows):
        selected = torch.randperm(args.num_total_experts, device="cuda")[: args.topk]
        topk_idx[i] = selected.to(torch.int16)

    buffer = deep_ep.HybridEPBuffer(
        group=group,
        hidden_dim=args.hidden_dim,
        max_num_of_tokens_per_rank=args.max_num_tokens,
        num_local_experts=args.num_local_experts,
        use_fp8=False,
        enable_custom_allgather=False,
        num_sms_dispatch_api=args.num_sms_dispatch,
        num_sms_combine_api=args.num_sms_combine,
        num_sms_preprocessing_api=args.num_sms_preprocessing,
    )
    config = buffer.update_template_config(
        hidden_dim=args.hidden_dim,
        num_of_tokens_per_rank=args.num_tokens,
        num_local_experts=args.num_local_experts,
        pad_multiple=args.pad_multiple,
        use_fp8=False,
        fuse_permute_dispatch=True,
        topk=args.topk,
        **(
            {"num_of_ranks_per_node": args.fake_ranks_per_node}
            if args.fake_ranks_per_node is not None
            else {}
        ),
    )

    def run_metadata():
        return buffer.runtime.metadata_preprocessing(
            config=config,
            routing_map=topk_idx,
            num_of_tokens_per_rank=args.num_tokens,
            num_permuted_tokens=None,
            pad_multiple=args.pad_multiple,
            enable_permute=True,
            fuse_permute_dispatch=True,
            non_blocking=False,
        )

    t = _bench(run_metadata, args.num_warmups, args.num_tests)
    t_tensor = torch.tensor([t], device="cuda", dtype=torch.float64)
    gathered = [torch.zeros(1, device="cuda", dtype=torch.float64) for _ in range(world_size)]
    dist.all_gather(gathered, t_tensor)
    if rank == 0:
        times = [x.item() for x in gathered]
        print(
            "dense scan+permute-preprocessing fused: "
            f"avg={sum(times) / len(times) * 1e6:.1f} us "
            f"[min={min(times) * 1e6:.1f}, max={max(times) * 1e6:.1f}] "
            f"T={args.num_tokens} localE={args.num_local_experts} "
            f"totalE={args.num_total_experts} topk={args.topk} "
            f"templateR={args.fake_ranks_per_node or world_size} rows={routing_rows} "
            f"threads={os.getenv('NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API', 'default')} "
            f"blocks={args.num_sms_preprocessing}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepep-path", default="/home/scratch.hhanyu_gpu/projects/moe/DeepEP")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=4096 * 3)
    parser.add_argument("--max-num-tokens", type=int, default=4096 * 3)
    parser.add_argument("--num-local-experts", type=int, default=32)
    parser.add_argument("--num-total-experts", type=int, default=2304)
    parser.add_argument(
        "--fake-ranks-per-node",
        type=int,
        default=None,
        help=(
            "Override the scan template's ranks-per-node and generate a pre-gathered "
            "routing tensor with num_tokens * fake_ranks_per_node rows. Useful for "
            "single-process NVL72 scan microbenchmarks."
        ),
    )
    parser.add_argument("--topk", type=int, default=36)
    parser.add_argument("--pad-multiple", type=int, default=32)
    parser.add_argument("--num-sms-dispatch", type=int, default=32)
    parser.add_argument("--num-sms-combine", type=int, default=32)
    parser.add_argument("--num-sms-preprocessing", type=int, default=108)
    parser.add_argument("--num-warmups", type=int, default=50)
    parser.add_argument("--num-tests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1025)
    args = parser.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29800 + os.getpid() % 1000))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    torch.multiprocessing.spawn(_run, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
