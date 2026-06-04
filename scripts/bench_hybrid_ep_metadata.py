#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Benchmark Hybrid-EP metadata preprocessing paths.

This helper is intentionally outside DeepEP so it can be reused across branch
checkouts without modifying the branch under test.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist


def _add_deepep_paths(deepep_path: str) -> None:
    sys.path.insert(0, deepep_path)
    sys.path.insert(0, os.path.join(deepep_path, "tests"))


def _bench(fn, num_warmups: int = 20, num_tests: int = 30) -> float:
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    return sum(times[1:]) / max(len(times) - 1, 1)


def _gather_time(t: float) -> list[float]:
    tensor = torch.tensor([t], device="cuda", dtype=torch.float64)
    gathered = [torch.zeros(1, device="cuda", dtype=torch.float64) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [x.item() for x in gathered]


def _run_rank(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    _add_deepep_paths(args.deepep_path)

    import deep_ep
    from utils import init_dist
    from test_hybrid_ep import init_tensor

    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    num_experts = args.num_local_experts * world_size
    _, _, _, routing_map, _, _ = init_tensor(
        hidden_dim=args.hidden_dim,
        seq_len=args.num_tokens,
        topk=args.topk,
        num_of_experts=num_experts,
        use_fp8=False,
    )

    buffer = deep_ep.HybridEPBuffer(
        group=group,
        hidden_dim=args.hidden_dim,
        max_num_of_tokens_per_rank=args.max_num_tokens,
        num_local_experts=args.num_local_experts,
        use_fp8=False,
        num_sms_dispatch_api=args.num_sms_dispatch,
        num_sms_combine_api=args.num_sms_combine,
        num_sms_preprocessing_api=args.num_sms_preprocessing,
        num_blocks_permute=args.num_blocks_permute,
        num_blocks_unpermute=args.num_blocks_unpermute,
    )

    if rank == 0:
        print("\n=== Metadata Preprocessing Benchmark (BF16) ===", flush=True)
        print(
            f"  H={args.hidden_dim}, T={args.num_tokens}, E/rank={args.num_local_experts}, "
            f"TOPK={args.topk}, ranks={world_size}, pad={args.pad_multiple}",
            flush=True,
        )

    if args.mode == "both":
        fuse_modes = (False, True)
    elif args.mode == "standalone":
        fuse_modes = (False,)
    else:
        fuse_modes = (True,)

    for fuse in fuse_modes:
        config = buffer.update_template_config(
            hidden_dim=args.hidden_dim,
            num_of_tokens_per_rank=args.num_tokens,
            num_local_experts=args.num_local_experts,
            pad_multiple=args.pad_multiple,
            use_fp8=False,
            fuse_permute_dispatch=fuse,
            topk=0,
        )

        def run_metadata():
            return buffer.runtime.metadata_preprocessing(
                config=config,
                routing_map=routing_map,
                num_of_tokens_per_rank=args.num_tokens,
                num_permuted_tokens=None,
                pad_multiple=args.pad_multiple,
                enable_permute=True,
                fuse_permute_dispatch=fuse,
                non_blocking=False,
            )

        t = _bench(run_metadata, args.num_warmups, args.num_tests)
        times = _gather_time(t)
        if rank == 0:
            label = "scan+permute-preprocessing fused" if fuse else "scan+permute-preprocessing standalone"
            print(
                f"  {label + ':':<48} avg={sum(times) / len(times) * 1e6:.1f} us "
                f"[min={min(times) * 1e6:.1f}, max={max(times) * 1e6:.1f}]",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepep-path", default="/home/scratch.hhanyu_gpu/projects/moe/DeepEP")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "512")))
    parser.add_argument("--num-tokens", type=int, default=int(os.getenv("NUM_TOKENS_PER_RANK", "8192")))
    parser.add_argument("--max-num-tokens", type=int, default=int(os.getenv("MAX_NUM_OF_TOKENS_PER_RANK", "8192")))
    parser.add_argument("--num-local-experts", type=int, default=int(os.getenv("NUM_LOCAL_EXPERTS", "32")))
    parser.add_argument("--topk", type=int, default=int(os.getenv("TOPK", "36")))
    parser.add_argument("--pad-multiple", type=int, default=int(os.getenv("PAD_MULTIPLE", "32")))
    parser.add_argument("--num-sms-dispatch", type=int, default=int(os.getenv("NUM_SMS_DISPATCH", "32")))
    parser.add_argument("--num-sms-combine", type=int, default=int(os.getenv("NUM_SMS_COMBINE", "32")))
    parser.add_argument("--num-sms-preprocessing", type=int, default=int(os.getenv("NUM_SMS_PREPROCESSING", "108")))
    parser.add_argument("--num-blocks-permute", type=int, default=int(os.getenv("NUM_BLOCKS_PERMUTE", "0")) or None)
    parser.add_argument("--num-blocks-unpermute", type=int, default=int(os.getenv("NUM_BLOCKS_UNPERMUTE", "0")) or None)
    parser.add_argument("--num-warmups", type=int, default=20)
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "1025")))
    parser.add_argument(
        "--mode",
        choices=("standalone", "fused", "both"),
        default="both",
        help="Which metadata preprocessing path(s) to benchmark.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29700 + os.getpid() % 1000))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    torch.multiprocessing.spawn(_run_rank, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
