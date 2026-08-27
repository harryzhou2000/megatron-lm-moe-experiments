#!/usr/bin/env python3
"""Private distributed smoke test for K3 latent MoE with decoupled ETP.

Run from a Megatron-LM checkout with two GPUs, for example:

    PYTHONPATH=. python -m torch.distributed.run --standalone --nproc-per-node=2 \
        ../scripts/test_k3_latent_rmsnorm_etp.py --dispatcher alltoall

This is intentionally kept outside Megatron-LM's unit-test tree. It verifies that the
source-side latent projections stay local while the routed expert MLP uses ETP=2.
"""

import argparse
import importlib.metadata
import os
from functools import partial

import torch

if os.getenv("MCORE_PRIVATE_DISABLE_FLASH_ATTN_4_IMPORT", "0") == "1":
    _package_version = importlib.metadata.version

    def _package_version_without_flash_attn_4(distribution_name: str) -> str:
        if distribution_name == "flash-attn-4":
            raise importlib.metadata.PackageNotFoundError(distribution_name)
        return _package_version(distribution_name)

    # This smoke test does not exercise attention. Allow it to run in containers whose
    # installed FlashAttention-4 package is incompatible with their CuTe DSL version.
    importlib.metadata.version = _package_version_without_flash_attn_4

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERMSNormDuplicatedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.transformer_config import TransformerConfig


def parse_args() -> argparse.Namespace:
    """Parse private-test arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dispatcher", choices=("alltoall", "allgather"), default="alltoall"
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--ffn-hidden-size", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    return parser.parse_args()


def assert_finite(name: str, tensor: torch.Tensor | None) -> None:
    """Require a populated finite tensor."""
    if tensor is None:
        raise AssertionError(f"{name} is None")
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values")


def main() -> None:
    """Run a full latent-MoE forward/backward with TP=1 and ETP=world size."""
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise ValueError(
            f"This focused test requires exactly two ranks, got {world_size}."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=world_size,
    )
    rank = torch.distributed.get_rank()
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123, force_reset_rng=True)

    try:
        # Keep this test focused on ETP dispatch/expert/combine and the source-side projection.
        # Some private containers carry a TE router GEMM binding from a different PyTorch build.
        moe_utils.te_general_gemm = None
        config = TransformerConfig(
            num_layers=1,
            hidden_size=args.hidden_size,
            num_attention_heads=4,
            num_moe_experts=args.num_experts,
            moe_token_dispatcher_type=args.dispatcher,
            moe_router_topk=args.topk,
            moe_router_load_balancing_type="none",
            moe_grouped_gemm=True,
            moe_ffn_hidden_size=args.ffn_hidden_size,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=world_size,
            moe_latent_size=args.latent_size,
            moe_latent_up_projection_rmsnorm=True,
        )
        submodules = MoESubmodules(
            experts=partial(
                TEGroupedMLP,
                submodules=GroupedMLPSubmodules(
                    linear_fc1=TEColumnParallelGroupedLinear,
                    linear_fc2=TERowParallelGroupedLinear,
                ),
            )
        )

        layer = MoELayer(config, submodules).cuda().train()
        if parallel_state.get_tensor_model_parallel_world_size() != 1:
            raise AssertionError("The source-side tensor-parallel size must be one.")
        if parallel_state.get_expert_tensor_parallel_world_size() != world_size:
            raise AssertionError(
                "The expert MLP did not receive the two-rank ETP group."
            )
        if layer.token_dispatcher.tp_group.size() != world_size:
            raise AssertionError("The dispatcher is not using the ETP group.")
        if not isinstance(layer.fc2_latent_proj, TERMSNormDuplicatedLinear):
            raise AssertionError("The latent up-projection is not RMSNorm+Linear.")
        if (
            layer.fc2_latent_proj.tp_size != 1
            or layer.fc2_latent_proj._tp_group.size() != 1
        ):
            raise AssertionError(
                "The source-side up-projection unexpectedly joined ETP."
            )

        expert_fc1 = layer.experts.linear_fc1.weight0
        expert_fc2 = layer.experts.linear_fc2.weight0
        expected_fc1_shape = (
            2 * args.ffn_hidden_size // world_size,
            args.latent_size,
        )
        expected_fc2_shape = (
            args.latent_size,
            args.ffn_hidden_size // world_size,
        )
        if tuple(expert_fc1.shape) != expected_fc1_shape:
            raise AssertionError(
                f"Expert FC1 is not ETP-sharded: {tuple(expert_fc1.shape)} != {expected_fc1_shape}"
            )
        if tuple(expert_fc2.shape) != expected_fc2_shape:
            raise AssertionError(
                f"Expert FC2 is not ETP-sharded: {tuple(expert_fc2.shape)} != {expected_fc2_shape}"
            )

        up_projection_inputs = []

        def record_up_projection_input(_module, inputs):
            up_projection_inputs.append(inputs[0])

        hook = layer.fc2_latent_proj.register_forward_pre_hook(
            record_up_projection_input
        )
        hidden_states = (
            torch.randn(
                args.sequence_length,
                args.micro_batch_size,
                args.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            + rank
        ).requires_grad_(True)
        output, output_bias = layer(hidden_states)
        hook.remove()

        if output_bias is not None:
            raise AssertionError("Expected no MoE output bias.")
        if tuple(output.shape) != tuple(hidden_states.shape):
            raise AssertionError(
                f"Output shape {tuple(output.shape)} != {tuple(hidden_states.shape)}"
            )
        if len(up_projection_inputs) != 1:
            raise AssertionError(
                f"Expected one source-side up-projection call, got {len(up_projection_inputs)}"
            )
        expected_up_input_shape = (
            args.sequence_length,
            args.micro_batch_size,
            args.latent_size,
        )
        if tuple(up_projection_inputs[0].shape) != expected_up_input_shape:
            raise AssertionError(
                "The up-projection did not receive the full source-token latent tensor: "
                f"{tuple(up_projection_inputs[0].shape)} != {expected_up_input_shape}"
            )

        loss = output.float().square().mean()
        loss.backward()
        assert_finite("output", output)
        assert_finite("input gradient", hidden_states.grad)
        assert_finite(
            "RMSNorm scale gradient", layer.fc2_latent_proj.layer_norm_weight.grad
        )
        assert_finite(
            "up-projection weight gradient", layer.fc2_latent_proj.weight.grad
        )

        torch.cuda.synchronize()
        print(
            f"rank={rank} PASS dispatcher={args.dispatcher} TP=1 ETP={world_size} "
            f"expert_fc1={tuple(expert_fc1.shape)} expert_fc2={tuple(expert_fc2.shape)} "
            f"up_input={tuple(up_projection_inputs[0].shape)} output={tuple(output.shape)}",
            flush=True,
        )
    finally:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
