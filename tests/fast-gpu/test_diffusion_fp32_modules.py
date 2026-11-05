"""A diffusion family's fp32 modules (Wan2.2's cross-attention norms) are gathered in fp32 inside bf16 FSDP2 blocks."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["fsdp"], hardware=["hopper", "blackwell"], num_gpus=1
)

import copy

import pytest
import torch
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from miles.diffusion.train.model import shard_fp32_modules

DIM = 64


def _block() -> WanTransformerBlock:
    torch.manual_seed(0)
    block = WanTransformerBlock(DIM, ffn_dim=2 * DIM, num_heads=4, cross_attn_norm=True).cuda()
    with torch.no_grad():
        # Weights bf16 cannot hold, so a bf16 gather of the norm would show.
        block.norm2.weight.copy_(1 + 1e-3 * torch.randn(DIM))
        block.norm2.bias.copy_(1e-3 * torch.randn(DIM))
    return block


def _forward(block) -> torch.Tensor:
    torch.manual_seed(1)
    hidden, context, temb = (torch.randn(shape, device="cuda") for shape in ((1, 16, DIM), (1, 8, DIM), (1, 6, DIM)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return block(hidden, context, temb, rotary_emb=None)


def _sharded(block, patterns):
    mesh = init_device_mesh("cuda", (1,))
    shard_fp32_modules(block, patterns, mesh=mesh, cpu_offload=False)
    bf16 = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    fully_shard(block, mesh=mesh, mp_policy=bf16)
    return block


def _bf16_except_norm2(block):
    reference = copy.deepcopy(block).to(torch.bfloat16)
    reference.norm2 = copy.deepcopy(block.norm2)
    return reference


@pytest.mark.usefixtures("nccl_world")
def test_the_matched_norm_is_gathered_in_fp32_and_the_block_in_bf16():
    block = _block()
    expected = _forward(_bf16_except_norm2(block))
    sharded = _sharded(block, ("norm2",))
    dtypes = {}
    sharded.norm2.register_forward_pre_hook(lambda module, _: dtypes.update(norm2=module.weight.dtype))
    sharded.attn2.to_q.register_forward_pre_hook(lambda module, _: dtypes.update(to_q=module.weight.dtype))

    actual = _forward(sharded)

    assert dtypes == {"norm2": torch.float32, "to_q": torch.bfloat16}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.usefixtures("nccl_world")
def test_without_the_pattern_the_norm_is_rounded_to_bf16():
    block = _block()
    expected = _forward(_bf16_except_norm2(block))

    assert not torch.equal(_forward(_sharded(block, ())), expected)
