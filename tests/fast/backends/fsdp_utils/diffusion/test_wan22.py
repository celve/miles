"""Wan2.2-T2V-A14B's family config: two experts split at t = 875, each guided as the engine guides it."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils.diffusion.pipeline_config import (
    Wan22TrainPipelineConfig,
    get_train_pipeline_config,
    resolve_model_family,
)
from miles.utils.diffusion.types import CondKwargs


def test_the_family_resolves_from_the_diffusers_checkpoint_name():
    assert resolve_model_family("/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers") == "wan2_2"


@pytest.mark.parametrize(
    "timestep,expert", [(999.0, "transformer"), (875.0, "transformer"), (874.9999, "transformer_2")]
)
def test_the_high_noise_expert_takes_timesteps_from_the_boundary_up(timestep, expert):
    assert Wan22TrainPipelineConfig().component_for_timestep(timestep) == expert


def test_the_low_noise_expert_is_guided_by_the_second_scale():
    config, args = Wan22TrainPipelineConfig(), SimpleNamespace(
        diffusion_guidance_scale=4.0, diffusion_guidance_scale_2=3.0
    )

    assert [config.guidance_scale(c, args) for c in config.components] == [4.0, 3.0]


def test_only_the_umt5_embeddings_condition_the_forward():
    config = get_train_pipeline_config("wan2_2")
    cond = CondKwargs(encoder_hidden_states=[torch.ones(512, 8)], pooled_projections=None)

    kwargs = [config.prepare_cond_kwargs(cond, torch.device("cpu")) for _ in range(2)]

    assert config.collate_cond(kwargs, torch.device("cpu"))["encoder_hidden_states"].shape == (2, 512, 8)


def test_the_gemm_patchify_equals_the_conv_and_falls_back_off_its_tiling():
    torch.manual_seed(0)
    conv = torch.nn.Conv3d(4, 6, kernel_size=(1, 2, 2), stride=(1, 2, 2)).double()
    model = torch.nn.Module()
    model.patch_embedding = conv
    tiled, untiled = torch.randn(2, 4, 3, 4, 6).double(), torch.randn(1, 4, 1, 3, 4).double()
    expected = [conv(tiled), conv(untiled)]

    Wan22TrainPipelineConfig().postprocess_model_after_materialize(model)

    torch.testing.assert_close(model.patch_embedding(tiled), expected[0])
    torch.testing.assert_close(model.patch_embedding(untiled), expected[1])
