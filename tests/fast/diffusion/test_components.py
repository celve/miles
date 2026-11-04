"""Families whose pipeline denoises with several components (Wan2.2's two experts) route each micro-batch to one."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.fsdp_utils.diffusion import actor as diffusion_actor
from miles.backends.fsdp_utils.diffusion.pipeline_config import SD3TrainPipelineConfig


class _TwoExpertConfig(SD3TrainPipelineConfig):
    model_family = "two_expert"
    components = ("transformer", "transformer_2")
    guidance_scale_2_component = "transformer_2"

    def component_for_timestep(self, timestep: float) -> str:
        return "transformer" if timestep >= 875 else "transformer_2"


def test_a_single_component_family_routes_every_timestep_to_it():
    assert SD3TrainPipelineConfig().component_for_timesteps(torch.tensor([999.0, 1.0])) == "transformer"


def test_a_micro_batch_runs_the_component_of_its_timesteps():
    config = _TwoExpertConfig()

    assert config.component_for_timesteps(torch.tensor([900.0, 875.0])) == "transformer"
    assert config.component_for_timesteps(torch.tensor([874.9999])) == "transformer_2"
    with pytest.raises(ValueError, match="mixes the timesteps"):
        config.component_for_timesteps(torch.tensor([900.0, 500.0]))


@pytest.mark.parametrize("component,model,scale", [("transformer", "high", 4.0), ("transformer_2", "low", 3.0)])
def test_the_trainer_forwards_the_component_with_the_scale_the_engine_guided_it_with(
    monkeypatch, component, model, scale
):
    seen = []
    config = _TwoExpertConfig()
    monkeypatch.setattr(config, "compute_noise_pred", lambda **kwargs: seen.append(kwargs))
    actor = object.__new__(diffusion_actor.DiffusionFSDPTrainRayActor)
    actor.forward_dtype = torch.float32
    actor.args = SimpleNamespace(diffusion_guidance_scale=4.0, diffusion_guidance_scale_2=3.0)
    actor.models = {"transformer": "high", "transformer_2": "low"}
    actor.pipeline_config = config

    actor.noise_pred(component, None, None, {}, None)

    assert [(kwargs["model"], kwargs["guidance_scale"]) for kwargs in seen] == [(model, scale)]


def _check_on_rank(rank, world_size, port, components_per_rank, raised):
    from tests.fast.dist_utils import init_gloo

    from miles.backends.fsdp_utils.diffusion.schedule import check_same_component_on_every_rank

    init_gloo(rank, world_size, port=port)
    try:
        check_same_component_on_every_rank(
            components_per_rank[rank],
            components=("transformer", "transformer_2"),
            group=None,
            device=torch.device("cpu"),
        )
    except RuntimeError as error:
        raised[rank] = str(error)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "components_per_rank,expect_error",
    [(("transformer", "transformer"), False), (("transformer", "transformer_2"), True)],
)
def test_ranks_that_would_forward_different_components_stop(components_per_rank, expect_error):
    import torch.multiprocessing as mp

    from tests.fast.dist_utils import find_free_port

    raised = mp.Manager().dict()
    mp.spawn(_check_on_rank, args=(2, find_free_port(), components_per_rank, raised), nprocs=2, join=True)

    assert len(raised) == (2 if expect_error else 0)
    assert all("['transformer', 'transformer_2']" in message for message in raised.values())
