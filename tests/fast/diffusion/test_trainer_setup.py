from types import SimpleNamespace

import pytest
import torch
import torch.distributed.fsdp as fsdp
from tests.fast.backends.fsdp_utils.checkpoint_fixtures import cpu_checkpoint_actor
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.fsdp_utils.diffusion import model as dit_model
from miles.backends.fsdp_utils.diffusion.pipeline_config import SD3TrainPipelineConfig

__all__ = ["cpu_checkpoint_actor"]


class _PeftDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(2, 2)
        self.block_norm = torch.nn.LayerNorm(2)

    def get_base_model(self):
        return SimpleNamespace(_no_split_modules=["Linear"])


def _load(monkeypatch, *, config, cpu_offload=False):
    calls = []
    monkeypatch.setattr(fsdp, "fully_shard", lambda module, **kwargs: calls.append((module, kwargs)))
    monkeypatch.setattr(dit_model.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dit_model, "_load_component", lambda *a, **k: _PeftDiT())
    monkeypatch.setattr(dit_model, "_apply_lora", lambda model, args, materialize: model)
    monkeypatch.setattr(dit_model, "_sync_dtypes_from_rank0", lambda model: None)
    loaded = []
    monkeypatch.setattr(
        dit_model,
        "fsdp2_load_full_state_dict",
        lambda model, state, *, cpu_offload: loaded.append(cpu_offload) or model,
    )
    args = SimpleNamespace(
        hf_checkpoint="/ckpt",
        diffusion_forward_dtype="fp16",
        backend=SimpleNamespace(gradient_checkpointing=False, fp16=False, optimizer_cpu_offload=cpu_offload),
    )

    with pytest.raises(AttributeError, match="_get_fsdp_state"):
        dit_model.load_fsdp_dit(args, config=config, mesh="mesh")
    return calls, loaded


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_the_dit_is_sharded_by_miles_apply_fsdp2_without_casting_inputs(monkeypatch, cpu_offload):
    calls, loaded = _load(monkeypatch, config=SD3TrainPipelineConfig(), cpu_offload=cpu_offload)

    assert [type(module).__name__ for module, _ in calls] == ["Linear", "_PeftDiT"]
    assert all(
        kwargs["mesh"] == "mesh"
        and kwargs["mp_policy"].param_dtype is torch.float16
        and kwargs["mp_policy"].reduce_dtype is torch.float32
        and not kwargs["mp_policy"].cast_forward_inputs
        and isinstance(kwargs["offload_policy"], fsdp.CPUOffloadPolicy) == cpu_offload
        for _, kwargs in calls
    )
    assert loaded == [cpu_offload]


@pytest.mark.parametrize(
    "save_optim, load_optim, saved_scaler_enabled, resumed_scale",
    [
        (True, True, True, 1024.0),
        (False, False, True, 65536.0),
        (True, False, True, 65536.0),
        (True, True, False, 65536.0),
    ],
)
def test_a_resumed_fp16_run_keeps_its_loss_scale_with_its_optimizer(
    cpu_checkpoint_actor, tmp_path, save_optim, load_optim, saved_scaler_enabled, resumed_scale
):
    actor = cpu_checkpoint_actor
    del actor.lr_scheduler
    actor.args.backend = FsdpArgsNamespace(
        load=str(tmp_path),
        save=str(tmp_path),
        no_save_optim=not save_optim,
        no_load_optim=not load_optim,
        no_load_rng=True,
        ckpt_step=None,
    )
    actor.scaler = ShardedGradScaler(device="cpu", enabled=saved_scaler_enabled)
    actor.scaler.scale(torch.ones(1))
    actor.scaler.update(new_scale=1024.0)
    checkpoint.save(actor=actor, iteration=0)
    scaler_file = tmp_path / "iter_0000001" / checkpoint.GRAD_SCALER_FILENAME
    assert scaler_file.exists() is (save_optim and saved_scaler_enabled)

    actor.scaler = ShardedGradScaler(device="cpu")
    checkpoint.finalize_load(actor=actor, checkpoint_payload=checkpoint.load(actor=actor))

    assert actor.scaler.get_scale() == resumed_scale


def test_fp32_modules_get_their_own_fp32_group_before_the_blocks_are_sharded(monkeypatch):
    class Fp32Norms(SD3TrainPipelineConfig):
        fp32_module_patterns = ("*_norm",)

    calls, _ = _load(monkeypatch, config=Fp32Norms())

    assert [type(module).__name__ for module, _ in calls] == ["LayerNorm", "Linear", "_PeftDiT"]
    assert [kwargs["mp_policy"].param_dtype for _, kwargs in calls] == [torch.float32, torch.float16, torch.float16]
    assert calls[0][1]["mp_policy"].reduce_dtype is torch.float32 and not calls[0][1]["mp_policy"].cast_forward_inputs
