from types import SimpleNamespace

import pytest
import torch
import torch.distributed.fsdp as fsdp

from miles.diffusion.train import model as dit_model
from miles.diffusion.train.pipeline_config import SD3TrainPipelineConfig


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


def test_fp32_modules_get_their_own_fp32_group_before_the_blocks_are_sharded(monkeypatch):
    class Fp32Norms(SD3TrainPipelineConfig):
        fp32_module_patterns = ("*_norm",)

    calls, _ = _load(monkeypatch, config=Fp32Norms())

    assert [type(module).__name__ for module, _ in calls] == ["LayerNorm", "Linear", "_PeftDiT"]
    assert [kwargs["mp_policy"].param_dtype for _, kwargs in calls] == [torch.float32, torch.float16, torch.float16]
    assert calls[0][1]["mp_policy"].reduce_dtype is torch.float32 and not calls[0][1]["mp_policy"].cast_forward_inputs
