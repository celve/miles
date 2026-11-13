"""The multimodal_gen LoRA protocol under the common WeightUpdater: reconnects on a replaced engine, and a
failed bucket stops the sync instead of leaving a partly merged adapter unnoticed."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing import reductions

from miles.backends.fsdp_utils.diffusion import weight_sync
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.utils import distributed_utils


class _Engine:
    def __init__(self, name, fail_on_call=None):
        self.name, self.fail_on_call, self.calls = name, fail_on_call, []

    async def update_lora_weights_from_tensor(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError(f"{self.name} rejected bucket {len(self.calls)}")
        return {"success": True}


class _TwoBucketIterator(HfWeightIteratorBase):
    def _iter_hf_param_units(self, weights, *, materialize):
        raise AssertionError("only the adapter is synced")

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        for module in ("blocks.0.attn.to_q", "blocks.1.attn.to_q"):
            yield [
                (f"transformer/{module}.lora_A", torch.ones(2, 4)),
                (f"transformer/{module}.lora_B", torch.ones(4, 2)),
            ]


@pytest.fixture
def single_rank(tmp_path, monkeypatch):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", dist.new_group(backend="gloo"))
    monkeypatch.setattr(weight_sync, "get_physical_gpu_id", lambda: "GPU-test")
    # The UUID reduction patch is process-wide and handles only CUDA tensors; an earlier test may have applied it.
    monkeypatch.setattr(weight_sync, "monkey_patch_torch_reductions", lambda: None)
    patched_reduce = reductions.reduce_tensor
    reductions.reduce_tensor = getattr(reductions, "_reduce_tensor_original", patched_reduce)
    reductions.init_reductions()
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    yield
    reductions.reduce_tensor = patched_reduce
    reductions.init_reductions()
    dist.destroy_process_group()


def _updater(model=None, iterator_factory=_TwoBucketIterator.build):
    args = Namespace(
        model_kind="diffusion",
        update_weight_buffer_size=1,
        update_weight_engine_request_timeout=30,
        ft_components=[],
        diffusion_lora_rank=2,
        diffusion_lora_alpha=4,
        check_lora_weight_equal=False,
        enable_mtp_training=False,
        sglang=SimpleNamespace(base_args={"speculative_algorithm": None}),
    )
    return WeightUpdater(
        args,
        torch.nn.Linear(1, 1) if model is None else model,
        weights_getter=lambda: None,
        model_name="sd3",
        quantization_config=None,
        iterator_factory=iterator_factory,
        parallel_state=None,
        is_lora=True,
        lora_sync_config={"r": 2, "lora_alpha": 4},
        protocol=weight_sync.DiffusionLoraFromTensor(args),
    )


def _engines(engine, generation_hash):
    return UpdatableEngines(
        rollout_engines=[engine],
        engine_gpu_counts=[1],
        engine_gpu_offsets=[0],
        engine_cell_ids=["cell-0"],
        snapshot_cell_id_to_hashes={"cell-0": generation_hash},
    )


def test_a_replaced_engine_in_the_same_cell_gets_the_next_sync(single_rank):
    updater = _updater()
    first, replacement = _Engine("first"), _Engine("replacement")

    assert updater.reconnect_if_needed(_engines(first, "gen-1"))
    assert updater.update_weights(weight_version=1)
    assert not updater.reconnect_if_needed(_engines(first, "gen-1"))
    assert updater.reconnect_if_needed(_engines(replacement, "gen-2"))
    assert updater.update_weights(weight_version=2)

    assert len(first.calls) == 2 and len(replacement.calls) == 2
    call = replacement.calls[0]
    assert call["target_module"] == "transformer" and (call["lora_rank"], call["lora_alpha"]) == (2, 4)


def test_a_failed_bucket_stops_the_sync(single_rank):
    updater = _updater()
    engine = _Engine("engine", fail_on_call=2)
    updater.reconnect_if_needed(_engines(engine, "gen-1"))

    with pytest.raises(RuntimeError, match="rejected bucket 2"):
        updater.update_weights(weight_version=1)
    assert len(engine.calls) == 2


def _peft_adapter(value: float) -> torch.nn.Module:
    """The state-dict names PEFT gives one LoRA module: ``to_q.lora_A.default.weight``."""
    lora = torch.nn.Module()
    lora.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
    lora.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})
    torch.nn.init.constant_(lora.lora_A["default"].weight, value)
    torch.nn.init.constant_(lora.lora_B["default"].weight, value)
    model = torch.nn.Module()
    model.to_q = lora
    return model


def _sent(call) -> dict[str, list[str]]:
    (payload,) = call["serialized_named_tensors"]
    return {component: [meta.name for meta in body["metadata"]] for component, body in payload.items()}


@pytest.mark.parametrize("components", [["transformer"], ["transformer", "transformer_2"]])
def test_each_component_adapter_reaches_the_engine_component_of_its_name(single_rank, monkeypatch, components):
    # The payload stays readable in-process; a CUDA IPC handle cannot be reopened by the process that made it.
    monkeypatch.setattr(weight_sync.MultiprocessingSerializer, "serialize", lambda obj, output_str: obj)
    models = {component: _peft_adapter(index + 1.0) for index, component in enumerate(components)}
    updater = _updater(models, iterator_factory=weight_sync.DiffusionLoraWeightIterator.build)
    updater.args.update_weight_buffer_size = 1 << 30
    engine = _Engine("engine")
    updater.reconnect_if_needed(_engines(engine, "gen-1"))

    assert updater.update_weights(weight_version=1)

    assert [call["target_module"] for call in engine.calls] == components
    assert [_sent(call) for call in engine.calls] == [
        {component: ["to_q.lora_A", "to_q.lora_B"]} for component in components
    ]
