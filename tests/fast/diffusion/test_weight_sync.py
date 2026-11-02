"""The multimodal_gen LoRA protocol under the common WeightUpdater: reconnects on a replaced engine, and a
failed bucket stops the sync instead of leaving a partly merged adapter unnoticed."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.diffusion.train import weight_sync
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
            yield [(f"{module}.lora_A", torch.ones(2, 4)), (f"{module}.lora_B", torch.ones(4, 2))]


@pytest.fixture
def single_rank(tmp_path, monkeypatch):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", dist.new_group(backend="gloo"))
    monkeypatch.setattr(weight_sync, "get_physical_gpu_id", lambda: "GPU-test")
    # The UUID reduction patch is process-wide and handles only CUDA tensors.
    monkeypatch.setattr(weight_sync, "monkey_patch_torch_reductions", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    yield
    dist.destroy_process_group()


def _updater():
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
        torch.nn.Linear(1, 1),
        weights_getter=lambda: None,
        model_name="sd3",
        quantization_config=None,
        iterator_factory=_TwoBucketIterator.build,
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
