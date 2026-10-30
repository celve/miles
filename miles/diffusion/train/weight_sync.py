"""Colocated LoRA weight sync to sglang.multimodal_gen, driven by the common `WeightUpdater`.

Each training rank all-gathers the adapter's A/B pairs and hands its own engine CUDA-IPC handles; the engine
merges ``W + alpha/r * B @ A`` into its weights (``weight_update_mode=lora_merge``).
"""

import logging
import re
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase, WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters
from miles.ray.utils import get_physical_gpu_id
from miles.utils import async_utils

logger = logging.getLogger(__name__)

DIT_COMPONENT = "transformer"
# PEFT appends the adapter name and ".weight" after lora_A / lora_B.
_PEFT_LORA_SUFFIX = re.compile(r"\.(lora_[AB])(?:\.[^.]+)?(?:\.weight)?$")


def to_hf_name(train_name: str) -> str:
    """base_model.model.blocks.0.attn.to_q.lora_A.default.weight -> blocks.0.attn.to_q.lora_A"""
    name = train_name.removeprefix("base_model.model.").replace(".base_layer.", ".")
    return _PEFT_LORA_SUFFIX.sub(r".\1", name)


class DiffusionLoraWeightIterator(HfWeightIteratorBase):
    """Streams the PEFT adapter of an FSDP2 DiT; one unit per LoRA module, its A and B together, since the
    engine merges each module from both halves."""

    def _iter_hf_param_units(self, weights, *, materialize: bool):
        raise NotImplementedError("the diffusion engine keeps its base weights; only the LoRA adapter is synced")

    def _iter_hf_adapter_units(self, adapter, *, materialize: bool) -> Iterator[list[tuple[str, torch.Tensor]]]:
        halves: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for name, param in self.model.state_dict().items():
            if ".lora_A" not in name and ".lora_B" not in name:
                continue
            full = gather_full_param(param)
            hf_name = to_hf_name(name)
            module = halves.setdefault(hf_name.rsplit(".lora_", 1)[0], [])
            module.append((hf_name, full))
            if len(module) == 2:
                del halves[hf_name.rsplit(".lora_", 1)[0]]
                if materialize:
                    yield module
        assert not halves, f"LoRA modules missing an A or B half: {sorted(halves)}"


class DiffusionLoraFromTensor(WeightTransferProtocol):
    """Each rank sends to the one engine whose GPUs it shares.

    sglang.multimodal_gen serves no pause/flush/begin/end session and reports no weight version, so the session
    frame is off. A failed bucket marks its cell errored, which without fault tolerance stops the run instead of
    serving a partly merged adapter.
    """

    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True
    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.group_name = "miles-diffusion-colocate"
        self._gather_groups: list[dist.ProcessGroup] = []

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        engine_cell_ids: Sequence[str],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.cell_updaters_of_cell_id = create_rollout_cell_updaters(self.args, rollout_engines, engine_cell_ids)
        for group in self._gather_groups:
            dist.destroy_process_group(group)
        self._gather_groups = []
        self._cell_id = self._gather_group = self._gather_src = None
        rank = dist.get_rank()
        # dist.new_group is collective: every rank creates every engine's group, in the same order.
        for cell_id, offset, count in zip(engine_cell_ids, engine_gpu_offsets, engine_gpu_counts, strict=True):
            group = dist.new_group(ranks=list(range(offset, offset + count)), backend="gloo")
            self._gather_groups.append(group)
            if offset <= rank < offset + count:
                self._cell_id, self._gather_group, self._gather_src = cell_id, group, offset
        self.is_sender = self._gather_group is not None

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
        for name, tensor in bucket:
            by_dtype.setdefault(tensor.dtype, []).append((name.split(":", 1)[-1], tensor))
        flats = [FlattenedTensorBucket(named_tensors=tensors) for tensors in by_dtype.values()]
        serialized = [
            MultiprocessingSerializer.serialize(
                {DIT_COMPONENT: {"flattened_tensor": flat.get_flattened_tensor(), "metadata": flat.get_metadata()}},
                output_str=True,
            )
            for flat in flats
        ]

        is_src = dist.get_rank() == self._gather_src
        gathered = [None] * dist.get_world_size(self._gather_group) if is_src else None
        dist.gather_object(
            obj=(get_physical_gpu_id(), serialized),
            object_gather_list=gathered,
            dst=self._gather_src,
            group=self._gather_group,
        )
        if is_src:
            updater = self.cell_updaters_of_cell_id[self._cell_id]
            async_utils.wait_futures(
                [
                    updater.submit_client_call(
                        "update_lora_weights_from_tensor",
                        serialized_named_tensors=[tensors[index] for _, tensors in gathered],
                        payload_gpu_uuids=[gpu_uuid for gpu_uuid, _ in gathered],
                        target_module=DIT_COMPONENT,
                        lora_rank=self.args.diffusion_lora_rank,
                        lora_alpha=self.args.diffusion_lora_alpha,
                    )
                    for index in range(len(serialized))
                ]
            )
        # Every rank owns storage behind the gathered IPC handles; keep it alive until the engine has read it.
        dist.barrier(group=self._gather_group)
        del flats

    def after_engines_resumed(self) -> None:
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
