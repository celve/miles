"""Colocated LoRA weight sync to sglang.multimodal_gen: each training rank all-gathers the adapter's A/B pairs,
hands its engine CUDA-IPC handles, and the engine merges ``W + alpha/r * B @ A`` (``weight_update_mode=lora_merge``)."""

import logging
import re
from collections.abc import Iterator

import torch
import torch.distributed as dist
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.utils import get_physical_gpu_id
from miles.utils import async_utils

logger = logging.getLogger(__name__)

# PEFT appends the adapter name and ".weight" after lora_A / lora_B.
_PEFT_LORA_SUFFIX = re.compile(r"\.(lora_[AB])(?:\.[^.]+)?(?:\.weight)?$")


def to_hf_name(train_name: str) -> str:
    """base_model.model.blocks.0.attn.to_q.lora_A.default.weight -> blocks.0.attn.to_q.lora_A"""
    name = train_name.removeprefix("base_model.model.").replace(".base_layer.", ".")
    return _PEFT_LORA_SUFFIX.sub(r".\1", name)


class DiffusionLoraIpcSync:
    def __init__(self, args, model: torch.nn.Module, *, component: str = "transformer") -> None:
        self.args = args
        self.model = model
        self.component = component
        adapter = model.peft_config[model.active_adapter]
        self.lora_alpha = adapter.lora_alpha
        self.lora_rank = adapter.r
        self._connected_cell_ids: tuple[str, ...] | None = None

    def connect_if_needed(self, engines: UpdatableEngines) -> None:
        if self._connected_cell_ids == tuple(engines.engine_cell_ids):
            return
        # Every rank creates every engine's gather group, in the same order, as dist.new_group requires.
        rank = dist.get_rank()
        for client, offset, count in zip(
            engines.rollout_engines, engines.engine_gpu_offsets, engines.engine_gpu_counts, strict=True
        ):
            group = dist.new_group(ranks=list(range(offset, offset + count)), backend="gloo")
            if offset <= rank < offset + count:
                self._engine, self._gather_group, self._gather_src = client, group, offset
        self._connected_cell_ids = tuple(engines.engine_cell_ids)

    @torch.no_grad()
    def update_weights(self, *, weight_version: int) -> None:
        monkey_patch_torch_reductions()
        for bucket in self._iter_buckets():
            self._send_bucket(bucket, weight_version=weight_version)

    def _iter_buckets(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """Adapter tensors in buckets of at most --update-weight-buffer-size bytes; a module's A and B share one."""
        bucket, bucket_bytes, halves = [], 0, {}
        for name, param in self.model.state_dict().items():
            if ".lora_A" not in name and ".lora_B" not in name:
                continue
            full = gather_full_param(param)
            hf_name = to_hf_name(name)
            module = hf_name.rsplit(".lora_", 1)[0]
            halves.setdefault(module, []).append((hf_name, full))
            if len(halves[module]) < 2:
                continue
            unit = halves.pop(module)
            bucket.extend(unit)
            bucket_bytes += sum(tensor.numel() * tensor.element_size() for _, tensor in unit)
            if bucket_bytes >= self.args.update_weight_buffer_size:
                yield bucket
                bucket, bucket_bytes = [], 0
        assert not halves, f"LoRA modules missing an A or B half: {sorted(halves)}"
        if bucket:
            yield bucket

    def _send_bucket(self, named_tensors: list[tuple[str, torch.Tensor]], *, weight_version: int) -> None:
        by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
        for name, tensor in named_tensors:
            by_dtype.setdefault(tensor.dtype, []).append((name, tensor))
        serialized = []
        for tensors in by_dtype.values():
            flat = FlattenedTensorBucket(named_tensors=tensors)
            payload = {
                self.component: {"flattened_tensor": flat.get_flattened_tensor(), "metadata": flat.get_metadata()}
            }
            serialized.append(MultiprocessingSerializer.serialize(payload, output_str=True))

        is_src = dist.get_rank() == self._gather_src
        gathered = [None] * dist.get_world_size(self._gather_group) if is_src else None
        dist.gather_object(
            obj=(get_physical_gpu_id(), serialized),
            object_gather_list=gathered,
            dst=self._gather_src,
            group=self._gather_group,
        )
        if not is_src:
            return
        for dtype_index in range(len(serialized)):
            body = {
                "serialized_named_tensors": [tensors[dtype_index] for _, tensors in gathered],
                "payload_gpu_uuids": [gpu_uuid for gpu_uuid, _ in gathered],
                "load_format": "flattened_bucket",
                "target_modules": [self.component],
                "weight_version": str(weight_version),
                "weight_update_mode": "lora_merge",
                "lora_alpha": self.lora_alpha,
                "lora_rank": self.lora_rank,
            }
            result = async_utils.run(
                self._engine._make_request(
                    "update_weights_from_tensor", body, timeout=self.args.update_weight_engine_request_timeout
                )
            )
            assert result and result.get("success"), f"LoRA weight sync failed on {self._engine.server_url}: {result}"
