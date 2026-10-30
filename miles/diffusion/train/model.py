"""Load a diffusers DiT component for training: rank 0 reads the checkpoint, the others build on meta and
receive their FSDP2 shards by broadcast; LoRA (PEFT) wraps the model before sharding."""

import importlib
import logging
import warnings
from contextlib import contextmanager
from itertools import chain

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from miles.diffusion.train.pipeline_config import TrainPipelineConfig

logger = logging.getLogger(__name__)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load_fsdp_dit(
    args,
    *,
    config: TrainPipelineConfig,
    mesh,
    component: str = "transformer",
    master_dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    is_source = dist.get_rank() == 0
    with _init_context(materialize=is_source):
        model = _load_component(args.hf_checkpoint, component, master_dtype=master_dtype, materialize=is_source)
    if args.backend.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model = _apply_lora(model, args, materialize=is_source)
    model.train(True)

    _sync_dtypes_from_rank0(model)
    full_state = model.state_dict() if is_source else {}
    model = _apply_fsdp2(model, mesh=mesh, param_dtype=DTYPES[args.diffusion_forward_dtype])
    _broadcast_full_state(model, full_state)
    del full_state
    # FSDP fixes each group's gradient dtype at lazy init from the params requiring grad; run it now,
    # before a first forward with the adapter disabled would see none.
    model._get_fsdp_state()._lazy_init()
    return model


@contextmanager
def _init_context(*, materialize: bool):
    if materialize:
        with torch.device("cpu"):
            yield
        return
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"for .*: copying from a non-meta parameter in the checkpoint.*")
        yield


def _load_component(checkpoint: str, component: str, *, master_dtype: torch.dtype, materialize: bool):
    """Load one pipeline component by the class `model_index.json` names, without loading its siblings."""
    library, class_name = DiffusionPipeline.load_config(checkpoint)[component][:2]
    model_cls = getattr(importlib.import_module(library), class_name)
    # Off rank 0, low_cpu_mem_usage=False keeps params on meta, which diffusers forbids for classes pinning
    # modules to fp32; the pin is lifted meanwhile and dtypes are synced from rank 0 afterwards.
    keep_in_fp32 = getattr(model_cls, "_keep_in_fp32_modules", None)
    if not materialize and keep_in_fp32 is not None:
        model_cls._keep_in_fp32_modules = None
    try:
        return model_cls.from_pretrained(
            checkpoint, subfolder=component, torch_dtype=master_dtype, low_cpu_mem_usage=materialize
        )
    finally:
        if not materialize and keep_in_fp32 is not None:
            model_cls._keep_in_fp32_modules = keep_in_fp32


def _apply_lora(model: torch.nn.Module, args, *, materialize: bool) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    init = True if args.diffusion_lora_init == "kaiming-uniform" else args.diffusion_lora_init
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.diffusion_lora_rank,
            lora_alpha=args.diffusion_lora_alpha,
            target_modules=args.diffusion_lora_target_modules,
            # Off rank 0 the adapters stay uninitialized on meta; rank 0's values arrive by broadcast.
            init_lora_weights=init if materialize else False,
        ),
        low_cpu_mem_usage=not materialize,
    )
    if materialize:
        model.print_trainable_parameters()
    return model


def _sync_dtypes_from_rank0(model: torch.nn.Module) -> None:
    tensors = list(chain(model.parameters(), model.buffers()))
    objects = [[tensor.dtype for tensor in tensors] if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    if dist.get_rank() != 0:
        for tensor, dtype in zip(tensors, objects[0], strict=True):
            tensor.data = tensor.data.to(dtype)


def _apply_fsdp2(model: torch.nn.Module, *, mesh, param_dtype: torch.dtype) -> torch.nn.Module:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    block_classes = set(base._no_split_modules)
    blocks = [module for module in model.modules() if type(module).__name__ in block_classes]
    # Inputs keep their dtype: autocast owns compute, so gradient-checkpoint recompute stays consistent.
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, cast_forward_inputs=False)
    logger.info(f"FSDP: wrapping {len(blocks)} blocks of {sorted(block_classes)}, param_dtype={param_dtype}")
    for block in blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    return model


def _broadcast_full_state(model: torch.nn.Module, full_state: dict) -> None:
    device = torch.cuda.current_device()
    if dist.get_rank() == 0:
        model.to(device=device, non_blocking=True)
    else:
        model.to_empty(device=device)
    set_model_state_dict(
        model, full_state, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    )
    # State dict loading skips non-persistent buffers.
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)
