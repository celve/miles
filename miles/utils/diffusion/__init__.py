"""Diffusion post-training (`--model-kind diffusion`): Flow-GRPO on diffusers DiTs served by sglang.multimodal_gen.

CUDA only: rollout weight sync uses CUDA IPC, and startup validation rejects other configurations.

The code lives beside the Miles code of each layer:

- `miles.utils.diffusion`: the `Sample.diffusion` payload (`types`), startup validation (`arguments`) and the
  metrics sink (`metrics`), shared by every layer below.
- `miles.backends.sglang_utils.sglang_diffusion_*`: the sglang.multimodal_gen engine's launch command, entry point
  and API client.
- `miles.rollout.diffusion_rollout`: the rollout function, prompt data source, response parser, SDE step strategy
  and the converter that turns samples into train pairs.
- `miles.rollout.rm_hub.ocr`: the OCR reward, selected by `--rm-type ocr`.
- `miles.backends.fsdp_utils.diffusion`: the DiT trainer (FSDP2 model, SDE log-prob, losses, weight sync); only
  that package imports it at module level.
"""
