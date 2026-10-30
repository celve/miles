"""Diffusion post-training (`--model-kind diffusion`): Flow-GRPO on diffusers DiTs served by sglang.multimodal_gen.

CUDA only: rollout weight sync uses CUDA IPC, and startup validation rejects other configurations.
"""
