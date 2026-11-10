"""Wan VACE reference-to-video as the keyframe recipe serves it: the planner's images as reference images.

``reference_model.py`` copies this file into a served model directory as ``pipeline.py``, and sglang's diffusers
backend loads it there as a diffusers custom pipeline, so it imports nothing from Miles. A request names its reference
images by path in ``reference_image_paths``, one of its ``diffusers_kwargs``, since a request carries JSON, not
images. ``lora_paths`` in ``model_index.json`` lists ``[path, component]`` pairs fused once at load.
"""

import torch
from diffusers import WanVACEPipeline
from PIL import Image


class WanVACEReferencePipeline(WanVACEPipeline):
    # diffusers matches model_index.json's keys against these parameters, so they are spelled out.
    def __init__(
        self,
        tokenizer,
        text_encoder,
        vae,
        scheduler,
        transformer=None,
        transformer_2=None,
        boundary_ratio=None,
        lora_paths=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
        )
        self.register_to_config(lora_paths=lora_paths)
        # Wan's VAE decodes in fp32, as LIN-2830 measured it; the backend loads every component in bf16.
        self.vae.to(torch.float32)
        for index, (path, component) in enumerate(lora_paths or []):
            name = f"lora{index}"
            self.load_lora_weights(path, adapter_name=name, load_into_transformer_2=component == "transformer_2")
            self.fuse_lora(components=[component], adapter_names=[name], lora_scale=1.0)
        if lora_paths:
            self.unload_lora_weights()

    @torch.no_grad()
    def __call__(self, *args, reference_image_paths=None, **kwargs):
        if reference_image_paths:
            kwargs["reference_images"] = [Image.open(path).convert("RGB") for path in reference_image_paths]
        if self.config.boundary_ratio is None:
            # A one-expert model refuses a second guidance scale, which the backend may pass along.
            kwargs.pop("guidance_scale_2", None)
        return super().__call__(*args, **kwargs)
