"""Model-family adapters for the DiT training forward: conditioning kwargs, collation and CFG."""

import abc
import types

import torch
import torch.nn.functional as F

from miles.utils.diffusion.types import CondKwargs

_REGISTRY: dict[str, type["TrainPipelineConfig"]] = {}


def register_train_pipeline_config(family: str):
    def wrapper(cls):
        cls.model_family = family
        _REGISTRY[family] = cls
        return cls

    return wrapper


def resolve_model_family(hf_checkpoint: str) -> str:
    """The family whose longest name pattern occurs in the checkpoint path."""
    ref = hf_checkpoint.lower()
    matches = [
        (len(pattern), family) for family, cls in _REGISTRY.items() for pattern in cls.name_patterns if pattern in ref
    ]
    if not matches:
        raise ValueError(f"Cannot resolve the diffusion family of {hf_checkpoint!r}; set --diffusion-model-family")
    return max(matches)[1]


def get_train_pipeline_config(family: str) -> "TrainPipelineConfig":
    if family not in _REGISTRY:
        raise ValueError(f"No train pipeline config for family {family!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[family]()


class TrainPipelineConfig(abc.ABC):
    model_family: str
    name_patterns: tuple[str, ...] = ()
    lora_target_modules: list[str] = []
    num_train_timesteps: int = 1000
    # The pipeline's denoiser components the trainer loads, trains and syncs, in this order.
    components: tuple[str, ...] = ("transformer",)
    # fnmatch patterns of module names whose weights the forward reads in fp32, as the engine computes them.
    fp32_module_patterns: tuple[str, ...] = ()
    # The component the engine guides with --diffusion-guidance-scale-2 instead of --diffusion-guidance-scale.
    guidance_scale_2_component: str | None = None

    def postprocess_model_after_materialize(self, model: torch.nn.Module) -> None:  # noqa: B027 — optional hook
        """Adjust the sharded, loaded model before its first forward; default none."""

    def guidance_scale(self, component: str, args) -> float:
        """The CFG scale the engine denoises this component's steps with."""
        if component == self.guidance_scale_2_component:
            return args.diffusion_guidance_scale_2
        return args.diffusion_guidance_scale

    def component_for_timestep(self, timestep: float) -> str:
        """The component that denoises at this scheduler timestep."""
        return self.components[0]

    def component_for_timesteps(self, timesteps: torch.Tensor) -> str:
        """The one component a micro-batch's forward runs; its pairs must not straddle components."""
        if len(self.components) == 1:
            return self.components[0]
        components = {self.component_for_timestep(t) for t in timesteps.tolist()}
        if len(components) > 1:
            raise ValueError(
                f"A micro-batch mixes the timesteps of components {sorted(components)}; draw one SDE step per "
                "rollout or set --diffusion-micro-batch-size-tstep 1"
            )
        return components.pop()

    def compute_noise_pred(
        self,
        *,
        model: torch.nn.Module,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        pos_cond: dict,
        neg_cond: dict | None,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Diffusers forward; with CFG, separate positive and negative forwards as the engine serves them."""

        def forward(cond: dict) -> torch.Tensor:
            return model(hidden_states=latents, timestep=timesteps, return_dict=False, **cond)[0]

        if neg_cond is None:
            return forward(pos_cond)
        return self.cfg_combine(forward(pos_cond), forward(neg_cond), guidance_scale)

    def process_sigma_as_timesteps_input(self, sigmas: torch.Tensor) -> torch.Tensor:
        """A sigma in [0, 1] as this family's DiT takes its timestep input."""
        return sigmas * float(self.num_train_timesteps)

    @abc.abstractmethod
    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        """One sample's conditioning as model forward kwargs."""

    @abc.abstractmethod
    def collate_cond(self, per_sample: list[dict], device: torch.device) -> dict:
        """Stack per-sample forward kwargs into one batch."""

    @abc.abstractmethod
    def cfg_combine(self, pred_pos: torch.Tensor, pred_neg: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        """Classifier-free guidance as this family's engine applies it."""


@register_train_pipeline_config("sd3")
class SD3TrainPipelineConfig(TrainPipelineConfig):
    """diffusers SD3Transformer2DModel."""

    name_patterns = ("stable-diffusion-3", "sd3")
    lora_target_modules = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
    ]

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None:
            return {}
        kwargs = {}
        if cond.encoder_hidden_states:
            encoder_hidden_states = torch.cat(cond.encoder_hidden_states).to(device)
            kwargs["encoder_hidden_states"] = (
                encoder_hidden_states.unsqueeze(0) if encoder_hidden_states.ndim == 2 else encoder_hidden_states
            )
        if cond.pooled_projections:
            pooled_projections = torch.cat(cond.pooled_projections).to(device)
            kwargs["pooled_projections"] = (
                pooled_projections.unsqueeze(0) if pooled_projections.ndim == 1 else pooled_projections
            )
        return kwargs

    def collate_cond(self, per_sample: list[dict], device: torch.device) -> dict:
        return {key: torch.cat([kwargs[key] for kwargs in per_sample], dim=0).to(device) for key in per_sample[0]}

    def cfg_combine(self, pred_pos: torch.Tensor, pred_neg: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        return pred_neg + guidance_scale * (pred_pos - pred_neg)


@register_train_pipeline_config("wan2_2")
class Wan22TrainPipelineConfig(TrainPipelineConfig):
    """diffusers WanTransformer3DModel, Wan2.2-T2V-A14B's two experts: `transformer` denoises high-noise timesteps
    (t >= 0.875 x 1000) with --diffusion-guidance-scale, `transformer_2` the rest with --diffusion-guidance-scale-2."""

    name_patterns = ("wan2.2", "wan-2.2")
    lora_target_modules = [
        *("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"),
        *("attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"),
        *("ffn.net.0.proj", "ffn.net.2"),
    ]
    components = ("transformer", "transformer_2")
    guidance_scale_2_component = "transformer_2"
    # The cross-attention LayerNorm's affine weights; the engine computes it from fp32 weights.
    fp32_module_patterns = ("*.norm2",)
    boundary_ratio = 0.875

    def component_for_timestep(self, timestep: float) -> str:
        return "transformer" if timestep >= self.boundary_ratio * self.num_train_timesteps else "transformer_2"

    def postprocess_model_after_materialize(self, model: torch.nn.Module) -> None:
        """Patchify with a GEMM, which matches the engine's fp32-accumulating F.linear where Conv3d does not."""
        for name, module in model.named_modules():
            if name.endswith("patch_embedding") and isinstance(module, torch.nn.Conv3d):
                module.forward = types.MethodType(_gemm_patchify_forward, module)

    def prepare_cond_kwargs(self, cond: CondKwargs | None, device: torch.device) -> dict:
        if cond is None or not cond.encoder_hidden_states:
            return {}
        encoder_hidden_states = torch.cat(cond.encoder_hidden_states).to(device)
        return {
            "encoder_hidden_states": (
                encoder_hidden_states.unsqueeze(0) if encoder_hidden_states.ndim == 2 else encoder_hidden_states
            )
        }

    def collate_cond(self, per_sample: list[dict], device: torch.device) -> dict:
        # UMT5 embeddings have a fixed length, so they concatenate without padding.
        return {
            "encoder_hidden_states": torch.cat([kwargs["encoder_hidden_states"] for kwargs in per_sample]).to(device)
        }

    def cfg_combine(self, pred_pos: torch.Tensor, pred_neg: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        return pred_neg + guidance_scale * (pred_pos - pred_neg)


def _gemm_patchify_forward(self: torch.nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    """Conv3d as a GEMM when the kernel equals the stride and tiles the input; otherwise the Conv3d itself."""
    pt, ph, pw = self.kernel_size
    batch, channels, t, h, w = x.shape
    if tuple(self.stride) != (pt, ph, pw) or t % pt or h % ph or w % pw:
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    t_, h_, w_ = t // pt, h // ph, w // pw
    x = x.reshape(batch, channels, t_, pt, h_, ph, w_, pw).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    x = F.linear(
        x.reshape(batch, t_ * h_ * w_, channels * pt * ph * pw),
        self.weight.reshape(self.weight.shape[0], -1),
        self.bias,
    )
    return x.reshape(batch, t_, h_, w_, -1).permute(0, 4, 1, 2, 3).contiguous()
