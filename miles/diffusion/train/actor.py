"""FSDP2 trainer of `--model-kind diffusion` runs: Flow-GRPO on a diffusers DiT with a LoRA adapter."""

import logging

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.tensor import DTensor

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.lr_scheduler import get_lr_scheduler
from miles.backends.fsdp_utils.parallel import create_fsdp_parallel_state
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.diffusion.metrics import log_metrics
from miles.diffusion.train.loss import collate_pairs, flow_grpo_loss
from miles.diffusion.train.metric_buffer import MetricBuffer
from miles.diffusion.train.model import DTYPES, load_fsdp_dit
from miles.diffusion.train.pipeline_config import get_train_pipeline_config
from miles.diffusion.train.schedule import tile_train_pairs
from miles.diffusion.train.weight_sync import DiffusionLoraFromTensor, DiffusionLoraWeightIterator
from miles.utils import object_store
from miles.utils.context_utils import with_defer
from miles.utils.object_store import StoreObjectRef
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils.tracking import init_tracking

logger = logging.getLogger(__name__)


class DiffusionFSDPTrainRayActor(TorchNativeTrainRayActor):
    backend_name = "fsdp"

    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args,
        role: str,
        *,
        ckpt_load_config=None,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info,
        indep_dp_store_addr: str | None,
    ) -> int | None:  # type: ignore[override]
        assert ckpt_load_config is None and recv_ckpt_src_rank is None and indep_dp_store_addr is None
        assert not with_ref and not with_opd_teacher, "the diffusion trainer's KL reference is its LoRA base"
        super()._init_common(args, role, with_ref, with_opd_teacher=with_opd_teacher)
        set_parallel_state(create_fsdp_parallel_state(args))
        if args.backend.deterministic_mode:
            _enable_deterministic_training()
        torch.manual_seed(args.backend.seed)
        if self.args.debug_rollout_only:
            return 0
        if dist.get_rank() == 0:
            init_tracking(args, primary=False)
        if self.args.start_rollout_id is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)
        self.forward_dtype = DTYPES[args.diffusion_forward_dtype]
        self.pipeline_config = get_train_pipeline_config(args.diffusion_model_family)
        self.model = load_fsdp_dit(args, config=self.pipeline_config, mesh=get_parallel_state().get_mesh("fsdp"))
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=args.backend.lr,
            betas=(args.backend.adam_beta1, args.backend.adam_beta2),
            eps=args.backend.adam_eps,
            weight_decay=args.backend.weight_decay,
        )
        # fp16 policy gradients underflow unscaled; the sharded scaler agrees on found_inf across ranks.
        self.scaler = ShardedGradScaler(enabled=self.forward_dtype == torch.float16)
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.global_step = 0
        self.micro_step = 0
        checkpoint_payload = checkpoint.load(self)
        self.model_parts = [self.model]
        self.optimizers = [self.optimizer]
        self.weight_updater = self._build_diffusion_weight_updater()
        checkpoint.finalize_load(self, checkpoint_payload)

        if self.args.offload_train:
            self.sleep()
        self.prof.on_init_end()
        return int(self.args.start_rollout_id)

    def train(
        self,
        rollout_id: int,
        rollout_data_ref: StoreObjectRef | list[StoreObjectRef],
        witness_info=None,
        attempt: int = 0,
        external_data: TrainStepOutput | None = None,
    ) -> TrainStepOutput:
        assert witness_info is None and attempt == 0 and external_data is None
        self._heartbeat.bump()
        if self.args.offload_train:
            self.wake_up()
        with inverse_timer("train_wait"), timer("train"):
            with object_store.get_instance().get(rollout_data_ref[get_parallel_state().intra_dp.rank]) as shard:
                if not self.args.debug_rollout_only:
                    self._train_core(rollout_id, shard["diffusion_train_pairs"])
        self._heartbeat.bump()
        return TrainStepOutput(outcome=TrainStepOutcome.NORMAL)

    def _train_core(self, rollout_id: int, rows: list[list[dict]]) -> None:  # type: ignore[override]
        args = self.args
        num_steps = args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
        steps = tile_train_pairs(
            rows,
            num_optim_steps=num_steps,
            sample_microbatch=args.diffusion_micro_batch_size_sample,
            tstep_microbatch=args.diffusion_micro_batch_size_tstep,
        )
        with timer("actor_train"):
            for micro_batches in steps:
                self._optimizer_step(rollout_id, micro_batches)
        # Gradients are dead until the next step's zero_grad; dropping them keeps them out of the offload.
        self.optimizer.zero_grad(set_to_none=True)
        self.prof.step(rollout_id=rollout_id)

    def _optimizer_step(self, rollout_id: int, micro_batches: list[list[dict]]) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        state = get_parallel_state()
        metrics = MetricBuffer(group=state.intra_dp.group, device=torch.device("cuda", torch.cuda.current_device()))
        num_pairs = sum(len(pairs) for pairs in micro_batches)
        for pairs in micro_batches:
            loss_sum = self._forward_loss(pairs, metrics=metrics)
            self.scaler.scale(loss_sum / float(num_pairs)).backward()

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.diffusion_clip_grad)
        if isinstance(grad_norm, DTensor):
            # clip returns a lazily reduced partial norm; the full tensor is the true global norm.
            grad_norm = grad_norm.full_tensor()
        metrics.emit_replicated("grad_norm", grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()
        self.global_step += 1

        reduced = {f"train/{key}": value for key, value in metrics.reduce().items()}
        if dist.get_rank() == 0:
            reduced |= {
                "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                "train/epoch": float(rollout_id),
                "rollout/step": rollout_id,
                "train/step": float(self.global_step),
            }
            logger.info(f"[train step {self.global_step}] rollout={rollout_id} {reduced}")
            log_metrics(self.args, reduced, step_key="train/step")

    def _forward_loss(self, pairs: list[dict], *, metrics: MetricBuffer) -> torch.Tensor:
        args = self.args
        use_cfg = args.diffusion_guidance_scale > 1.0
        batch = collate_pairs(
            pairs,
            config=self.pipeline_config,
            use_cfg=use_cfg,
            adv_clip_max=args.diffusion_adv_clip_max,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        new_pred = self._noise_pred(batch)
        ref_pred = None
        if args.diffusion_kl_beta > 0:
            with torch.no_grad(), self.model.disable_adapter():
                ref_pred = self._noise_pred(batch)
                # PEFT re-enables adapter grads on exit; reshard first so that lands on the shards.
                _reshard(self.model)
        return flow_grpo_loss(
            batch,
            new_pred=new_pred,
            ref_pred=ref_pred,
            noise_level=args.diffusion_noise_level,
            clip_range=args.diffusion_clip_range,
            kl_beta=args.diffusion_kl_beta,
            metrics=metrics,
        )

    def _noise_pred(self, batch) -> torch.Tensor:
        with torch.autocast("cuda", dtype=self.forward_dtype, enabled=self.forward_dtype != torch.float32):
            return self.pipeline_config.compute_noise_pred(
                model=self.model,
                latents=batch.latents,
                timesteps=batch.timesteps,
                pos_cond=batch.pos_cond,
                neg_cond=batch.neg_cond,
                guidance_scale=self.args.diffusion_guidance_scale,
            )

    def _build_diffusion_weight_updater(self) -> WeightUpdater:
        self._published_weight_version = 0
        return WeightUpdater(
            self.args,
            self.model,
            weights_getter=lambda: None,
            model_name=self.pipeline_config.model_family,
            quantization_config=None,
            iterator_factory=DiffusionLoraWeightIterator.build,
            parallel_state=get_parallel_state(),
            is_lora=True,
            lora_sync_config=dict(r=self.args.diffusion_lora_rank, lora_alpha=self.args.diffusion_lora_alpha),
            protocol=DiffusionLoraFromTensor(self.args),
        )

    def _save_checkpoint(self, rollout_id: int) -> None:
        checkpoint.save(self, iteration=rollout_id)


def _enable_deterministic_training() -> None:
    """Torch-runtime half of deterministic mode; NCCL and cuBLAS read theirs from the spawn environment."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=False is required: SDPA's deterministic backward is gated on !warnOnly.
    torch.use_deterministic_algorithms(True, warn_only=False)


def _reshard(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
