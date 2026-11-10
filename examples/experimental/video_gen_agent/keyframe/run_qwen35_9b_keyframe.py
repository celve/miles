"""Keyframe planner RL: a Qwen3.5-9B (or Qwen3.8-27B) planner makes up to three keyframe images with a frozen
Z-Image-Turbo, sees each, then makes one video from the image it picks with a frozen Wan2.2-A14B and the lightx2v
4-step LoRAs.
The video is scored against the request by phase 1's gated reward (``video_gen_agent.gated_reward``): by default
HPSv3 on its frames plus VQAScore on the request, each standardized within the group and summed, zeroed when
the Qwen3-VL judge's motion axis says the video barely moves and halved when OCR finds text the request did not ask
for. Its reward pools share the rollout GPUs, one pool per GPU from the planner's on; OCR runs on the CPU.

One node, not colocated (frozen generators cannot share GPUs with training): the trainer (Qwen3.5 plugin model
under raw Megatron, vision tower frozen), the planner's engine with the motion judge's reward pool beside it, the image
engines, the video engines, and on 8 GPUs an eval engine that scores exported snapshots off the training path; on
4 GPUs eval runs on the planner's engine. Every engine has one GPU.
The env is ``examples.experimental.video_gen_agent.keyframe.env`` under session server v2; each image comes back to
the planner as an ``image_url`` part in token space.

Args:
    --planner: ``qwen3.5-9b`` (default) or ``qwen3.8-27b``, the planner's checkpoint, TITO family and layouts.
    --num-gpus-per-node: picks the planner's layout of trainer, planner, image, video and eval GPUs (9B: 4 is
        1/1/1/1/0, 8 is 1/1/2/3/1, or 1/1/1/4/1 in reference mode; 27B: 8 is 4/1/1/2/0, its trainer at TP4); --train-gpus, --policy-gpus, --image-gpus,
        --video-gpus, --eval-gpus override it. The trainer's GPUs are its tensor-parallel group.
    --optimizer-cpu-offload: keep Adam's state in host memory (default: the planner's recipe, on for the 27B).
    --video-mode: ``preview`` (default) makes the video from text, the chosen image's prompt plus the motion prompt,
        with Wan2.2-T2V-A14B; ``i2v`` makes it from the image with Wan2.2-I2V-A14B, whose clips LIN-2809 measured
        moving far less; ``reference`` makes it with Wan2.1-VACE-14B from the planner's asset images (one per subject,
        isolated, and one of the setting) as reference images, which LIN-2830 measured holding the image at no cost in reward, at twice T2V's GPU time.
        Each takes its own lightx2v LoRAs. Reference mode serves VACE through sglang's diffusers backend from a
        directory ``prepare`` builds beside the checkpoint (``keyframe/reference_model.py``).
    --reference-scale: how strongly a reference-mode clip holds its images, VACE's conditioning scale (default 0.5,
        LIN-2830's pick; 1.0 copies the image as a near-still frame).
    --generation-timeout: seconds after which a generation counts as an outage, which drops its group (default: the
        env's 600; in reference mode, long enough for the queue on a VACE engine, which renders one clip at a time).
    --image-model, --image-steps, --image-guidance-scale, --video-model: the generators (default: Z-Image-Turbo at
        9 steps without CFG; Qwen/Qwen-Image-2.1-Turbo gets 8 steps at CFG 1; the video model follows --video-mode).
    --num-rollout, --rollout-batch-size, --n-samples-per-prompt: GRPO steps, requests per step, episodes per request.
    --max-images, --max-turns, --seconds: the episode's budgets and the video's length (images: 4 in reference mode,
        one asset per subject plus the setting; 3 otherwise).
    --max-prompt-words: a longer generator prompt gets an error reply (default 0, no cap: Linyu, 2026-11-10, "more
        detailed prompt might generate better video"); 120 is the survey's cap.
    --reward: the signal terms, ``hpsv3+vqascore`` (default), ``hpsv3+unifiedreward``, ``hpsv3`` or ``pickscore``;
        --adherence-gate adds the judge's prompt-adherence gate, off by default.
    --requests: the training requests, ``vbench`` (default; VBench's prompts, 10% held out by hash) or
        ``t2v-compbench`` (T2V-CompBench's 1,400 compositional prompts, --compbench-heldout of them held out across its
        seven categories, LIN-2834). With ``t2v-compbench`` eval runs both held-out sets: ``compbench_heldout``, logged
        per category too, and VBench's as ``heldout``, so ``eval/heldout`` stays comparable with VBench runs.
    --model-dir, --data-dir: where models and the requests live; checkpoints, images and clips go under --output-dir.

Example:
    python examples/experimental/video_gen_agent/keyframe/run_qwen35_9b_keyframe.py --num-gpus-per-node 4
    python examples/experimental/video_gen_agent/keyframe/run_qwen35_9b_keyframe.py --num-rollout 2 --rollout-batch-size 4
    python examples/experimental/video_gen_agent/keyframe/run_qwen35_9b_keyframe.py --image-model Qwen/Qwen-Image-2.1-Turbo
    python examples/experimental/video_gen_agent/keyframe/run_qwen35_9b_keyframe.py --planner qwen3.8-27b
"""

import math
from dataclasses import dataclass, field, replace

import typer
import yaml
from examples.experimental.video_gen_agent.multiturn.run_qwen35_9b_fastwan import AGENT, JUDGE, VBENCH
from scripts.run_sd35_medium_flow_grpo_ocr import OCR_DEPS_INSTALL

from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser
from miles.utils.external_utils import command_utils

LIGHTX2V = "lightx2v/Wan2.2-Distill-Loras"
LIGHTX2V_WAN21 = "lightx2v/Wan2.1-Distill-Loras"
FTFY = "ftfy==6.3.1"
# Twice the GPU-seconds a served VACE-14B clip took at 832x480, 33 frames, 4 steps on H200 (LIN-2830).
_REFERENCE_SECONDS_PER_CLIP = 20.0
# T2V-CompBench's repository has no licence: prepare fetches its prompts at this commit and nothing redistributes them.
COMPBENCH = "https://github.com/KaiyueSun98/T2V-CompBench.git"
COMPBENCH_COMMIT = "dd5eff7b93af0550b9efa2bdabbb21b3b017ceda"
# Lines in VBench's prompts/all_dimension.txt, which bound its held-out share in the reference-mode timeout.
_VBENCH_REQUESTS = 946
HPSV3 = "MizzenAI/HPSv3"
HPSV3_BASE = "Qwen/Qwen2-VL-7B-Instruct"
UNIFIED_REWARD = "CodeGoat24/UnifiedReward-2.0-qwen3vl-8b"
PICKSCORE = "yuvalkirstain/PickScore_v1"
PICKSCORE_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


@dataclass(frozen=True)
class _PlannerRecipe:
    repo: str
    megatron_model_type: str
    tito_model: str
    # Trainer, planner, image, video and eval GPUs, by GPUs per node.
    layouts: dict[int, tuple[int, int, int, int, int]]
    optimizer_cpu_offload: bool = False
    # Layouts that replace those in reference mode, whose video model is the bottleneck (LIN-2830).
    reference_layouts: dict[int, tuple[int, int, int, int, int]] = field(default_factory=dict)


_PLANNERS = {
    "qwen3.5-9b": _PlannerRecipe(
        repo="Qwen/Qwen3.5-9B",
        megatron_model_type="qwen3.5-9B",
        tito_model="qwen35",
        layouts={4: (1, 1, 1, 1, 0), 8: (1, 1, 2, 3, 1)},
        reference_layouts={8: (1, 1, 1, 4, 1)},
    ),
    # LIN-2699 trained Qwen3.8-27B at TP4; with four trainer GPUs, eval shares the planner's engine.
    "qwen3.8-27b": _PlannerRecipe(
        repo="Qwen/Qwen3.8-27B",
        megatron_model_type="qwen3.8-27B",
        tito_model="qwen38small",
        layouts={8: (4, 1, 1, 2, 0)},
        optimizer_cpu_offload=True,
    ),
}


# Denoising steps and CFG scale of each known image model at 832x480 (LIN-2809).
_IMAGE_RECIPES = {"Tongyi-MAI/Z-Image-Turbo": (9, 0.0), "Qwen/Qwen-Image-2.1-Turbo": (8, 1.0)}


@dataclass(frozen=True)
class _VideoRecipe:
    model: str
    lora_repo: str
    # Each LoRA file in lora_repo, with the expert it merges into.
    loras: tuple[tuple[str, str], ...]
    # Served by sglang's diffusers backend from a directory reference_model.py builds, which fuses the LoRAs at load
    # and runs its scheduler at this flow shift; None serves the checkpoint natively and merges them by set_lora.
    reference_flow_shift: float | None = None


_VIDEO_RECIPES = {
    "i2v": _VideoRecipe(
        model="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        lora_repo=LIGHTX2V,
        loras=(
            ("wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors", "transformer"),
            ("wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors", "transformer_2"),
        ),
    ),
    "preview": _VideoRecipe(
        model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        lora_repo=LIGHTX2V,
        loras=(
            ("wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors", "transformer"),
            ("wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors", "transformer_2"),
        ),
    ),
    # lightx2v's Wan2.1-T2V-14B 4-step LoRA on VACE-14B, at the flow shift LIN-2830 measured.
    "reference": _VideoRecipe(
        model="Wan-AI/Wan2.1-VACE-14B-diffusers",
        lora_repo=LIGHTX2V_WAN21,
        loras=(("wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors", "transformer"),),
        reference_flow_shift=5.0,
    ),
}


# Each signal's scores that map to 0 and 1: its lowest and highest score over LIN-2736's 300 calibration
# (request, clip) items, scored on one H200 (LIN-2821).
_SIGNAL_RANGES = {
    "hpsv3": (-10.213, 11.652),
    "vqascore": (0.0, 1.0),
    "unifiedreward": (0.248, 3.229),
    "pickscore": (0.604, 0.910),
}
# hpsv3+unifiedreward is the survey's pick; UnifiedReward-2.0 failed LIN-2736's overlay and mismatch pairs on its
# own (0.62, 0.29), so the survey's fallback, VQAScore on the request, takes its place by default (LIN-2821).
_REWARDS = {
    "hpsv3+vqascore": ("hpsv3", "vqascore"),
    "hpsv3+unifiedreward": ("hpsv3", "unifiedreward"),
    "hpsv3": ("hpsv3",),
    "pickscore": ("pickscore",),
}
# The adherence gate's threshold, set between the scores of LIN-2736's 24 mismatch pairs' two sides (LIN-2821).
_ADHERENCE_THRESHOLD = 0.016


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    planner: str = "qwen3.5-9b"
    num_gpus_per_node: int = 8
    train_gpus: int | None = None
    policy_gpus: int | None = None
    image_gpus: int | None = None
    video_gpus: int | None = None
    eval_gpus: int | None = None
    optimizer_cpu_offload: bool | None = None
    video_mode: str = "preview"
    image_model: str = "Tongyi-MAI/Z-Image-Turbo"
    image_steps: int | None = None
    image_guidance_scale: float | None = None
    video_model: str | None = None
    reference_scale: float | None = None
    generation_timeout: float | None = None
    num_rollout: int = 200
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    max_images: int | None = None
    max_turns: int = 6
    max_prompt_words: int = 0
    seconds: float = 2.0
    reward: str = "hpsv3+vqascore"
    adherence_gate: bool = False
    eval_interval: int = 10
    heldout_percent: int = 10
    requests: str = "vbench"
    compbench_heldout: int = 200
    feature_cache: bool = True
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    def __post_init__(self):
        assert self.num_nodes == 1, "the recipe places its trainer, engines and judge on one node"
        assert self.planner in _PLANNERS, f"--planner is one of {sorted(_PLANNERS)}"
        if self.optimizer_cpu_offload is None:
            self.optimizer_cpu_offload = self.recipe.optimizer_cpu_offload
        assert self.video_mode in _VIDEO_RECIPES, f"--video-mode is one of {sorted(_VIDEO_RECIPES)}"
        if self.max_images is None:
            # Reference mode's assets are one per subject plus the setting, so two subjects and a place need 3 and
            # three subjects need 4 (LIN-2830).
            self.max_images = 4 if self.video_mode == "reference" else 3
        if self.video_mode == "reference":
            self.reference_scale = 0.5 if self.reference_scale is None else self.reference_scale
            assert self.reference_scale > 0, f"--reference-scale must be positive, got {self.reference_scale}"
        else:
            assert self.reference_scale is None, "--reference-scale applies to --video-mode reference only"
        assert self.generation_timeout is None or self.generation_timeout > 0, "--generation-timeout must be positive"
        assert self.reward in _REWARDS, f"--reward is one of {sorted(_REWARDS)}"
        assert self.requests in ("vbench", "t2v-compbench"), "--requests is vbench or t2v-compbench"
        if self.image_steps is None or self.image_guidance_scale is None:
            recipe = _IMAGE_RECIPES.get(self.image_model)
            assert (
                recipe is not None
            ), f"{self.image_model} has no recipe, so set --image-steps and --image-guidance-scale"
            self.image_steps = recipe[0] if self.image_steps is None else self.image_steps
            self.image_guidance_scale = recipe[1] if self.image_guidance_scale is None else self.image_guidance_scale
        split = (self.train_gpus, self.policy_gpus, self.image_gpus, self.video_gpus, self.eval_gpus)
        if None in split:
            layouts = self.recipe.layouts | (self.recipe.reference_layouts if self.video_mode == "reference" else {})
            layout = layouts.get(self.num_gpus_per_node)
            assert (
                layout is not None
            ), f"{self.planner} has no layout for {self.num_gpus_per_node} GPUs, so set all five GPU counts"
            split = tuple(
                layout_gpus if gpus is None else gpus for gpus, layout_gpus in zip(split, layout, strict=True)
            )
            self.train_gpus, self.policy_gpus, self.image_gpus, self.video_gpus, self.eval_gpus = split
        assert (
            sum(split) == self.num_gpus_per_node
        ), f"the GPU split adds up to {sum(split)}, not {self.num_gpus_per_node}"

    @property
    def recipe(self) -> _PlannerRecipe:
        return _PLANNERS[self.planner]

    @property
    def video(self) -> _VideoRecipe:
        recipe = _VIDEO_RECIPES[self.video_mode]
        return replace(recipe, model=self.video_model) if self.video_model else recipe

    @property
    def video_path(self) -> str:
        """The directory the video engines load: the checkpoint, or in reference mode the directory built beside it."""
        local = self.local(self.video.model)
        return f"{local}-reference" if self.video.reference_flow_shift is not None else local

    def video_lora_paths(self) -> list[tuple[str, str]]:
        return [(f"{self.local(self.video.lora_repo)}/{name}", target) for name, target in self.video.loras]

    @property
    def requests_dir(self) -> str:
        """Where the training requests and their held-out set live."""
        return (
            self.vbench_requests_dir
            if self.requests == "vbench"
            else f"{self.data_dir}/video_gen_requests_t2v_compbench"
        )

    @property
    def vbench_requests_dir(self) -> str:
        return f"{self.data_dir}/video_gen_requests"

    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/{self.run_id}"

    def local(self, repo: str) -> str:
        return f"{self.model_dir}/{repo.split('/')[1]}"


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    for repo in (args.recipe.repo, JUDGE, args.image_model, args.video.model, *_reward_models(args)):
        U.exec_command_cpu(f"hf download {repo} --local-dir {args.local(repo)}")
    U.exec_command_cpu(OCR_DEPS_INSTALL)
    # Fetch PaddleOCR's models once; the OCR pool's workers would each download them and race on a cold cache.
    U.exec_command_cpu(
        f"PYTHONPATH={command_utils.repo_base_dir} python -c "
        "'from miles.diffusion.rollout.ocr import OcrScorer; OcrScorer()'"
    )
    video = args.video
    lora_files = " ".join(name for name, _ in video.loras)
    U.exec_command_cpu(f"hf download {video.lora_repo} {lora_files} --local-dir {args.local(video.lora_repo)}")
    if video.reference_flow_shift is not None:
        # diffusers' WanVACEPipeline cleans every prompt with ftfy, which the Miles image does not ship.
        U.exec_command_cpu(f"python -c 'import ftfy' 2>/dev/null || pip install {FTFY}")
        loras = " ".join(f"--lora {path}:{target}" for path, target in args.video_lora_paths())
        U.exec_command_cpu(
            f"cd {command_utils.repo_base_dir} && python -m {AGENT}.keyframe.reference_model "
            f"--src {args.local(video.model)} --dst {args.video_path} --flow-shift {video.reference_flow_shift} {loras}"
        )
    U.convert_checkpoint(
        model_name=args.recipe.repo.split("/")[1],
        megatron_model_type=args.recipe.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.local(args.recipe.repo),
        megatron_path=args.megatron_path,
    )
    U.exec_command_cpu(f"[ -d {args.data_dir}/VBench ] || git clone --depth 1 {VBENCH} {args.data_dir}/VBench")
    U.exec_command_cpu(
        f"cd {command_utils.repo_base_dir} && python -m {AGENT}.make_requests "
        f"{args.data_dir}/VBench/prompts/all_dimension.txt {args.vbench_requests_dir} "
        f"--heldout-percent {args.heldout_percent}"
    )
    if args.requests == "t2v-compbench":
        compbench = f"{args.data_dir}/T2V-CompBench"
        U.exec_command_cpu(
            f"[ -d {compbench} ] || git clone -q --filter=blob:none --no-checkout {COMPBENCH} {compbench}"
        )
        U.exec_command_cpu(f"git -C {compbench} checkout -q {COMPBENCH_COMMIT} -- prompts meta_data")
        U.exec_command_cpu(
            f"cd {command_utils.repo_base_dir} && python -m {AGENT}.make_compbench_requests {compbench} "
            f"{args.requests_dir} --heldout {args.compbench_heldout}"
        )


def execute(args: ScriptArgs):
    U = args.create_backend()
    U.execute_train(
        train_args=_train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.recipe.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path, **_agent_env(args)},
    )


def _generation_timeout(args: ScriptArgs) -> float | None:
    """The flag, or in reference mode a bound on the wait behind a full queue: VACE renders one clip at a time, so a
    step's episodes plus an eval's, spread over the video engines, at twice LIN-2830's 9.7 s a clip."""
    if args.generation_timeout is not None or args.video_mode != "reference":
        return args.generation_timeout
    episodes = args.rollout_batch_size * args.n_samples_per_prompt
    if args.requests == "t2v-compbench":
        # Eval runs both held-out sets, one episode a request; VBench's hash split holds about its share out.
        episodes += args.compbench_heldout + math.ceil(_VBENCH_REQUESTS * args.heldout_percent / 100)
    else:
        episodes *= 2
    queued = math.ceil(episodes / args.video_gpus)
    return max(600.0, queued * _REFERENCE_SECONDS_PER_CLIP)


def _agent_env(args: ScriptArgs) -> dict[str, str]:
    env = {
        "AGENT_VIDEO_GEN_GENERATOR": f"{AGENT}.multiturn.generator.RouterClipGenerator",
        "AGENT_VIDEO_GEN_IMAGE_GENERATOR": f"{AGENT}.keyframe.generator.RouterImageGenerator",
        "AGENT_VIDEO_GEN_CLIP_DIR": f"{args.run_dir}/clips",
        "AGENT_VIDEO_GEN_IMAGE_DIR": f"{args.run_dir}/images",
        "AGENT_VIDEO_GEN_SUBMITTED_DIR": f"{args.run_dir}/submitted",
        "AGENT_VIDEO_GEN_MAX_IMAGES": str(args.max_images),
        "AGENT_VIDEO_GEN_MAX_TURNS": str(args.max_turns),
        "AGENT_VIDEO_GEN_MAX_PROMPT_WORDS": str(args.max_prompt_words),
        "AGENT_VIDEO_GEN_SECONDS": str(args.seconds),
        "AGENT_VIDEO_GEN_WIDTH": "832",
        "AGENT_VIDEO_GEN_HEIGHT": "480",
        "AGENT_VIDEO_GEN_FPS": "16",
        "AGENT_VIDEO_GEN_VIDEO_MODE": args.video_mode,
        "AGENT_VIDEO_GEN_IMAGE_MODEL": args.image_model,
        "AGENT_VIDEO_GEN_VIDEO_MODEL": args.video.model,
        "AGENT_VIDEO_GEN_IMAGE_NUM_INFERENCE_STEPS": str(args.image_steps),
        "AGENT_VIDEO_GEN_IMAGE_GUIDANCE_SCALE": str(args.image_guidance_scale),
        # lightx2v's 4-step distill runs without CFG, on both of an A14B's experts.
        "AGENT_VIDEO_GEN_NUM_INFERENCE_STEPS": "4",
        "AGENT_VIDEO_GEN_GUIDANCE_SCALE": "1.0",
    }
    if args.video_mode == "reference":
        env["AGENT_VIDEO_GEN_REFERENCE_SCALE"] = str(args.reference_scale)
    else:
        env["AGENT_VIDEO_GEN_GUIDANCE_SCALE_2"] = "1.0"
    if (timeout := _generation_timeout(args)) is not None:
        env["AGENT_VIDEO_GEN_TIMEOUT_S"] = str(timeout)
    return env


def _train_args(args: ScriptArgs) -> str:
    planner_path = args.local(args.recipe.repo)
    reasoning_parser, tool_call_parser = resolve_reasoning_and_tool_call_parser(args.recipe.tito_model)

    # A fresh --load directory starts the run from the --ref-load weights at rollout 0.
    ckpt_args = (
        f"--hf-checkpoint {planner_path} "
        f"--ref-load {planner_path}_torch_dist "
        f"--load {args.run_dir}/checkpoints --save {args.run_dir}/checkpoints --save-interval 50 "
        "--megatron-to-hf-mode raw "
        "--custom-model-provider-path miles_plugins.models.qwen3_5.qwen3_5_mm_model_provider "
    )

    # 2,048 tokens a turn: the multi-turn recipe lost held-out episodes to thinking turns cut at 1,024.
    rollout_args = (
        f"--prompt-data {args.requests_dir}/train.jsonl "
        "--input-key messages "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--rollout-max-response-len 2048 "
        "--max-seq-len 16384 "
        "--rollout-temperature 1 "
    )

    agent_args = (
        f"--custom-generate-function-path {AGENT}.keyframe.generate.generate "
        f"--custom-agent-function-path {AGENT}.keyframe.env.run_agent "
        "--use-session-server v2 "
        f"--tito-model {args.recipe.tito_model} "
        f"--sglang-reasoning-parser {reasoning_parser} --sglang-tool-call-parser {tool_call_parser} "
        "--router miles "
        # Generation dominates an episode, so every episode of a round is in flight at once (LIN-2634).
        f"--sglang-server-concurrency {args.rollout_batch_size * args.n_samples_per_prompt} "
        f"--custom-rollout-log-function-path {AGENT}.multiturn.episode_metrics.log_episodes "
        f"--custom-eval-rollout-log-function-path {AGENT}.multiturn.episode_metrics.log_eval_episodes "
    )

    multimodal_args = (
        "--multimodal-token-space-processing "
        "--sglang-enable-token-space-processor "
        f"--shared-media-dir {args.run_dir}/media "
        "--trainer-mm-processor-worker-num 8 --trainer-mm-io-worker-num 8 "
    )
    if args.feature_cache:
        # Images are written by content hash and never rewritten, so a cached path never goes stale.
        multimodal_args += (
            "--sglang-enable-token-space-preprocess-cache "
            "--sglang-token-space-preprocess-cache-trust-url "
            f"--sglang-token-space-preprocess-cache-dir {args.run_dir}/feature_cache "
        )

    # No --group-rm, which eval refuses: the post-process standardizes each term over a prompt's episodes instead.
    reward_args = (
        f"--custom-rm-path {AGENT}.gated_reward.reward_func "
        f"--custom-reward-post-process-path {AGENT}.gated_reward.group_normalized_rewards "
        f"--reward-pool-config {command_utils.encode_pseudo_file(yaml.safe_dump(_reward_pools(args)))} "
    )
    for name, value in _reward_settings(args).items():
        values = " ".join(map(str, value)) if isinstance(value, list) else value
        reward_args += f"--{name.replace('_', '-')} {values} "

    eval_args = (
        f"--eval-interval {args.eval_interval} "
        f"--eval-config {command_utils.encode_pseudo_file(yaml.safe_dump(_eval_config(args)))} "
    )
    if args.eval_gpus:
        # The eval engine loads an HF snapshot of each eval step, so evaluation never waits on training.
        eval_args += (
            f"--eval-num-gpus {args.eval_gpus} --eval-num-gpus-per-engine 1 "
            f"--eval-hf-dir {args.run_dir}/eval_snapshots "
        )

    # TimeLens GRPO on Qwen3.5 (#4633): no KL, constant lr 1e-6, TIS against the rollout log-probs.
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.999 "
    )
    if args.optimizer_cpu_offload:
        optimizer_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "

    sglang_args = (
        f"--sglang-config {command_utils.encode_pseudo_file(yaml.safe_dump(_sglang_config(args)))} "
        f"--rollout-num-gpus {args.policy_gpus + args.image_gpus + args.video_gpus} "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.train_gpus} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 20480 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.train_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--dump-details {args.run_dir}/dump_details "
    )

    return (
        f"{ckpt_args} {rollout_args} {agent_args} {multimodal_args} {reward_args} {eval_args} {grpo_args} "
        f"{optimizer_args} {command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{sglang_args} {perf_args} {misc_args} {args.extra_args} "
    )


def _sglang_config(args: ScriptArgs) -> dict:
    def frozen(name: str, model_path: str, num_gpus: int, **group) -> dict:
        diffusion = {
            "worker_type": "regular",
            "num_gpus": num_gpus,
            "num_gpus_per_engine": 1,
            "engine_kind": "diffusion",
        }
        return {
            "name": name,
            "model_path": model_path,
            "update_weights": False,
            "server_groups": [diffusion | group],
        }

    if args.video.reference_flow_shift is not None:
        # The served directory's pipeline fuses its LoRAs at load, which the diffusers backend needs: it has no set_lora.
        video = {"overrides": {"backend": "diffusers"}}
    else:
        video = {"loras": [{"path": path, "target": target} for path, target in args.video_lora_paths()]}
    models = [
        {"name": "policy", "server_groups": [{"worker_type": "regular", "num_gpus": args.policy_gpus}]},
        frozen("image", args.local(args.image_model), args.image_gpus),
        frozen("video", args.video_path, args.video_gpus, **video),
    ]
    if args.eval_gpus:
        models.append({"name": "eval", "server_groups": [{"worker_type": "regular", "num_gpus": args.eval_gpus}]})
    return {"sglang": models}


def _reward_settings(args: ScriptArgs) -> dict:
    """The gated reward's config beyond its defaults; the eval arms score with the same settings."""
    signals = _REWARDS[args.reward]
    settings = {
        "video_reward_signal_pools": list(signals),
        "video_reward_signal_ranges": [value for signal in signals for value in _SIGNAL_RANGES[signal]],
    }
    if args.adherence_gate:
        settings |= {
            "video_reward_adherence_pool": "adherence",
            "video_reward_adherence_threshold": _ADHERENCE_THRESHOLD,
        }
    return settings


def _reward_models(args: ScriptArgs) -> list[str]:
    repos = {
        "hpsv3": [HPSV3, HPSV3_BASE],
        "vqascore": [],
        "unifiedreward": [UNIFIED_REWARD],
        "pickscore": [PICKSCORE, PICKSCORE_PROCESSOR],
    }
    return [repo for signal in _REWARDS[args.reward] for repo in repos[signal]]


def _reward_pools(args: ScriptArgs) -> list[dict]:
    """The motion judge, each signal and the optional adherence judge, colocated on consecutive rollout GPUs from the
    planner's, then the OCR pool on the CPU."""
    judge = "miles.rollout.rm_hub.video.vlm_judge.VlmJudgeScorer"
    scorers = {
        "motion": (judge, {"model_path": args.local(JUDGE), "axis_weights": {"motion": 1.0}}),
        "hpsv3": (
            "miles.rollout.rm_hub.video.hpsv3.VideoHpsv3Scorer",
            {
                "checkpoint_path": f"{args.local(HPSV3)}/HPSv3.safetensors",
                "base_model_path": args.local(HPSV3_BASE),
                # The survey's reduction, after Flash-GRPO: the mean of the top 30% of the frames' scores.
                "top_fraction": 0.3,
            },
        ),
        "vqascore": ("miles.rollout.rm_hub.video.vqascore.VideoVqaScorer", {"model_path": args.local(JUDGE)}),
        "unifiedreward": (
            "miles.rollout.rm_hub.video.unified_reward.UnifiedRewardScorer",
            {"model_path": args.local(UNIFIED_REWARD)},
        ),
        "pickscore": (
            "miles.rollout.rm_hub.video.pickscore.VideoPickScoreScorer",
            {"processor_path": args.local(PICKSCORE_PROCESSOR), "model_path": args.local(PICKSCORE)},
        ),
        "adherence": (judge, {"model_path": args.local(JUDGE), "axis_weights": {"prompt_adherence": 1.0}}),
    }
    names = ["motion", *_REWARDS[args.reward], *(["adherence"] if args.adherence_gate else [])]
    pools = [
        {
            "name": name,
            "scorer_path": scorers[name][0],
            "scorer_kwargs": scorers[name][1],
            "num_workers": 1,
            "uses_gpu": True,
            "colocated": True,
            "batch_size": 4,
        }
        for name in names
    ]
    ocr = {"name": "ocr", "scorer_path": "miles.rollout.rm_hub.video.ocr.VideoOcrScorer", "num_workers": 4}
    return pools + [ocr | {"uses_gpu": False, "batch_size": 4}]


def _eval_config(args: ScriptArgs) -> dict:
    return {
        "eval": {
            "defaults": {"n_samples_per_eval_prompt": 1, "top_k": 1, "max_response_len": 2048},
            "datasets": [{"name": "heldout", "path": f"{args.vbench_requests_dir}/heldout.jsonl"}]
            + (
                [{"name": "compbench_heldout", "path": f"{args.requests_dir}/heldout.jsonl"}]
                if args.requests == "t2v-compbench"
                else []
            ),
        }
    }


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
