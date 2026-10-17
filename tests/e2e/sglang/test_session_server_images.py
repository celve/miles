import io
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.utils import session_image_agent as agent

from miles.rollout.session.images.processing import placeholder_runs
from miles.utils.audit_utils.event_logger.logger import read_events
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.external_utils import command_utils
from miles.utils.processing_utils import call_processor, extract_multimodal_train_inputs, load_processor

register_cuda_ci(
    est_time=600, suite="stage-c-2-gpu-h200", labels=["sglang", "megatron"], hardware=["hopper"], num_gpus=2
)

MODEL_NAME = "Qwen3-VL-2B-Instruct"
MODEL_DIR = f"/root/models/{MODEL_NAME}"
MEGATRON_MODEL_TYPE = "qwen3-1.7B"
ROTARY_BASE = "5000000"
NUM_GPUS = 2
NUM_ROLLOUTS = 2
ROLLOUT_BATCH_SIZE = 2
N_SAMPLES_PER_PROMPT = 4
PROMPT_DATA = "/root/datasets/session_images_e2e.jsonl"
SYSTEM_PROMPT = "You operate a desktop. Each user turn shows the current screen."
IMAGE_TOKEN = "<|image_pad|>"
MAX_LOGPROB_ABS_DIFF = 0.05


def preflight():
    from sglang.srt.entrypoints.openai import serving_chat

    # Without it every image turn fails the session server's check, and the rollout resamples until the timeout.
    assert hasattr(serving_chat, "_collect_messages_media"), (
        "this SGLang drops message images from chat requests that carry input_ids; "
        "the session server's image turns need https://github.com/celve/sglang/pull/5"
    )


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_DIR}")
    rows = [
        {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}], "metadata": {"frame_seed": seed}}
        for seed in range(NUM_ROLLOUTS * ROLLOUT_BATCH_SIZE)
    ]
    Path(PROMPT_DATA).write_text("".join(json.dumps(row) + "\n" for row in rows))


def execute(dump_dir: Path, journal: Path):
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint {MODEL_DIR} --megatron-to-hf-mode bridge "
    rollout_args = (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--metadata-key metadata "
        f"--num-rollout {NUM_ROLLOUTS} "
        f"--rollout-batch-size {ROLLOUT_BATCH_SIZE} "
        f"--n-samples-per-prompt {N_SAMPLES_PER_PROMPT} "
        f"--global-batch-size {ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT} "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1.0 "
    )
    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        f"--custom-agent-function-path {agent.__name__}.run_agent "
        f"--custom-rm-path {agent.__name__}.reward_by_index "
        "--use-session-server v2 "
        "--session-image-support qwen3_vl "
        "--tito-model qwen3_vl "
        "--session-server-workers 2 "
    )
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    megatron_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )
    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.6 "
    debug_args = (
        f"--save-debug-rollout-data {dump_dir}/rollout_data/{{rollout_id}}.pt "
        f"--save-debug-train-data {dump_dir}/train_data/{{rollout_id}}_{{rank}}.pt "
        f"--save-debug-event-data {dump_dir}/events "
    )
    ci_args = "--ci-test --no-enable-sample-ownership-checker "
    misc_args = f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "
    train_args = (
        f"{ckpt_args} {rollout_args} {agent_args} {grpo_args} {optimizer_args} {megatron_args} {sglang_args} "
        f"{debug_args} {ci_args} {misc_args} {command_utils.get_default_wandb_args(__file__)}"
    )

    # scripts/models/qwen3-1.7B.py reads it in this process; Qwen3-VL's rope_theta is not Qwen3's.
    os.environ["MODEL_ARGS_ROTARY_BASE"] = ROTARY_BASE
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        extra_env_vars={agent.JOURNAL_PATH_ENV: str(journal)},
    )


def check(dump_dir: Path, journal: Path) -> dict:
    episodes = [json.loads(line) for line in journal.read_text().splitlines()]
    unfinished = [e for e in episodes if "error" in e or e["turns"] != agent.NUM_IMAGE_TURNS]
    assert not unfinished, f"{len(unfinished)} of {len(episodes)} episodes did not finish every turn: {unfinished[:2]}"

    processor = load_processor(MODEL_DIR, trust_remote_code=True)
    placeholder_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    rollouts = [
        torch.load(dump_dir / "rollout_data" / f"{rollout_id}.pt", weights_only=False)["samples"]
        for rollout_id in range(NUM_ROLLOUTS)
    ]
    seeds = {sample["metadata"]["frame_seed"] for samples in rollouts for sample in samples}
    expected = {seed: _expected_images(processor, seed) for seed in seeds}
    for samples in rollouts:
        _check_rollout(samples, expected, placeholder_id=placeholder_id)
    versions = [_weight_versions(samples) for samples in rollouts]
    assert max(versions[0]) < min(versions[1]), f"the engines served rollout 1 with stale weights: {versions}"
    return {
        "episodes": len(episodes),
        "rollouts": [_summarize(samples, placeholder_id=placeholder_id) for samples in rollouts],
        "weight_versions": [sorted(v) for v in versions],
        "train_batches": [
            _check_train_batch(dump_dir, rollout_id, rollouts[rollout_id]) for rollout_id in range(NUM_ROLLOUTS)
        ],
        "train": _check_training(dump_dir / "events"),
    }


def _expected_images(processor, seed: int) -> dict:
    frames = agent.episode_frames(seed)
    per_image = [_process(processor, payload) for payload in frames]
    merge = processor.image_processor.merge_size
    return {
        "ids": [agent.image_id(payload) for payload in frames],
        "counts": [int(inputs["image_grid_thw"].prod()) // (merge * merge) for inputs in per_image],
        "pixel_values": torch.cat([inputs["pixel_values"] for inputs in per_image]),
        "image_grid_thw": torch.cat([inputs["image_grid_thw"] for inputs in per_image]),
    }


def _process(processor, payload: bytes) -> dict:
    with Image.open(io.BytesIO(payload)) as image:
        return extract_multimodal_train_inputs(
            call_processor(processor, IMAGE_TOKEN, {"images": [image.convert("RGB")]})
        )


def _check_rollout(samples: list[dict], expected: dict, *, placeholder_id: int) -> None:
    assert len(samples) == ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT, f"{len(samples)} samples"
    assert len({sample["index"] for sample in samples}) == len(samples), "an episode yielded more than one sample"
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["group_index"]].append(sample["metadata"]["frame_seed"])
        _check_sample(sample, expected[sample["metadata"]["frame_seed"]], placeholder_id=placeholder_id)
    assert sorted(len(seeds) for seeds in groups.values()) == [N_SAMPLES_PER_PROMPT] * ROLLOUT_BATCH_SIZE, groups
    assert all(len(set(seeds)) == 1 for seeds in groups.values()), f"a group mixes prompts: {groups}"


def _check_sample(sample: dict, images: dict, *, placeholder_id: int) -> None:
    metadata = sample["metadata"]
    assert sample["status"] == "completed", sample["status"]
    assert metadata["tito_session_mismatch"] == [], metadata["tito_session_mismatch"]
    assert metadata["sent_image_ids"] == images["ids"]
    path_images = [metadata["prompt_image_ids"][seq] for seq in sorted(metadata["prompt_image_ids"], key=int)]
    assert path_images == [images["ids"][: turn + 1] for turn in range(agent.NUM_IMAGE_TURNS)], path_images
    tokens = sample["tokens"]
    assert placeholder_runs(tokens, placeholder_id) == images["counts"]
    inputs = sample["multimodal_train_inputs"]
    assert torch.equal(inputs["image_grid_thw"], images["image_grid_thw"]), inputs["image_grid_thw"]
    assert torch.equal(inputs["pixel_values"], images["pixel_values"]), "pixels differ from the frames the agent sent"
    loss_mask, response = sample["loss_mask"], tokens[-sample["response_length"] :]
    assert len(loss_mask) == len(response) == len(sample["rollout_log_probs"])
    assert any(loss_mask) and not all(loss_mask), "observations and completions are not both in the response"
    assert all(not mask for token, mask in zip(response, loss_mask, strict=True) if token == placeholder_id)
    assert any(logprob != 0.0 for logprob in sample["rollout_log_probs"])
    assert len(sample["weight_versions"]) == agent.NUM_IMAGE_TURNS
    assert sample["reward"] == sample["index"] % 2


def _weight_versions(samples: list[dict]) -> set[int]:
    return {int(span["version"]) for s in samples for call in s["weight_versions"] for span in call["spans"]}


def _summarize(samples: list[dict], *, placeholder_id: int) -> dict:
    return {
        "samples": len(samples),
        "tokens": [len(s["tokens"]) for s in samples],
        "response_lengths": [s["response_length"] for s in samples],
        "trainable_tokens": [sum(s["loss_mask"]) for s in samples],
        "placeholder_runs": placeholder_runs(samples[0]["tokens"], placeholder_id),
        "image_grid_thw": samples[0]["multimodal_train_inputs"]["image_grid_thw"].tolist(),
        "pixel_values_shape": list(samples[0]["multimodal_train_inputs"]["pixel_values"].shape),
    }


def _check_train_batch(dump_dir: Path, rollout_id: int, samples: list[dict]) -> dict:
    group_of = {s["index"]: s["group_index"] for s in samples}
    grid_of = {s["index"]: s["multimodal_train_inputs"]["image_grid_thw"] for s in samples}
    rewards = defaultdict(list)
    for path in sorted((dump_dir / "train_data").glob(f"{rollout_id}_*.pt")):
        batch = torch.load(path, weights_only=False)["rollout_data"]
        for index, reward, inputs in zip(
            batch["sample_indices"], batch["rewards"], batch["multimodal_train_inputs"], strict=True
        ):
            assert torch.equal(inputs["image_grid_thw"].cpu(), grid_of[index]), f"sample {index} lost its images"
            rewards[group_of[index]].append(float(reward))
    assert sorted(len(r) for r in rewards.values()) == [N_SAMPLES_PER_PROMPT] * ROLLOUT_BATCH_SIZE, rewards
    for group_rewards in rewards.values():
        assert all(r != 0.0 for r in group_rewards) and abs(sum(group_rewards)) < 1e-4, group_rewards
    return {str(group): values for group, values in rewards.items()}


def _check_training(events_dir: Path) -> dict:
    metrics = [event.metrics for event in read_events(events_dir) if isinstance(event, MetricEvent)]
    series = {
        name: _series(metrics, name)
        for name in ("train/loss", "train/grad_norm", "train/train_rollout_logprob_abs_diff")
    }
    assert all(len(values) == NUM_ROLLOUTS for values in series.values()), series
    assert all(math.isfinite(value) for values in series.values() for value in values), series
    assert all(value > 0 for value in series["train/grad_norm"]), series
    assert all(value < MAX_LOGPROB_ABS_DIFF for value in series["train/train_rollout_logprob_abs_diff"]), series
    return series


def _series(metrics: list[dict], name: str) -> list[float]:
    return [value for step in metrics for key, value in step.items() if key == name or key.endswith(f"/{name}")]


if __name__ == "__main__":
    preflight()
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    dump_dir = Path(tempfile.mkdtemp(prefix="session-images-e2e-"))
    journal = dump_dir / "episodes.jsonl"
    execute(dump_dir, journal)
    print(json.dumps(check(dump_dir, journal), indent=2))
    shutil.rmtree(dump_dir)
