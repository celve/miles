import random
from argparse import Namespace
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import torch
from safetensors.torch import save

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.diffusion_rollout.data_source import DiffusionPromptDataSource
from miles.rollout.diffusion_rollout.response import parse_rollout_response
from miles.rollout.diffusion_rollout.rollout_fn import build_sampling_params
from miles.rollout.diffusion_rollout.step_strategy import epoch_global_random_choice, sde_window
from miles.rollout.rm_hub.ocr import image_to_rgb_uint8, ocr_reward, ocr_target_text
from miles.utils.simple_checkpointer import save_simple_checkpoint


def _tensor(t):
    return {"__tensor__": True, "data": save({"t": t.contiguous()})}


def test_response_parsing_restores_tensors_and_the_image_layout():
    body = {
        "seed": 7,
        "generated_output": _tensor(torch.rand(8, 6, 3)),
        "rollout_log_probs": _tensor(torch.tensor([-0.5, -0.6])),
        "denoising_env": {
            "pos_cond_kwargs": {
                "encoder_hidden_states": _tensor(torch.ones(4, 2)),
                "pooled_projections": _tensor(torch.ones(3)),
            },
            "neg_cond_kwargs": None,
        },
        "dit_trajectory": {
            "latents": _tensor(torch.zeros(3, 1)),
            "timesteps": _tensor(torch.zeros(3)),
            "sigmas": _tensor(torch.zeros(3)),
        },
    }

    (rollout,) = parse_rollout_response(msgpack.packb([body]))

    assert rollout.generated_output.shape == (3, 1, 8, 6)
    assert rollout.rollout_log_probs.tolist() == pytest.approx([-0.5, -0.6])
    assert rollout.denoising_env.pos_cond_kwargs.encoder_hidden_states[0].shape == (4, 2)
    assert rollout.denoising_env.neg_cond_kwargs is None
    assert rollout.dit_trajectory.latent_step_indices is None


def test_sde_window_is_a_seeded_contiguous_window_and_returns_its_latents():
    args = Namespace(diffusion_num_sde_steps=3, diffusion_sde_window_range="2,8")

    steps, latents = sde_window(args, None, 10, seed=5)

    assert steps == sde_window(args, None, 10, seed=5)[0]
    assert len(steps) == 3 and steps == list(range(steps[0], steps[0] + 3)) and 2 <= steps[0] <= 5
    assert latents == steps + [steps[-1] + 1]


def test_the_full_window_covers_every_step():
    args = Namespace(diffusion_num_sde_steps=10, diffusion_sde_window_range="0,10")

    assert sde_window(args, None, 10, seed=1)[0] == list(range(10))


def test_ocr_reward_is_one_minus_capped_normalized_edit_distance():
    pytest.importorskip("Levenshtein")
    assert ocr_target_text('A sign that reads "Take With Food" in red.') == "Take With Food"
    assert ocr_reward("TakeWithFood!", "Take With Food") == 1.0
    assert ocr_reward("Take Wth Food", "Take With Food") == pytest.approx(1 - 1 / 12)
    assert ocr_reward("zzzzzzzzzzzzzzzzzzzzzzzz", "abc") == 0.0


def test_ocr_input_is_rounded_rgb_uint8():
    image = torch.tensor([0.0, 0.5, 1.0]).view(3, 1, 1, 1).expand(3, 1, 2, 2)

    out = image_to_rgb_uint8(image)

    assert out.dtype == np.uint8 and out.shape == (2, 2, 3)
    assert out[0, 0].tolist() == [0, 128, 255]


def test_the_data_source_repeats_prompts_wraps_the_file_and_resumes(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"input": " a "}\n{"input": ""}\n{"input": "b"}\n')
    args = Namespace(
        prompt_data=str(path),
        input_key="input",
        metadata_key="metadata",
        label_key=None,
        n_samples_per_prompt=2,
        rollout_shuffle=False,
        rollout_seed=0,
    )
    source = DiffusionPromptDataSource(args)

    groups = source.get_samples(3)
    source.save(tmp_path / "ckpt")
    resumed = DiffusionPromptDataSource(args)
    resumed.load(tmp_path / "ckpt")

    assert [[s.prompt for s in g] for g in groups] == [["a", "a"], ["b", "b"], ["a", "a"]]
    assert [[(s.group_index, s.index) for s in g] for g in groups][2] == [(2, 4), (2, 5)]
    assert [(s.prompt, s.group_index, s.index) for s in resumed.get_samples(1)[0]] == [("b", 3, 6), ("b", 3, 7)]


def test_num_epoch_counts_the_diffusion_prompts(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(f'{{"input": "p{i}"}}\n' for i in range(6)))
    args = Namespace(
        prompt_data=str(path),
        input_key="input",
        metadata_key="metadata",
        label_key=None,
        n_samples_per_prompt=2,
        rollout_global_dataset=True,
        rollout_batch_size=2,
        rollout_shuffle=False,
        rollout_seed=0,
    )
    executor = SimpleNamespace(args=args, data_source=DiffusionPromptDataSource(args))

    assert RolloutExecutor.get_num_rollout_per_epoch(executor) == 3


def test_rollout_shuffle_permutes_each_epoch_by_seed_and_resumes(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(f'{{"input": "p{i}"}}\n' for i in range(5)))
    args = Namespace(
        prompt_data=str(path),
        input_key="input",
        metadata_key="metadata",
        label_key=None,
        n_samples_per_prompt=1,
        rollout_shuffle=True,
        rollout_seed=7,
    )
    source = DiffusionPromptDataSource(args)

    first_epoch = [g[0].prompt for g in source.get_samples(5)]
    source.get_samples(2)
    source.save(tmp_path / "ckpt")
    resumed = DiffusionPromptDataSource(args)
    resumed.load(tmp_path / "ckpt")

    expected = []
    for epoch in range(2):
        order = list(range(5))
        random.Random(7 + epoch).shuffle(order)
        expected.append([f"p{i}" for i in order])
    assert first_epoch == expected[0] != [f"p{i}" for i in range(5)]
    assert [g[0].prompt for g in resumed.get_samples(3)] == expected[1][2:]


def test_a_checkpoint_saved_before_rollout_shuffle_still_loads(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(f'{{"input": "p{i}"}}\n' for i in range(3)))
    args = Namespace(
        prompt_data=str(path),
        input_key="input",
        metadata_key="metadata",
        label_key=None,
        n_samples_per_prompt=1,
        rollout_shuffle=False,
        rollout_seed=0,
    )
    save_simple_checkpoint(
        directory=tmp_path / "ckpt", data=dict(prompt_offset=2, sample_group_index=5, sample_index=5)
    )
    source = DiffusionPromptDataSource(args)

    source.load(tmp_path / "ckpt")

    assert [(g[0].prompt, g[0].index) for g in source.get_samples(2)] == [("p2", 5), ("p0", 6)]


def test_a_label_key_carries_each_rows_label_to_its_samples(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"input": "a", "answer": "A"}\n{"input": "b", "answer": "B"}\n')
    args = Namespace(
        prompt_data=str(path),
        input_key="input",
        metadata_key="metadata",
        label_key="answer",
        n_samples_per_prompt=2,
        rollout_shuffle=False,
        rollout_seed=0,
    )

    groups = DiffusionPromptDataSource(args).get_samples(2)

    assert [[s.label for s in g] for g in groups] == [["A", "A"], ["B", "B"]]


def _request_args(**overrides):
    values = dict(
        diffusion_width=480,
        diffusion_height=480,
        diffusion_num_steps=10,
        diffusion_guidance_scale=4.0,
        diffusion_sde_type="sde",
        diffusion_noise_level=0.9,
        diffusion_quality="exact",
        diffusion_negative_prompt=None,
        diffusion_output_num_frames=1,
        diffusion_guidance_scale_2=None,
    )
    return Namespace(**(values | overrides))


def test_an_image_request_keeps_one_frame_and_one_guidance_scale():
    params = build_sampling_params(_request_args())

    assert params["num_frames"] == 1 and params["extra_sampling_params"] == {"quality": "exact"}


def test_a_video_request_carries_its_frames_and_the_second_guidance_scale():
    params = build_sampling_params(_request_args(diffusion_output_num_frames=5, diffusion_guidance_scale_2=3.0))

    assert params["num_frames"] == 5
    assert params["extra_sampling_params"] == {"quality": "exact", "guidance_scale_2": 3.0}


def _choice(group_index, *, num_sde_steps=1, candidates="1,2,3"):
    args = Namespace(
        diffusion_sde_candidate_steps=candidates,
        diffusion_num_sde_steps=num_sde_steps,
        rollout_batch_size=4,
        rollout_seed=42,
    )
    return epoch_global_random_choice(args, SimpleNamespace(group_index=group_index), 10, seed=group_index)


def test_one_rollout_draws_one_sde_step_for_all_its_groups_seeded_by_the_rollout():
    rollouts = [[_choice(group_index) for group_index in range(rollout * 4, rollout * 4 + 4)] for rollout in range(4)]

    for rollout, choices in enumerate(rollouts):
        (choice,) = {(tuple(steps), tuple(latents)) for steps, latents in choices}
        step = [1, 2, 3][torch.randperm(3, generator=torch.Generator().manual_seed(rollout + 42))[0].item()]
        assert choice == ((step,), (step, step + 1))


def test_drawing_every_candidate_takes_them_all_in_order():
    assert _choice(0, num_sde_steps=3, candidates="3,1,2") == ([1, 2, 3], [1, 2, 3, 4])


@pytest.mark.parametrize("candidates,message", [(None, "needs --diffusion-sde-candidate-steps"), ("1,10", r"\[10\]")])
def test_candidate_steps_must_be_given_and_inside_the_schedule(candidates, message):
    with pytest.raises(ValueError, match=message):
        _choice(0, candidates=candidates)
