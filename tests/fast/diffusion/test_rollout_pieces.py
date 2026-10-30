from argparse import Namespace

import msgpack
import numpy as np
import pytest
import torch
from safetensors.torch import save

from miles.diffusion.rollout.data_source import DiffusionPromptDataSource
from miles.diffusion.rollout.ocr import image_to_rgb_uint8, ocr_reward, ocr_target_text
from miles.diffusion.rollout.response import parse_rollout_response
from miles.diffusion.rollout.step_strategy import sde_window


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
    args = Namespace(prompt_data=str(path), input_key="input", metadata_key="metadata", n_samples_per_prompt=2)
    source = DiffusionPromptDataSource(args)

    groups = source.get_samples(3)
    source.save(tmp_path / "ckpt")
    resumed = DiffusionPromptDataSource(args)
    resumed.load(tmp_path / "ckpt")

    assert [[s.prompt for s in g] for g in groups] == [["a", "a"], ["b", "b"], ["a", "a"]]
    assert [[(s.group_index, s.index) for s in g] for g in groups][2] == [(2, 4), (2, 5)]
    assert [(s.prompt, s.group_index, s.index) for s in resumed.get_samples(1)[0]] == [("b", 3, 6), ("b", 3, 7)]
