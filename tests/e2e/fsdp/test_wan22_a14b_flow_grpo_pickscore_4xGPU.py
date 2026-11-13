"""Wan2.2-T2V-A14B Flow-GRPO with the PickScore reward for 4 rollouts (8 optimizer steps) on 4 colocated GPUs, a proxy
of the 5-GPU recipe that scores PickScore on CPU instead of a fifth GPU, checked against the metric series recorded in
the companion JSON.

Four rollouts of 4 prompts x 8 videos: with the default --rollout-seed they draw SDE steps 1, 2, 2 and 3, so the first
three train the high-noise expert and the last the low-noise one. miles_diffusion has no gate for this recipe; the
standard was recorded against it on the same box and image (see the JSON's meta), and
MILES_DIFFUSION_PARITY_EXACT=1 drops every tolerance to 0 for that check.
"""

import json
import math
import os
import tempfile
from pathlib import Path

from scripts import run_wan22_a14b_flow_grpo_pickscore as recipe
from tests.ci.ci_register import register_cuda_ci
from tests.utils.dirs import get_test_data_dir, get_test_model_dir

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=2400, suite="stage-c-4-gpu-h200", num_gpus=4, labels=["fsdp"], hardware=["hopper"])

STANDARD = Path(__file__).with_name("wan22_a14b_flow_grpo_pickscore_standard.json")
TOLERANCES = {
    "rollout/reward/raw_num_samples": 0.0,
    "rollout/reward/raw_mean": 0.01,
    "rollout/reward/raw_median": 0.01,
    "rollout/reward/raw_std": 0.01,
    "rollout/reward/norm_mean": 1e-6,
    "rollout/reward/norm_median": 0.1,
    "rollout/reward/norm_std": 0.05,
    "train/log_prob_old_idx_0": 1e-3,
    "train/log_prob_new_idx_0": 1e-3,
    "train/log_prob_mean_abs_diff": 1e-3,
    "train/grad_norm": 1e-3,
    "train/loss": 1e-4,
    # Their point counts say which expert each step trained: 6 high-noise steps, then 2 low-noise.
    "train/log_prob_mean_abs_diff_transformer": 1e-3,
    "train/log_prob_mean_abs_diff_transformer_2": 1e-3,
}
EXACT = os.environ.get("MILES_DIFFUSION_PARITY_EXACT") == "1"


def _series(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        for key in TOLERANCES.keys() & record.keys():
            out.setdefault(key, []).append(record[key])
    return out


def _script_args() -> command_utils.ExecuteTrainConfig:
    args = command_utils.default_config(recipe.ScriptArgs)
    args.num_rollout = 4
    args.rollout_batch_size = 4
    args.n_samples_per_prompt = 8
    args.pickscore_gpus = 0
    args.model_dir = get_test_model_dir()
    args.data_dir = get_test_data_dir()
    return args


def execute(metrics_path: Path) -> None:
    os.environ["MILES_DIFFUSION_METRICS_JSONL"] = str(metrics_path)
    recipe.execute(_script_args())


def check(metrics_path: Path) -> None:
    got, want = _series(metrics_path), json.loads(STANDARD.read_text())["metrics"]
    for key, tolerance in TOLERANCES.items():
        assert len(got.get(key, [])) == len(want[key]), f"{key}: {got.get(key)} vs {want[key]}"
        tolerance = 0.0 if EXACT else tolerance
        for step, (g, w) in enumerate(zip(got[key], want[key], strict=True)):
            assert math.isfinite(g) and abs(g - w) <= tolerance, f"{key}[{step}]: {g} vs {w} (tolerance {tolerance})"
    assert got["train/log_prob_mean_abs_diff"][0] < 2e-4, "engine and trainer log-probs disagree at the first step"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics = Path(tmp_dir) / "metrics.jsonl"
        recipe.prepare(_script_args())
        execute(metrics)
        check(metrics)
