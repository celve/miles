"""SD3.5-medium Flow-GRPO with the OCR reward for 2 rollouts (4 optimizer steps) on 2 colocated GPUs, checked
against the metric series recorded in the companion JSON.

The run is deterministic: on the recording image the series matches bit for bit, and so does radixark/miles_diffusion
at 221f965f on that image. The tolerances only absorb engine, torch and PaddleOCR drift between images; the
first step's log_prob_mean_abs_diff is the engine/trainer log-prob agreement that guards the port itself.
"""

import json
import math
import os
import tempfile
from pathlib import Path

from scripts import run_sd35_medium_flow_grpo_ocr as recipe
from tests.ci.ci_register import register_cuda_ci
from tests.utils.dirs import get_test_data_dir, get_test_model_dir

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=900, suite="stage-c-2-gpu-h200", num_gpus=2, labels=["fsdp"], hardware=["hopper"])

STANDARD = Path(__file__).with_name("sd35_medium_flow_grpo_ocr_standard.json")
TOLERANCES = {
    "rollout/reward/raw_num_samples": 0.0,
    "rollout/reward/raw_mean": 0.05,
    "rollout/reward/raw_median": 0.1,
    "rollout/reward/raw_std": 0.05,
    "train/log_prob_old_idx_0": 1e-3,
    "train/log_prob_new_idx_0": 1e-3,
    "train/log_prob_mean_abs_diff": 1e-3,
    "train/grad_norm": 1e-3,
}


def _series(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        for key in TOLERANCES.keys() & record.keys():
            out.setdefault(key, []).append(record[key])
    return out


def _script_args() -> recipe.ScriptArgs:
    args = command_utils.default_config(recipe.ScriptArgs)
    args.num_rollout = 2
    args.model_dir = get_test_model_dir()
    args.data_dir = get_test_data_dir()
    return args


def prepare() -> None:
    recipe.prepare(_script_args())


def execute(metrics_path: Path) -> None:
    os.environ["MILES_DIFFUSION_METRICS_JSONL"] = str(metrics_path)
    recipe.execute(_script_args())


def check(metrics_path: Path) -> None:
    got, want = _series(metrics_path), json.loads(STANDARD.read_text())["metrics"]
    for key, tolerance in TOLERANCES.items():
        assert len(got.get(key, [])) == len(want[key]), f"{key}: {got.get(key)} vs {want[key]}"
        for step, (g, w) in enumerate(zip(got[key], want[key], strict=True)):
            assert math.isfinite(g) and abs(g - w) <= tolerance, f"{key}[{step}]: {g} vs {w} (tolerance {tolerance})"
    assert got["train/log_prob_mean_abs_diff"][0] < 2e-4, "the trainer does not re-score the engine's own transitions"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics = Path(tmp_dir) / "metrics.jsonl"
        prepare()
        execute(metrics)
        check(metrics)
