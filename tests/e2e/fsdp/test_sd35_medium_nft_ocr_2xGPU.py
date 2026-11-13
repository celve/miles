"""SD3.5-medium DiffusionNFT with the OCR reward for 4 rollouts (4 optimizer steps) on 2 colocated GPUs, checked
against the metric series recorded in the companion JSON.

Four rollouts, as the fork's NFT gate uses: step 1 overflows the fp16 grad scaler and is skipped (its grad norm is
NaN by design), so the EMA still equals the policy through step 2 and the two loss branches first separate at step 3.
The standard was recorded on 2xB300 in radixark/miles:dev-202610300231, where radixark/miles_diffusion at ecb1363e
reproduces it bit for bit; MILES_DIFFUSION_PARITY_EXACT=1 drops every tolerance to 0 for that check.
"""

import json
import math
import os
import tempfile
from pathlib import Path

from scripts import run_sd35_medium_nft_ocr as recipe
from tests.ci.ci_register import register_cuda_ci
from tests.utils.dirs import get_test_data_dir, get_test_model_dir

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=900, suite="stage-c-2-gpu-h200", num_gpus=2, labels=["fsdp"], hardware=["hopper"])

STANDARD = Path(__file__).with_name("sd35_medium_nft_ocr_standard.json")
TOLERANCES = {
    "rollout/reward/raw_num_samples": 0.0,
    "rollout/reward/raw_mean": 0.05,
    "rollout/reward/raw_median": 0.1,
    "rollout/reward/raw_std": 0.05,
    "train/grad_norm": 1e-3,
    "train/nft_loss": 0.5,
    "train/nft_pos_loss": 0.05,
    "train/nft_neg_loss": 0.05,
    "train/nft_r_mean": 0.05,
    "train/nft_t_mean": 1e-6,
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
            if math.isnan(w):
                assert math.isnan(g), f"{key}[{step}]: {g}, recorded as the skipped fp16 step"
                continue
            assert math.isfinite(g) and abs(g - w) <= tolerance, f"{key}[{step}]: {g} vs {w} (tolerance {tolerance})"
    assert got["train/nft_pos_loss"][1] == got["train/nft_neg_loss"][1], "the EMA moved before the policy did"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics = Path(tmp_dir) / "metrics.jsonl"
        recipe.prepare(_script_args())
        execute(metrics)
        check(metrics)
