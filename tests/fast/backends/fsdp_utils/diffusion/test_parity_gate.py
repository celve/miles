import json

import pytest

from tests.e2e.fsdp import test_sd35_medium_flow_grpo_ocr_2xGPU as gate


def _write_standard_series(path, *, nudge=0.0):
    want = json.loads(gate.STANDARD.read_text())["metrics"]
    records = [
        {key: values[i] + (nudge if key == "train/loss" else 0.0) for key, values in want.items() if i < len(values)}
        for i in range(max(len(values) for values in want.values()))
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_the_standard_gates_normalized_rewards_and_loss():
    want = json.loads(gate.STANDARD.read_text())["metrics"]

    assert {"rollout/reward/norm_mean", "rollout/reward/norm_std", "train/loss"} <= gate.TOLERANCES.keys()
    assert gate.TOLERANCES.keys() == want.keys()


def test_exact_mode_rejects_a_difference_the_tolerances_absorb(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics.jsonl"
    _write_standard_series(metrics, nudge=1e-9)

    gate.check(metrics)
    monkeypatch.setattr(gate, "EXACT", True)
    with pytest.raises(AssertionError, match="train/loss"):
        gate.check(metrics)


def test_exact_mode_accepts_the_standard_itself(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics.jsonl"
    _write_standard_series(metrics)
    monkeypatch.setattr(gate, "EXACT", True)

    gate.check(metrics)
