from dataclasses import replace

import polars as pl
import pytest
import torch
from fastapi.testclient import TestClient
from tests.fast.dashboard.dummy_dump import dump_segmented_run

from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore
from miles.utils.types import SampleLineage


@pytest.mark.parametrize("explicit_ids", [True, False], ids=["rollout_id", "legacy_index"])
@pytest.mark.parametrize("compact_rewards", [True, False], ids=["episode_rewards", "sample_rewards"])
def test_segmented_rollouts_keep_tokens_and_equal_episode_weight(tmp_path, explicit_ids, compact_rewards):
    samples = dump_segmented_run(tmp_path, explicit_ids=explicit_ids, compact_rewards=compact_rewards)
    reader = DumpReader(tmp_path)
    joined = reader.load_joined(0)
    assert joined.train_coverage == 1.0
    for key, row in joined.train_rows.items():
        sample = [s for s in samples if s.index == key[0]][key[1]]
        assert row.raw_reward == sample.reward
        assert row.tokens.tolist() == sample.tokens
    assert reader.summary(0).height == 8
    assert reader.episodes(0)["n_samples"].to_list() == [3, 1, 2, 2]
    assert reader.episodes(0)["raw_reward"].to_list() == [1.0, 0.0, 0.25, 0.75]
    assert reader.groups(0)["n"].to_list() == [2, 2]
    assert reader.groups(0)["reward_mean"].to_list() == [0.5, 0.5]
    assert reader.groups(0)["reward_std"].to_list() == pytest.approx([2**-0.5, 0.5 * 2**-0.5])
    assert reader.step_aggregates()["reward_mean"].to_list() == [0.5]
    assert reader.episodes(0, evaluation=True)["reward"].to_list() == [1.0, 0.0, 0.25, 0.75]

    client = TestClient(make_app(MetricStore(tmp_path / "dashboard"), reader))
    for endpoint in ("summary", "episodes", "groups", "trajectories"):
        response = client.get(f"/api/rollout/0/{endpoint}")
        assert response.status_code == 200, response.text
    assert client.get("/api/advisory").status_code == 200
    episode = reader.episodes(0).row(0, named=True)
    tokens = client.get(f"/api/rollout/0/sample/{episode['sample_index']}/tokens")
    assert tokens.status_code == 200
    assert tokens.json()["token_ids"] == samples[0].tokens


def test_episode_rewards_survive_a_train_only_compact_reward_layout(tmp_path):
    dump_segmented_run(tmp_path)
    for path in (tmp_path / "train_data").glob("*.pt"):
        pack = torch.load(path, weights_only=False)
        pack["rollout_data"]["raw_reward"] = [99.0, 88.0]
        torch.save(pack, path)
    assert DumpReader(tmp_path).episodes(0)["raw_reward"].to_list() == [1.0, 0.0, 0.25, 0.75]


def test_segment_lifecycle_uses_source_identity(tmp_path):
    samples = dump_segmented_run(tmp_path)
    path = tmp_path / "rollout_data/0.pt"
    for i, sample in enumerate(samples[:3]):
        samples[i] = replace(sample, lineage=SampleLineage(source_sample_index=88, output_index=i, output_count=3))
    torch.save(dict(rollout_id=0, samples=[s.to_dict() for s in samples]), path)
    from miles.dashboard.store import TrajectoryEvent

    store = MetricStore(tmp_path / "dashboard")
    for ts, kind in [(10.0, "attempt_start"), (11.0, "attempt_end")]:
        store.append(TrajectoryEvent(ts, kind, 88, 0, -1, "", ""))
    store.flush()
    client = TestClient(make_app(store, DumpReader(tmp_path)))
    for endpoint in ("/api/rollout/0/trajectories", "/api/rollout/0/trajectories?sample_index=1"):
        assert client.get(endpoint).json()["lanes"][0]["sample_index"] == 88
    assert DumpReader(tmp_path).summary(0).filter(pl.col("sample_index") == 1)["lifecycle_index"].item() == 88


def test_real_dp_padding_does_not_join_dummy_rows_or_count_their_rewards(tmp_path):
    from tests.fast.dashboard.dummy_dump import dump_dp_padded_run

    samples = dump_dp_padded_run(tmp_path)
    reader = DumpReader(tmp_path)
    joined = reader.load_joined(0)
    assert joined.train_coverage == 1.0
    assert reader.episodes(0)["raw_reward"].to_list() == [1.0, 0.0, 0.5]
    for sample in samples:
        token = reader.tokens(0, sample.index)
        assert token["token_ids"] == sample.tokens
        assert token["train_log_probs"] is not None


def test_known_episode_reward_survives_unjoined_zero_loss_replacements(tmp_path):
    samples = dump_segmented_run(tmp_path, explicit_ids=False)
    for path in (tmp_path / "train_data").glob("*.pt"):
        pack = torch.load(path, weights_only=False)
        data = pack["rollout_data"]
        for row, index in enumerate(data["sample_indices"]):
            if index == 0:
                data["tokens"][row] = torch.tensor([0, 0])
                data["loss_masks"][row] = torch.tensor([0])
                data["response_lengths"][row] = 1
                data["total_lengths"][row] = 2
        torch.save(pack, path)
    reader = DumpReader(tmp_path)
    assert reader.episodes(0)["raw_reward"].to_list() == [1.0, 0.0, 0.25, 0.75]
    assert reader.groups(0)["reward_mean"].to_list() == [0.5, 0.5]
    assert reader.tokens(0, samples[0].index)["train_log_probs"] is None


def test_unsegmented_global_reward_overrides_survive_missing_token_rows(tmp_path):
    from tests.fast.dashboard.dummy_dump import dump_dummy_run

    dump_dummy_run(tmp_path, steps=1, dp_size=2, tp_dup=1)
    for path in (tmp_path / "train_data").glob("*.pt"):
        pack = torch.load(path, weights_only=False)
        data = pack["rollout_data"]
        data["raw_reward"] = [i + 0.25 for i in range(8)]
        count = len(data["sample_indices"])
        retained = [i for i, index in enumerate(data["sample_indices"]) if index != 0]
        for key, values in data.items():
            if key != "raw_reward" and isinstance(values, list) and len(values) == count:
                data[key] = [values[i] for i in retained]
        torch.save(pack, path)
    summary = DumpReader(tmp_path).summary(0).sort("sample_index")
    assert summary["raw_reward"].to_list() == [i + 0.25 for i in range(8)]
    assert summary["dumped_rank"][0] is None


def test_dp_padding_preserves_an_unsegmented_custom_reward_override(tmp_path):
    from tests.fast.dashboard.dummy_dump import dump_dp_padded_run

    dump_dp_padded_run(tmp_path, raw_reward_override=[9.0, 8.0, 7.0])
    assert DumpReader(tmp_path).episodes(0)["raw_reward"].to_list() == [9.0, 8.0, 7.0]
