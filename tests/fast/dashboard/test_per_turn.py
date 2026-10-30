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


def test_bad_reward_cardinality_is_reported_instead_of_misattributed(tmp_path):
    dump_segmented_run(tmp_path)
    path = tmp_path / "train_data/0_0.pt"
    pack = torch.load(path, weights_only=False)
    pack["rollout_data"]["raw_reward"] = [99.0, 88.0]
    torch.save(pack, path)
    with pytest.raises(AssertionError, match="8 samples or 4 rollouts, got 2"):
        DumpReader(tmp_path).load_joined(0)


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
