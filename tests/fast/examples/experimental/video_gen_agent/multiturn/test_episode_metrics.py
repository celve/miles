from types import SimpleNamespace

from examples.experimental.video_gen_agent.multiturn import episode_metrics
from examples.experimental.video_gen_agent.multiturn.episode_metrics import log_episodes

from miles.utils.types import Sample


def _record(tokens, seconds, *, folded, recalls, submitted):
    turns = [{"prompt_tokens": t, "seconds": s} for t, s in zip(tokens, seconds, strict=True)]
    return {"turn_stats": turns, "folded_clips": folded, "recalls": recalls, "submitted": submitted}


def _samples():
    chained = {"clip_id": "clip_3", "path": "/kept/a.mp4", "shots": ["clip_1", "clip_2", "clip_3"]}
    folded = _record([1000, 3000, 2000], [1.0, 2.0, 3.0], folded=2, recalls=1, submitted=chained)
    plain = _record(
        [500, None], [0.5, 1.5], folded=0, recalls=0, submitted={"clip_id": "clip_1", "path": "/kept/b.mp4"}
    )
    # A folded episode comes back as one sample per branch, each with the episode's record.
    return [Sample(metadata={"session_id": "a", "video_gen": folded}) for _ in range(3)] + [
        Sample(metadata={"session_id": "b", "video_gen": plain}),
        Sample(metadata={"session_id": "c"}),
    ]


def test_each_episode_counts_once_however_many_branches_it_left():
    assert episode_metrics.episode_metrics(_samples()) == {
        "episode/turn_seconds": 1.6,
        "episode/folded_clips": 1.0,
        "episode/recalls": 0.5,
        "episode/samples": 2.0,
        "episode/prompt_tokens_per_turn": 1250.0,
        "episode/peak_prompt_tokens": 1750.0,
        "episode/shots_submitted": 2.0,
    }


def test_the_hook_adds_the_metrics_and_prunes_only_a_configured_cache(monkeypatch):
    pruned = []
    monkeypatch.setattr(episode_metrics, "prune_feature_cache", lambda *args: pruned.append(args[0]))
    extra = {"rollout/x": 1}

    for cache_dir in (None, "/cache"):
        args = SimpleNamespace(sglang=SimpleNamespace(common_value=lambda name, value=cache_dir: value))
        assert log_episodes(7, args, _samples(), extra, 10.0) is False

    assert pruned == [7]
    assert extra["rollout/x"] == 1 and extra["episode/samples"] == 2.0
