"""``--custom-rollout-log-function-path examples.experimental.video_gen_agent.multiturn.episode_metrics.log_episodes``:
adds per-episode context and timing to the rollout's metrics, and prunes the feature cache when the run shares one.

Session v2 turns one episode into a sample per branch, and every sample carries the episode's ``video_gen`` record,
so the metrics count each episode once.

  episode/prompt_tokens_per_turn  the policy's prompt per turn, as the engine counted it (clips expanded)
  episode/peak_prompt_tokens      the episode's largest prompt
  episode/turn_seconds            wall time of a policy call
  episode/folded_clips, episode/recalls, episode/shots_submitted, episode/samples
"""

import numpy as np

from examples.experimental.video_gen_agent.multiturn.feature_cache import prune_feature_cache


def log_episodes(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    if args.sglang.common_value("token_space_preprocess_cache_dir"):
        prune_feature_cache(rollout_id, args, samples, rollout_extra_metrics, rollout_time)
    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(episode_metrics(samples))
    return False


def episode_metrics(samples) -> dict[str, float]:
    episodes: dict[str, dict] = {}
    branches: dict[str, int] = {}
    for sample in samples:
        record = sample.metadata.get("video_gen")
        if record is None:
            continue
        session_id = sample.metadata["session_id"]
        episodes[session_id] = record
        branches[session_id] = branches.get(session_id, 0) + 1
    if not episodes:
        return {}
    tokens = [
        [t["prompt_tokens"] for t in e["turn_stats"] if t["prompt_tokens"] is not None] for e in episodes.values()
    ]
    tokens = [episode for episode in tokens if episode]
    submitted = [e["submitted"] for e in episodes.values() if e["submitted"]]
    metrics = {
        "episode/turn_seconds": np.mean([t["seconds"] for e in episodes.values() for t in e["turn_stats"]]).item(),
        "episode/folded_clips": np.mean([e["folded_clips"] for e in episodes.values()]).item(),
        "episode/recalls": np.mean([e["recalls"] for e in episodes.values()]).item(),
        "episode/samples": np.mean(list(branches.values())).item(),
    }
    if tokens:
        metrics["episode/prompt_tokens_per_turn"] = np.mean([np.mean(episode) for episode in tokens]).item()
        metrics["episode/peak_prompt_tokens"] = np.mean([max(episode) for episode in tokens]).item()
    if submitted:
        metrics["episode/shots_submitted"] = np.mean([len(s.get("shots", [s["clip_id"]])) for s in submitted]).item()
    return metrics
