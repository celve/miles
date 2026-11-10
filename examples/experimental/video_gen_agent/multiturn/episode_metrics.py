"""Rollout log hooks that count each episode once, however many session v2 branches folding split it into.

``--custom-rollout-log-function-path examples.experimental.video_gen_agent.multiturn.episode_metrics.log_episodes``
adds per-episode context and timing to the rollout's metrics, and prunes the feature cache when the run
shares one. Every branch's sample carries the episode's ``video_gen`` record. The per-episode training reward is
Miles' own ``rollout/episode_raw_reward``.

  episode/prompt_tokens_per_turn  the policy's prompt per turn, as the engine counted it (clips expanded)
  episode/peak_prompt_tokens      the episode's largest prompt
  episode/turn_seconds            wall time of a policy call
  episode/folded_clips, episode/recalls, episode/shots_submitted, episode/samples
  episode/images                  images per episode, in the keyframe env
  episode/video_prompt_words      words in the submitted video's generator prompt
  episode/image_prompt_words      words per image prompt, in the keyframe env
  reward/<gate>_fired             under ``gated_reward``, the share of submitted episodes each gate fired on
  reward/score_<pool>, reward/term_<pool>  its mean pool scores and gated terms over submitted episodes

``--custom-eval-rollout-log-function-path examples.experimental.video_gen_agent.multiturn.episode_metrics.log_eval_episodes``
keeps one row per episode in each eval dataset before Miles' own eval metrics run, so ``eval/<dataset>`` is the
mean episode reward: a K that folds more would otherwise weigh its episodes by their branch count. Per-sample eval
statistics then describe each episode's first branch.
"""

import numpy as np

from examples.experimental.video_gen_agent.gated_reward import GATES, RECORD_KEY
from examples.experimental.video_gen_agent.multiturn.feature_cache import prune_feature_cache
from miles.ray.rollout.metrics import _keep_eval_rows


def log_episodes(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    if args.sglang.common_value("token_space_preprocess_cache_dir"):
        prune_feature_cache(rollout_id, args, samples, rollout_extra_metrics, rollout_time)
    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(episode_metrics(samples))
    return False


def log_eval_episodes(rollout_id, args, data, extra_metrics) -> bool:
    for name, dataset in data.items():
        if dataset.get("samples") is not None:
            # Replaces the dataset in the dict Miles logs next; nothing reads eval data after logging.
            data[name] = _keep_eval_rows(dataset, keep=_first_of_each_episode(dataset["samples"]))
    return False


def _first_of_each_episode(samples) -> list[bool]:
    """True for each episode's first sample; a sample without a session (an aborted attempt) is its own episode."""
    seen, keep = set(), []
    for sample in samples:
        session_id = sample.metadata.get("session_id")
        keep.append(session_id is None or session_id not in seen)
        seen.add(session_id)
    return keep


def episode_metrics(samples) -> dict[str, float]:
    episodes: dict[str, dict] = {}
    rewards: dict[str, dict] = {}
    branches: dict[str, int] = {}
    for sample in samples:
        record = sample.metadata.get("video_gen")
        if record is None:
            continue
        session_id = sample.metadata["session_id"]
        episodes[session_id] = record
        if (reward := sample.metadata.get(RECORD_KEY)) is not None:
            rewards[session_id] = reward
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
        "episode/samples": np.mean(list(branches.values())).item(),
    }
    # The clip env records folds and recalls; the keyframe env records its images instead.
    for key in ("folded_clips", "recalls"):
        if counts := [e[key] for e in episodes.values() if key in e]:
            metrics[f"episode/{key}"] = np.mean(counts).item()
    if images := [len(e["images"]) for e in episodes.values() if "images" in e]:
        metrics["episode/images"] = np.mean(images).item()
    if image_words := [len(i["prompt"].split()) for e in episodes.values() for i in e.get("images", [])]:
        metrics["episode/image_prompt_words"] = np.mean(image_words).item()
    if video_words := [len(s["prompt"].split()) for s in submitted if s.get("prompt")]:
        metrics["episode/video_prompt_words"] = np.mean(video_words).item()
    if tokens:
        metrics["episode/prompt_tokens_per_turn"] = np.mean([np.mean(episode) for episode in tokens]).item()
        metrics["episode/peak_prompt_tokens"] = np.mean([max(episode) for episode in tokens]).item()
    if submitted:
        metrics["episode/shots_submitted"] = np.mean([len(s.get("shots", [s["clip_id"]])) for s in submitted]).item()
    return metrics | _reward_metrics([r for r in rewards.values() if r["submitted"] and r["reward"] is not None])


def _reward_metrics(records: list[dict]) -> dict[str, float]:
    if not records:
        return {}
    metrics = {f"reward/{gate}_fired": np.mean([r["fired"][gate] for r in records]).item() for gate in GATES}
    for part in ("scores", "terms"):
        for name in records[0][part]:
            metrics[f"reward/{part[:-1]}_{name.removeprefix('signal:')}"] = np.mean(
                [r[part][name] for r in records]
            ).item()
    return metrics
