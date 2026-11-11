"""Score eval-arm clips with the training reward and the outside checks, on one GPU.

  python -m examples.experimental.video_gen_agent.keyframe.eval.score clips.jsonl... --out scored.jsonl \\
      [--model-dir /root/models] [--reward <the launcher's --reward>] [--adherence-gate] [--anchor-dir <dir>]

Each row gains ``scores`` (every training pool's score, the three-axis Qwen3-VL ``judge`` and ``pickscore``),
``reward`` (the training reward's gated mean, as eval reports it), ``fired`` (its gates), ``frame_diff`` (mean absolute
difference of consecutive grayscale frames, 0-255), ``ocr_lines`` and ``prompt_words``. The pools and the reward's
settings come from the launcher, so they are the run's. A row without a clip scores reward 0.
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from examples.experimental.video_gen_agent.gated_reward import GatedVideoRewardConfig, clip_record
from examples.experimental.video_gen_agent.keyframe.run_qwen35_9b_keyframe import (
    JUDGE,
    PICKSCORE,
    PICKSCORE_PROCESSOR,
    ScriptArgs,
    _reward_pools,
    _reward_settings,
)

from miles.utils.function_registry import load_function


def reward_config(args: ScriptArgs) -> GatedVideoRewardConfig:
    return GatedVideoRewardConfig(**_reward_settings(args))


def scorer_specs(args: ScriptArgs) -> dict[str, tuple[str, dict]]:
    """The run's reward pools, plus the outside checks: the three-axis judge and PickScore."""
    specs = {pool["name"]: (pool["scorer_path"], pool.get("scorer_kwargs", {})) for pool in _reward_pools(args)}
    specs["judge"] = ("miles.rollout.rm_hub.video.vlm_judge.VlmJudgeScorer", {"model_path": args.local(JUDGE)})
    specs.setdefault(
        "pickscore",
        (
            "miles.rollout.rm_hub.video.pickscore.VideoPickScoreScorer",
            {"processor_path": args.local(PICKSCORE_PROCESSOR), "model_path": args.local(PICKSCORE)},
        ),
    )
    return specs


def frame_diff(path: str) -> float:
    with av.open(path) as container:
        frames = [frame.to_ndarray(format="gray").astype(np.float32) for frame in container.decode(video=0)]
    return float(np.mean([np.abs(b - a).mean() for a, b in zip(frames, frames[1:], strict=False)]))


def score_rows(rows: list[dict], scorers: dict, config: GatedVideoRewardConfig) -> list[dict]:
    """``scorers`` maps a pool name to an object with the pools' ``score_batch(paths, requests)``."""
    with_clip = [row for row in rows if row["path"] is not None]
    scores = {
        name: scorer.score_batch([row["path"] for row in with_clip], [row["request"] for row in with_clip])
        for name, scorer in scorers.items()
    }
    parts = {f"signal:{pool}": pool for pool in config.video_reward_signal_pools}
    parts["ocr"] = config.video_reward_ocr_pool
    if config.video_reward_motion_pool is not None:
        parts["motion"] = config.video_reward_motion_pool
    if config.video_reward_adherence_pool is not None:
        parts["adherence"] = config.video_reward_adherence_pool
    out = []
    for index, row in enumerate(with_clip):
        clip_scores = {name: values[index] for name, values in scores.items()}
        record = clip_record(config, {part: clip_scores[name] for part, name in parts.items()})
        out.append(
            row
            | {
                "scores": clip_scores,
                "reward": record["reward"],
                "fired": record.get("fired"),
                "frame_diff": frame_diff(row["path"]),
                "ocr_lines": clip_scores[config.video_reward_ocr_pool],
                "prompt_words": len(row["prompt"].split()),
            }
        )
    missing = [
        row | {"scores": {}, "reward": 0.0, "fired": None, "frame_diff": None, "ocr_lines": None}
        for row in rows
        if row["path"] is None
    ]
    return out + [row | {"prompt_words": len(row["prompt"].split()) if row["prompt"] else None} for row in missing]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", default="/root/models")
    parser.add_argument("--reward", default=ScriptArgs.reward, help="as the launcher's --reward, and its default")
    parser.add_argument("--adherence-gate", action="store_true")
    parser.add_argument("--anchor-dir", help="as the launcher's --anchor-dir, for --reward pairwise")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run = ScriptArgs(
        run_id="eval",
        model_dir=args.model_dir,
        reward=args.reward,
        adherence_gate=args.adherence_gate,
        anchor_dir=args.anchor_dir,
    )
    rows = [json.loads(line) for path in args.clips for line in path.read_text().splitlines()]
    scored = {}
    # One scorer on the GPU at a time: the judge and the reward models do not all fit beside each other on one GPU.
    for name, (path, kwargs) in scorer_specs(run).items():
        scorer = load_function(path)(device=torch.device(args.device), **kwargs)
        scored[name] = _Cached(scorer, rows)
        del scorer
        torch.accelerator.empty_cache()
    rows = score_rows(rows, scored, reward_config(run))
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))


class _Cached:
    """A scorer's scores of every row's clip, computed once, served through ``score_batch``."""

    def __init__(self, scorer, rows: list[dict]):
        with_clip = [row for row in rows if row["path"] is not None]
        values = scorer.score_batch([row["path"] for row in with_clip], [row["request"] for row in with_clip])
        self.scores = {(row["path"], row["request"]): value for row, value in zip(with_clip, values, strict=True)}

    def score_batch(self, paths: list[str], requests: list[str]) -> list[float | None]:
        return [self.scores[key] for key in zip(paths, requests, strict=True)]


if __name__ == "__main__":
    main()
