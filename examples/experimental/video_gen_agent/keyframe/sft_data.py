"""An SFT set from a keyframe run's rollout dumps: each request's best episode, when it stands out from its group.

A group is one request's ``--n-samples-per-prompt`` episodes. Its top episode by reward is kept when no gate fired on
it and its reward is at least ``--margin`` standard deviations (over the whole group) above the group mean. The kept
episodes' recorded messages are written for ``sft_rollout.py``, split into train and held-out rows by a hash of the
request id; ``manifest.json`` lists what was kept by request id only.

usage: python -m examples.experimental.video_gen_agent.keyframe.sft_data <dump_details>... --out DIR
"""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from examples.experimental.video_gen_agent.keyframe.env import tools_for


def _episodes(dump_dir: Path) -> list[dict]:
    """Every scored training episode of a dump directory, in rollout order."""
    paths = sorted((dump_dir / "rollout_data").glob("[0-9]*.pt"), key=lambda p: int(p.stem))
    episodes = []
    for path in paths:
        data = torch.load(path, weights_only=False)
        for sample in data["samples"]:
            record = sample["metadata"].get("video_reward") or {}
            episodes.append(
                {
                    "group": (path.stem, sample["group_index"]),
                    "request_id": sample["metadata"].get("id"),
                    "reward": sample["reward"],
                    "scored": "scores" in record,
                    "gated": any((record.get("fired") or {}).values()),
                    "video_gen": sample["metadata"].get("video_gen") or {},
                }
            )
    return episodes


def select(episodes: list[dict], margin: float) -> list[dict]:
    """Each group's top episode, when ungated and at least ``margin`` group standard deviations above its mean."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["group"]].append(episode)
    kept = []
    for group in groups.values():
        rewards = [e["reward"] or 0.0 for e in group]
        if len(group) < 2:
            continue
        mean, std = statistics.mean(rewards), statistics.stdev(rewards)
        top = max(group, key=lambda e: e["reward"] or 0.0)
        if top["scored"] and not top["gated"] and std > 0 and top["reward"] - mean >= margin * std:
            kept.append(top | {"group_mean": mean, "group_std": std})
    return kept


def _heldout(request_id: str, percent: int) -> bool:
    return int(hashlib.sha256(request_id.encode()).hexdigest(), 16) % 100 < percent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dumps", nargs="+", type=Path, help="dump_details directories of the runs to draw from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--heldout-percent", type=int, default=5)
    args = parser.parse_args()

    episodes = [e for dump in args.dumps for e in _episodes(dump)]
    kept = select(episodes, args.margin)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "heldout": []}
    for episode in kept:
        video_gen = episode["video_gen"]
        assert "messages" in video_gen, "the dump predates the env recording its messages"
        split = "heldout" if _heldout(episode["request_id"], args.heldout_percent) else "train"
        rows[split].append(
            {
                "messages": video_gen["messages"],
                "tools": tools_for(video_gen["video_mode"]),
                "request_id": episode["request_id"],
            }
        )
    for split, split_rows in rows.items():
        (args.out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in split_rows))
    groups = len({e["group"] for e in episodes})
    manifest = {
        "dumps": [str(d) for d in args.dumps],
        "margin": args.margin,
        "episodes": len(episodes),
        "groups": groups,
        "kept": {split: [r["request_id"] for r in split_rows] for split, split_rows in rows.items()},
        "kept_rewards": {e["request_id"]: [e["reward"], e["group_mean"], e["group_std"]] for e in kept},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{groups} groups, {len(kept)} kept: {len(rows['train'])} train, {len(rows['heldout'])} held out")


if __name__ == "__main__":
    main()
