"""Turn recorded LIBERO episodes into supervised-training files for ``run_libero_qwen38_27b.py --mode sft``.

Reads rollout dumps (``<run>/dump/rollout_data/<id>.pt``, written by ``--mode collect``), keeps the complete episodes
the simulator scored a success on the chosen trials (the first ``--per-task`` of each task), and writes
``<out>/<step>.pt``: each step one batch of whole episodes, every epoch reshuffled, in the format
``--load-debug-rollout-data`` reads. The engine's own tokens and response masks pass through unchanged, so thinking
and every prior turn are trained exactly as they were generated.

Miles deletes a run's shared media directory when the run exits, so every frame an episode references is copied into
``<out>/media`` (from the collection's ``--media-dir``, or from ``--media-backup``, a copy of it taken while the run
lived) and the references are rewritten there; an unresolvable frame is an error.

    python make_libero_sft_data.py --dumps run/dump/rollout_data/0.pt --trials 0 1 2 3 --per-task 4 \\
        --media-dir run/media --media-backup run/media_keep --out sft --batch 4 --epochs 3
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import torch


def successful_episodes(dumps: list[Path], trials: set[int], per_task: int | None) -> list[dict]:
    by_task: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for dump in dumps:
        for sample in torch.load(dump, weights_only=False)["samples"]:
            metadata = sample["metadata"]
            if (
                metadata["trial_id"] in trials
                and sample["reward"] == 1.0
                and str(sample["status"]).endswith("completed")
                and metadata.get("libero", {}).get("end_reason") == "env"
            ):
                by_task[(metadata["suite"], metadata["task_id"])].append(sample)
    return [sample for key in sorted(by_task) for sample in _across_states(by_task[key])[:per_task]]


def _across_states(samples: list[dict]) -> list[dict]:
    """One success per initial state in turn, so a cap keeps as many states as the data has."""
    by_state: dict[int, list[dict]] = defaultdict(list)
    for sample in samples:
        by_state[sample["metadata"]["trial_id"]].append(sample)
    queues = [by_state[state] for state in sorted(by_state)]
    return [q[i] for i in range(max(map(len, queues), default=0)) for q in queues if i < len(q)]


def make_media_durable(episodes: list[dict], *, media_dir: Path, backup: Path | None, out: Path) -> int:
    """Copy every referenced frame under ``out`` and point the episode at the copy; returns the frames copied."""
    prefix = str(media_dir.resolve()).rstrip("/") + "/"
    copied: set[str] = set()

    def durable(path: str) -> str:
        if not path.startswith(prefix):
            raise ValueError(f"a frame outside the collection's media dir {prefix}: {path}")
        relative = path[len(prefix) :]
        source = Path(path) if Path(path).exists() else (backup / relative if backup else None)
        if source is None or not source.exists():
            raise FileNotFoundError(f"frame {path} is in neither the media dir nor the backup")
        target = out / relative
        if relative not in copied:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.add(relative)
        return str(target.resolve())

    # In the media fields every string is a frame, so one outside the prefix is refused; metadata holds other text.
    def rewrite(value, *, every_string: bool):
        if isinstance(value, str) and (every_string or value.startswith(prefix)):
            return durable(value)
        if isinstance(value, list):
            return [rewrite(v, every_string=every_string) for v in value]
        if isinstance(value, dict):
            return {k: rewrite(v, every_string=every_string) for k, v in value.items()}
        return value

    for sample in episodes:
        sample["multimodal_inputs"] = rewrite(sample["multimodal_inputs"], every_string=True)
        sample["multimodal_train_inputs"] = rewrite(sample["multimodal_train_inputs"], every_string=True)
        metadata = dict(sample["metadata"])
        refs = rewrite(metadata.pop("prompt_media_refs", None), every_string=True)
        sample["metadata"] = {**rewrite(metadata, every_string=False), **({"prompt_media_refs": refs} if refs else {})}
    return len(copied)


def write_steps(episodes: list[dict], out: Path, *, batch: int, epochs: int, seed: int) -> int:
    if not episodes or len(episodes) % batch:
        raise ValueError(f"{len(episodes)} episodes do not fill whole batches of {batch}; every one must train once")
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    for _ in range(epochs):
        order = episodes[:]
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            samples = [dict(sample, index=i, group_index=i) for i, sample in enumerate(order[start : start + batch])]
            torch.save({"rollout_id": step, "metadata": {}, "samples": samples}, out / f"{step}.pt")
            step += 1
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dumps", type=Path, nargs="+", required=True)
    parser.add_argument("--trials", type=int, nargs="+", required=True, help="initial states whose episodes train")
    parser.add_argument("--per-task", type=int, default=None, help="the first N successes of each task")
    parser.add_argument("--media-dir", type=Path, required=True, help="the collection run's --shared-media-dir")
    parser.add_argument("--media-backup", type=Path, default=None, help="a copy of it taken while the run lived")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4, help="episodes per step")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2940)
    args = parser.parse_args()

    episodes = successful_episodes(args.dumps, set(args.trials), args.per_task)
    frames = make_media_durable(episodes, media_dir=args.media_dir, backup=args.media_backup, out=args.out / "media")
    steps = write_steps(episodes, args.out, batch=args.batch, epochs=args.epochs, seed=args.seed)
    tasks: dict[str, list[int]] = defaultdict(list)
    for sample in episodes:
        tasks[f"{sample['metadata']['suite']}/{sample['metadata']['task_id']}"].append(sample["metadata"]["trial_id"])
    summary = {
        "episodes": len(episodes),
        "frames": frames,
        "steps": steps,
        "tasks": tasks,
        "tokens": sum(len(s["tokens"]) for s in episodes),
        "trained_tokens": sum(sum(s["loss_mask"]) for s in episodes),
        "sft_flags": f"--mode sft --sft-data {args.out} --rollout-batch-size {args.batch} "
        f"--n-samples-per-prompt 1 --num-rollout {steps}",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
