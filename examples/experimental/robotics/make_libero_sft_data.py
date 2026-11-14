"""Turn recorded LIBERO episodes into supervised-training files for ``run_libero_qwen38_27b.py --mode sft``.

Reads rollout dumps (``<run>/dump/rollout_data/<id>.pt``, written by ``--mode collect``), keeps the complete episodes
the simulator scored a success on the chosen trials, and writes ``<out>/<step>.pt``: each step one batch of whole
episodes, every epoch reshuffled, in the format ``--load-debug-rollout-data`` reads. The engine's own tokens and
response masks pass through unchanged, so thinking and every prior turn are trained exactly as they were generated.

    python make_libero_sft_data.py --dumps run/dump/rollout_data/0.pt --trials 0 1 2 3 --out sft --batch 4 --epochs 3
"""

import argparse
import json
import random
from pathlib import Path

import torch


def successful_episodes(dumps: list[Path], trials: set[int]) -> list[dict]:
    episodes = []
    for dump in dumps:
        for sample in torch.load(dump, weights_only=False)["samples"]:
            metadata = sample["metadata"]
            if (
                metadata["trial_id"] in trials
                and sample["reward"] == 1.0
                and str(sample["status"]).endswith("completed")
                and metadata.get("libero", {}).get("end_reason") == "env"
            ):
                episodes.append(sample)
    return episodes


def write_steps(episodes: list[dict], out: Path, *, batch: int, epochs: int, seed: int) -> int:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    for _ in range(epochs):
        order = episodes[:]
        rng.shuffle(order)
        for start in range(0, len(order) - batch + 1, batch):
            samples = [dict(sample, index=i, group_index=i) for i, sample in enumerate(order[start : start + batch])]
            torch.save({"rollout_id": step, "metadata": {}, "samples": samples}, out / f"{step}.pt")
            step += 1
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dumps", type=Path, nargs="+", required=True)
    parser.add_argument("--trials", type=int, nargs="+", required=True, help="initial states whose episodes train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4, help="episodes per step")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2940)
    args = parser.parse_args()

    episodes = successful_episodes(args.dumps, set(args.trials))
    steps = write_steps(episodes, args.out, batch=args.batch, epochs=args.epochs, seed=args.seed)
    tasks: dict[str, int] = {}
    for sample in episodes:
        key = f"{sample['metadata']['suite']}/{sample['metadata']['task_id']}"
        tasks[key] = tasks.get(key, 0) + 1
    summary = {
        "episodes": len(episodes),
        "steps": steps,
        "tasks": tasks,
        "tokens": sum(len(s["tokens"]) for s in episodes),
        "trained_tokens": sum(sum(s["loss_mask"]) for s in episodes),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
