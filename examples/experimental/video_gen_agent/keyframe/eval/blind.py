"""The phase 1 blind look, after LIN-2741's ``blind_pairs.py``: 20 held-out requests drawn with a fixed seed, outside
LIN-2741's blind 20, in two sets on the same requests. Set A is K against B0 and Set B is K against B1; each pair's
sides are shuffled, and the key goes to a separate file to apply only once the picks are saved.

  python -m examples.experimental.video_gen_agent.keyframe.eval.blind make clips.jsonl... --exclude lin2741_pairs.json \\
      --out blind/
  python -m examples.experimental.video_gen_agent.keyframe.eval.blind unblind blind/pairs.json blind_key.json picks.json

``make`` writes ``<out>/pairs.json`` and the clips as ``<out>/<pair>_<left|right>.mp4``, and the key beside ``<out>``
as ``<out>_key.json``. ``picks.json`` maps a pair to ``left``, ``right`` or ``tie``.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

SEED = 2821
PAIRS = 20
SETS = {"A": "B0", "B": "B1"}


def draw(clips: dict[str, dict[str, str]], exclude: set[str]) -> list[str]:
    """The drawn requests: ones K and both baselines have clips for, outside ``exclude``, in a fixed order."""
    eligible = sorted(r for r in clips["K"].keys() & clips["B0"].keys() & clips["B1"].keys() if r not in exclude)
    assert len(eligible) >= PAIRS, f"only {len(eligible)} requests have K, B0 and B1 clips outside the excluded ones"
    return random.Random(SEED).sample(eligible, PAIRS)


def make(clips: dict[str, dict[str, str]], exclude: set[str], out: Path) -> tuple[list[dict], dict[str, str]]:
    requests = draw(clips, exclude)
    rng = random.Random(SEED + 1)
    out.mkdir(parents=True, exist_ok=True)
    pairs, key = [], {}
    for set_name, baseline in SETS.items():
        for index, request in enumerate(requests, 1):
            pair = f"{set_name}{index:02d}"
            k_left = rng.random() < 0.5
            sides = (clips["K"][request], clips[baseline][request])
            for side, path in zip(("left", "right"), sides if k_left else sides[::-1], strict=True):
                shutil.copyfile(path, out / f"{pair}_{side}.mp4")
            pairs.append({"pair": pair, "set": set_name, "request": request})
            key[pair] = "left" if k_left else "right"
    (out / "pairs.json").write_text(json.dumps(pairs, indent=1))
    (out.parent / f"{out.name}_key.json").write_text(json.dumps(key, indent=1))
    return pairs, key


def unblind(pairs: list[dict], key: dict[str, str], picks: dict[str, str]) -> dict:
    """Per set, how often Linyu picked K, the baseline, or neither."""
    result = {name: {"K": 0, baseline: 0, "tie": 0} for name, baseline in SETS.items()}
    for pair in pairs:
        choice = picks[pair["pair"]]
        winner = "tie" if choice == "tie" else "K" if choice == key[pair["pair"]] else SETS[pair["set"]]
        result[pair["set"]][winner] += 1
    return result


def _clips(paths: list[Path]) -> dict[str, dict[str, str]]:
    clips: dict[str, dict[str, str]] = {}
    for row in (json.loads(line) for path in paths for line in path.read_text().splitlines()):
        if row["path"] is not None:
            clips.setdefault(row["arm"], {})[row["request"]] = row["path"]
    return clips


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("make")
    made.add_argument("clips", type=Path, nargs="+")
    made.add_argument("--exclude", type=Path, required=True, help="LIN-2741's blind pairs.json")
    made.add_argument("--out", type=Path, required=True)
    unblinded = commands.add_parser("unblind")
    for name in ("pairs", "key", "picks"):
        unblinded.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    if args.command == "make":
        exclude = {pair["request"] for pair in json.loads(args.exclude.read_text())}
        pairs, _ = make(_clips(args.clips), exclude, args.out)
        print(len(pairs), "pairs in", args.out)
    else:
        load = lambda path: json.loads(path.read_text())  # noqa: E731
        print(json.dumps(unblind(load(args.pairs), load(args.key), load(args.picks)), indent=1))


if __name__ == "__main__":
    main()
