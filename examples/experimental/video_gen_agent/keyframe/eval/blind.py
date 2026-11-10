"""The phase 1 blind look, after LIN-2741's ``blind_pairs.py``: 20 held-out requests drawn with a fixed seed, outside
LIN-2741's blind 20, in two sets on the same requests. Set A is K against B0 (B0-i2v with ``--i2v``) and Set B is K
against B1; each pair's sides are shuffled, and the key goes to a separate file to apply only once the picks are saved.

  python -m examples.experimental.video_gen_agent.keyframe.eval.blind make clips.jsonl... --exclude lin2741_pairs.json \\
      --out blind/ [--i2v]
  python -m examples.experimental.video_gen_agent.keyframe.eval.blind unblind blind/pairs.json blind_key.json picks.json

``make`` writes the page, ``<out>/index.html`` (``blind_page.html``, LIN-2741's page), with ``<out>/pairs.json`` and the
clips as ``<out>/<pair>_<left|right>.mp4``, and the key beside ``<out>`` as ``<out>_key.json``, which the page never
reads. Publish ``<out>`` as an Artifact with a database, or open it locally; the page saves each pick as it is made and
downloads them as ``picks.json``, which maps a pair to ``left``, ``right`` or ``tie``.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from examples.experimental.video_gen_agent.keyframe.eval.compare import one_shot_arm

_PAGE = Path(__file__).with_name("blind_page.html")
SEED = 2821
PAIRS = 20


def sets(i2v: bool = False) -> dict[str, str]:
    return {"A": one_shot_arm(i2v), "B": "B1"}


def draw(clips: dict[str, dict[str, str]], exclude: set[str], *, i2v: bool = False) -> list[str]:
    """The drawn requests: ones K and both baselines have clips for, outside ``exclude``, in a fixed order."""
    arms = ["K", *sets(i2v).values()]
    eligible = sorted(r for r in set.intersection(*(set(clips[arm]) for arm in arms)) if r not in exclude)
    assert (
        len(eligible) >= PAIRS
    ), f"only {len(eligible)} requests have {', '.join(arms)} clips outside the excluded ones"
    return random.Random(SEED).sample(eligible, PAIRS)


def make(
    clips: dict[str, dict[str, str]], exclude: set[str], out: Path, *, i2v: bool = False
) -> tuple[list[dict], dict[str, str]]:
    requests = draw(clips, exclude, i2v=i2v)
    rng = random.Random(SEED + 1)
    out.mkdir(parents=True, exist_ok=True)
    pairs, key = [], {}
    for set_name, baseline in sets(i2v).items():
        for index, request in enumerate(requests, 1):
            pair = f"{set_name}{index:02d}"
            k_left = rng.random() < 0.5
            sides = (clips["K"][request], clips[baseline][request])
            for side, path in zip(("left", "right"), sides if k_left else sides[::-1], strict=True):
                shutil.copyfile(path, out / f"{pair}_{side}.mp4")
            pairs.append({"pair": pair, "set": set_name, "baseline": baseline, "request": request})
            key[pair] = "left" if k_left else "right"
    (out / "pairs.json").write_text(json.dumps(pairs, indent=1))
    (out / "index.html").write_text(_PAGE.read_text().replace("__PAIRS__", json.dumps(pairs)))
    (out.parent / f"{out.name}_key.json").write_text(json.dumps(key, indent=1))
    return pairs, key


def unblind(pairs: list[dict], key: dict[str, str], picks: dict[str, str]) -> dict:
    """Per set, how often Linyu picked K, the baseline, or neither."""
    result: dict[str, dict[str, int]] = {}
    for pair in pairs:
        baseline = pair.get("baseline") or sets()[pair["set"]]
        choice = picks[pair["pair"]]
        winner = "tie" if choice == "tie" else "K" if choice == key[pair["pair"]] else baseline
        counts = result.setdefault(pair["set"], {"K": 0, baseline: 0, "tie": 0})
        counts[winner] += 1
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
    made.add_argument("--i2v", action="store_true", help="Set A against B0-i2v, for an agent trained in i2v mode")
    unblinded = commands.add_parser("unblind")
    for name in ("pairs", "key", "picks"):
        unblinded.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    if args.command == "make":
        exclude = {pair["request"] for pair in json.loads(args.exclude.read_text())}
        pairs, _ = make(_clips(args.clips), exclude, args.out, i2v=args.i2v)
        print(len(pairs), "pairs in", args.out)
    else:
        load = lambda path: json.loads(path.read_text())  # noqa: E731
        print(json.dumps(unblind(load(args.pairs), load(args.key), load(args.picks)), indent=1))


if __name__ == "__main__":
    main()
