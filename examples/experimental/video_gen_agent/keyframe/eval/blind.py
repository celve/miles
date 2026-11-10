"""The phase 1 blind look, after LIN-2741's ``blind_pairs.py``: 20 held-out requests drawn with a fixed seed, outside
LIN-2741's blind 20, in two sets on the same requests. Set A is K against the one-shot arm of ``--mode`` (B0, B0-i2v or B0-ref) and Set B is K
against B1; each pair's sides are shuffled, and the key goes to a separate file to apply only once the picks are saved.

  python -m examples.experimental.video_gen_agent.keyframe.eval.blind make clips.jsonl... --exclude lin2741_pairs.json \\
      --out blind/ [--mode t2v|i2v|reference]
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

from examples.experimental.video_gen_agent.keyframe.eval.compare import ONE_SHOT_ARMS

_PAGE = Path(__file__).with_name("blind_page.html")
SEED = 2821
_MODELS = {
    "t2v": "by the same frozen Wan2.2-T2V-A14B with the lightx2v LoRAs.",
    "i2v": (
        "by frozen Wan2.2-A14B with the lightx2v LoRAs: in set A both clips are image-to-video from a generated "
        "image; in set B one is image-to-video and the other text-to-video."
    ),
    "reference": (
        "by frozen video models with lightx2v LoRAs: in set A both clips are made by Wan2.1-VACE-14B with a generated "
        "image as reference; in set B one is made that way and the other text-to-video by Wan2.2-T2V-A14B."
    ),
}
PAIRS = 20


def sets(mode: str = "t2v") -> dict[str, str]:
    return {"A": ONE_SHOT_ARMS[mode], "B": "B1"}


def draw(clips: dict[str, dict[str, str]], exclude: set[str], *, mode: str = "t2v") -> list[str]:
    """The drawn requests: ones K and both baselines have clips for, outside ``exclude``, in a fixed order."""
    arms = ["K", *sets(mode).values()]
    eligible = sorted(r for r in set.intersection(*(set(clips[arm]) for arm in arms)) if r not in exclude)
    assert (
        len(eligible) >= PAIRS
    ), f"only {len(eligible)} requests have {', '.join(arms)} clips outside the excluded ones"
    return random.Random(SEED).sample(eligible, PAIRS)


def make(
    clips: dict[str, dict[str, str]], exclude: set[str], out: Path, *, mode: str = "t2v"
) -> tuple[list[dict], dict[str, str]]:
    requests = draw(clips, exclude, mode=mode)
    rng = random.Random(SEED + 1)
    out.mkdir(parents=True, exist_ok=True)
    pairs, key = [], {}
    for set_name, baseline in sets(mode).items():
        for index, request in enumerate(requests, 1):
            pair = f"{set_name}{index:02d}"
            k_left = rng.random() < 0.5
            sides = (clips["K"][request], clips[baseline][request])
            for side, path in zip(("left", "right"), sides if k_left else sides[::-1], strict=True):
                shutil.copyfile(path, out / f"{pair}_{side}.mp4")
            pairs.append({"pair": pair, "set": set_name, "baseline": baseline, "request": request})
            key[pair] = "left" if k_left else "right"
    (out / "pairs.json").write_text(json.dumps(pairs, indent=1))
    page = _PAGE.read_text().replace("__MODELS__", _MODELS[mode]).replace("__PAIRS__", json.dumps(pairs))
    (out / "index.html").write_text(page)
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
    made.add_argument(
        "--mode", choices=sorted(ONE_SHOT_ARMS), default="t2v", help="the agent's video mode; Set A's arm"
    )
    unblinded = commands.add_parser("unblind")
    for name in ("pairs", "key", "picks"):
        unblinded.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    if args.command == "make":
        exclude = {pair["request"] for pair in json.loads(args.exclude.read_text())}
        pairs, _ = make(_clips(args.clips), exclude, args.out, mode=args.mode)
        print(len(pairs), "pairs in", args.out)
    else:
        load = lambda path: json.loads(path.read_text())  # noqa: E731
        print(json.dumps(unblind(load(args.pairs), load(args.key), load(args.picks)), indent=1))


if __name__ == "__main__":
    main()
