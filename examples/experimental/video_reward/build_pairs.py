"""Build a calibration set of clip pairs for video reward scorers, from two generated clips per request.

usage: python build_pairs.py --requests requests.jsonl --clips <dir> --out <dir>

Each requests line is {"id", "request", "dimension"}; the clips directory holds `<id>_s0.mp4` and `<id>_s1.mp4`. The
output directory gets `pairs.jsonl` and `derived/`, the clips made for the reward-hacking pairs. Four kinds of pair:

- `seed`: a request's two seeds. Only a human label says which is better.
- `static`: a clip against its own first frame held for the clip's length, the hack of a scene that does not move.
- `overlay`: a clip against another request's clip with this request's text drawn over it, the hack of a judge that
  reads text instead of watching.
- `mismatch`: a clip against another request's clip, a plain text-alignment check.

Every pair but `seed` has a known answer, `expected`: the honest clip. A and B are shuffled with a fixed seed.
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Requests whose subject moves, so a frozen clip is a worse answer to them.
_MOVING_DIMENSIONS = ("human_action", "subject_consistency", "temporal_style", "overall_consistency")
_KINDS = ("seed", "static", "overlay", "mismatch")
_NUM_OVERLAY = 40
_NUM_MISMATCH = 24


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2736)
    parser.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    args = parser.parse_args()

    requests = [json.loads(line) for line in args.requests.read_text().splitlines()]
    rng = random.Random(args.seed)
    derived = args.out / "derived"
    derived.mkdir(parents=True, exist_ok=True)

    pairs = [_pair("seed", item, f"{item['id']}_s0.mp4", f"{item['id']}_s1.mp4", honest=None) for item in requests]
    for item in requests:
        if item["dimension"] in _MOVING_DIMENSIONS:
            static = f"derived/{item['id']}_static.mp4"
            _write_static(args.clips / f"{item['id']}_s0.mp4", args.out / static)
            pairs.append(_pair("static", item, f"{item['id']}_s0.mp4", static, honest=0))
    for item, other in _pair_up(requests, count=_NUM_OVERLAY, rng=rng):
        overlay = f"derived/{item['id']}_over_{other['id']}.mp4"
        _write_overlay(args.clips / f"{other['id']}_s1.mp4", args.out / overlay, text=item["request"], font=args.font)
        pairs.append(_pair("overlay", item, f"{item['id']}_s0.mp4", overlay, honest=0))
    for item, other in _pair_up(requests, count=_NUM_MISMATCH, rng=rng):
        pairs.append(_pair("mismatch", item, f"{item['id']}_s1.mp4", f"{other['id']}_s0.mp4", honest=0))

    with (args.out / "pairs.jsonl").open("w") as file:
        for index, pair in enumerate(pairs):
            if rng.random() < 0.5:
                pair["a"], pair["b"] = pair["b"], pair["a"]
                pair["expected"] = {"A": "B", "B": "A", None: None}[pair["expected"]]
            file.write(json.dumps({"pair_id": f"p{index:03d}", **pair}) + "\n")
    print(f"{len(pairs)} pairs: " + ", ".join(f"{kind} {sum(p['kind'] == kind for p in pairs)}" for kind in _KINDS))


def _pair(kind: str, item: dict, a: str, b: str, *, honest: int | None) -> dict:
    """Clip paths are relative to the clips directory, or to the output directory under `derived/`."""
    return dict(
        kind=kind,
        request_id=item["id"],
        request=item["request"],
        a=a,
        b=b,
        expected=None if honest is None else "AB"[honest],
    )


def _pair_up(requests: list[dict], *, count: int, rng: random.Random) -> list[tuple[dict, dict]]:
    """`count` requests, each with another request of a different dimension."""
    chosen = rng.sample(requests, count)
    return [
        (item, rng.choice([other for other in requests if other["dimension"] != item["dimension"]])) for item in chosen
    ]


def _read(path: Path) -> tuple[list[np.ndarray], float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)], float(stream.average_rate)


def _write(path: Path, frames: list[np.ndarray], fps: float) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=round(fps))
        stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        stream.options = {"crf": "18"}
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


def _write_static(source: Path, target: Path) -> None:
    frames, fps = _read(source)
    _write(target, [frames[0]] * len(frames), fps)


def _write_overlay(source: Path, target: Path, *, text: str, font: str) -> None:
    frames, fps = _read(source)
    height, width = frames[0].shape[:2]
    font = ImageFont.truetype(font, size=max(16, height // 16))
    lines = textwrap.wrap(text, width=max(20, width // (font.size * 3 // 5)))
    overlaid = []
    for frame in frames:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        y = height // 12
        for line in lines:
            box = draw.textbbox((0, 0), line, font=font)
            x = (width - (box[2] - box[0])) // 2
            draw.text((x, y), line, font=font, fill="white", stroke_width=3, stroke_fill="black")
            y += (box[3] - box[1]) * 5 // 4
        overlaid.append(np.asarray(image))
    _write(target, overlaid, fps)


if __name__ == "__main__":
    main()
