"""Phase 1's eval arms (LIN-2795's recipe §5), as one ``clips.jsonl`` row per arm and held-out request:
``{arm, request, prompt, seed, path}``, with ``path`` None when the arm made no clip for the request.

  B0       the raw request → the video model
  B0-i2v   the raw request → the image model → image-to-video from that image (``prompts --i2v``), the one-shot arm
           an i2v-mode agent is fair against, since B0's text-to-video does not pay image-to-video's stillness
  B1       Phase 0's trained planner prompt (LIN-2737's ``answer99``) → the video model
  K0, K    the keyframe agent at its step-0 and final eval: the video request its episode submitted (prompt, negative
           prompt, and in i2v mode the image), read from the run's eval dumps and generated again
  K-blind  the final agent's first image prompt → the video model, with no image watched

Every arm takes Phase 0's per-request seed and the run's clip shape (832x480, 2 s at 16 fps: 33 frames), so arms
differ only in what the generator is asked; K0 and K keep the clip and seed their episode made as ``episode_path`` and
``episode_seed``. Generate every arm on one box, since clips differ across GPU types and parallel layouts.

  python -m examples.experimental.video_gen_agent.keyframe.eval.arms prompts --heldout heldout.jsonl \\
      --phase0 heldout_paired_step0_step99.jsonl --final-dump eval_199.pt --out prompts.jsonl
  python -m examples.experimental.video_gen_agent.keyframe.eval.arms from-dump eval_0.pt --arm K0 --heldout heldout.jsonl \
      --out k0.jsonl
  python -m examples.experimental.video_gen_agent.keyframe.eval.arms generate prompts.jsonl k0.jsonl k.jsonl --out clips.jsonl

``generate`` makes clips with the env's generator, so ``AGENT_VIDEO_GEN_GENERATOR`` and its options pick the router
(or a fake) exactly as in training; a row with an ``image_prompt`` first gets its image from
``AGENT_VIDEO_GEN_IMAGE_GENERATOR``. B0 and B1 are text-to-video and B0-i2v image-to-video, so generate them in
separate calls, each against its own video engines.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import torch

from examples.experimental.video_gen_agent.generator import ClipRequest, ImageRequest
from examples.experimental.video_gen_agent.phase0.reward import planner_prompt, request_seed
from miles.utils.function_registry import load_function

GENERATED_ARMS = ("B0", "B1", "K-blind")
SHAPE = {"seconds": 2.0, "width": 832, "height": 480, "fps": 16}


def heldout_requests(path: Path) -> list[str]:
    return [json.loads(line)["messages"][-1]["content"] for line in path.read_text().splitlines()]


def episodes(dump: Path) -> dict[str, dict]:
    """Each request's ``video_gen`` record in an eval dump, from its episode's first sample."""
    data = torch.load(dump, weights_only=False)
    out = {}
    for sample in data["samples"] if isinstance(data, dict) else data:
        if (record := sample["metadata"].get("video_gen")) is not None:
            out.setdefault(record["request"], record)
    return out


def arm_prompts(
    requests: list[str], phase0: dict[str, str], final: dict[str, dict], *, i2v: bool = False
) -> list[dict]:
    """B0, B1 and K-blind's prompt per request, and B0-i2v's with ``i2v``; a request an arm has no prompt for gets
    prompt None."""
    rows = []
    for request in requests:
        images = (final.get(request) or {}).get("images") or []
        answer = phase0.get(request)
        prompts = {
            "B0": request,
            "B1": planner_prompt(answer) if answer is not None else None,
            "K-blind": images[0]["prompt"] if images else None,
        }
        rows += [
            {"arm": arm, "request": request, "prompt": prompts[arm], "seed": request_seed(request)}
            for arm in GENERATED_ARMS
        ]
        if i2v:
            rows.append(
                {"arm": "B0-i2v", "request": request, "prompt": request, "image_prompt": request}
                | {"seed": request_seed(request)}
            )
    return rows


async def generate(rows: list[dict], generator, image_generator=None, concurrency: int = 8) -> list[dict]:
    if image_generator is None and any(row.get("image_prompt") for row in rows):
        raise ValueError("rows with an image_prompt need AGENT_VIDEO_GEN_IMAGE_GENERATOR")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> dict:
        if row["prompt"] is None:
            return row | {"path": None}
        async with semaphore:
            if row.get("image_prompt"):
                image = await image_generator.generate(
                    ImageRequest(
                        prompt=row["image_prompt"],
                        negative_prompt="",
                        seed=row["seed"],
                        width=SHAPE["width"],
                        height=SHAPE["height"],
                    )
                )
                row = row | {"image_path": image.path}
            clip = await generator.generate(
                ClipRequest(
                    prompt=row["prompt"],
                    negative_prompt=row.get("negative_prompt") or "",
                    seed=row["seed"],
                    image_path=row.get("image_path"),
                    **SHAPE,
                )
            )
        return row | {"path": clip.path}

    return list(await asyncio.gather(*(one(row) for row in rows)))


def from_dump(dump: Path, arm: str, requests: list[str]) -> list[dict]:
    """The agent's video request per held-out request, to generate again on the common seed; an episode that
    submitted no video has prompt None."""
    recorded = episodes(dump)
    rows = []
    for request in requests:
        record = recorded.get(request) or {}
        video = record.get("video") or {}
        rows.append(
            {
                "arm": arm,
                "request": request,
                "prompt": video.get("prompt"),
                "negative_prompt": video.get("negative_prompt") or "",
                "seed": request_seed(request),
                "image_path": video.get("image_path"),
                "episode_path": (record.get("submitted") or {}).get("path"),
                "episode_seed": video.get("seed"),
            }
        )
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prompts = commands.add_parser("prompts")
    prompts.add_argument("--heldout", type=Path, required=True)
    prompts.add_argument("--phase0", type=Path, required=True, help="LIN-2737's heldout_paired_step0_step99.jsonl")
    prompts.add_argument("--final-dump", type=Path, required=True, help="the final eval's eval_<step>.pt")
    prompts.add_argument("--out", type=Path, required=True)
    prompts.add_argument("--i2v", action="store_true", help="also write B0-i2v, for an agent trained in i2v mode")
    generated = commands.add_parser("generate")
    generated.add_argument("prompts", type=Path, nargs="+")
    generated.add_argument("--out", type=Path, required=True)
    dumped = commands.add_parser("from-dump")
    dumped.add_argument("dump", type=Path)
    dumped.add_argument("--arm", choices=("K0", "K"), required=True)
    dumped.add_argument("--heldout", type=Path, required=True)
    dumped.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "prompts":
        phase0 = {r["request"]: r["answer99"] for r in map(json.loads, args.phase0.read_text().splitlines())}
        _write(args.out, arm_prompts(heldout_requests(args.heldout), phase0, episodes(args.final_dump), i2v=args.i2v))
    elif args.command == "generate":
        generator = load_function(os.environ["AGENT_VIDEO_GEN_GENERATOR"])()
        image_path = os.environ.get("AGENT_VIDEO_GEN_IMAGE_GENERATOR")
        image_generator = load_function(image_path)() if image_path else None
        rows = [json.loads(line) for path in args.prompts for line in path.read_text().splitlines()]
        _write(args.out, asyncio.run(generate(rows, generator, image_generator)))
    else:
        _write(args.out, from_dump(args.dump, args.arm, heldout_requests(args.heldout)))


if __name__ == "__main__":
    main()
