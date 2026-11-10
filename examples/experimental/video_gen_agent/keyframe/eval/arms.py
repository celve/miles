"""Phase 1's eval arms (LIN-2795's recipe §5), as one ``clips.jsonl`` row per arm and held-out request:
``{arm, request, prompt, seed, path}``, with ``path`` None when the arm made no clip for the request.

  B0       the raw request → the video model
  B0-i2v   the raw request → the image model → image-to-video from that image (``prompts --mode i2v``), the one-shot
           arm an i2v-mode agent is fair against, since B0's text-to-video does not pay image-to-video's stillness
  B0-ref   the same image as the video model's reference (``prompts --mode reference``), for a reference-mode agent
  B1       Phase 0's trained planner prompt (LIN-2737's ``answer99``) → the video model, for VBench's requests only
  B1-rewrite  the untrained planner's one-turn rewrite of the request (``rewrite``: Phase 0's system prompt, no images,
           no thinking, greedy) → the video model; Phase 0's checkpoint is gone, so this stands in for B1 on new requests
  K0, K    the keyframe agent at its step-0 and final eval: the video request its episode submitted (prompt, negative
           prompt, and in i2v or reference mode the images), read from the run's eval dumps and generated again
  K-blind  the final agent's first image prompt → the video model, with no image watched; with ``--mode i2v``, the
           video starts from the first image the agent made, so K-blind differs from K only in watching. With
           ``--mode reference``, whose images are assets, it is an asset-prompt replay: its prompt is all the agent's
           image prompts joined and its references all its images in the order made, so it also differs from K in the
           composing prompt and in which assets, in which order, the video takes

Every arm takes Phase 0's per-request seed and the run's clip shape (832x480, 2 s at 16 fps: 33 frames), so arms
differ only in what the generator is asked; K0 and K keep the clip and seed their episode made as ``episode_path`` and
``episode_seed``. Generate every arm on one box, since clips differ across GPU types and parallel layouts. A held-out
row's ``metadata["category"]`` is copied onto its rows as ``category``, for ``compare.py --by-category``.

  python -m examples.experimental.video_gen_agent.keyframe.eval.arms rewrite --heldout heldout.jsonl \\
      --base-url http://localhost:30000 --out rewrites.jsonl
  python -m examples.experimental.video_gen_agent.keyframe.eval.arms prompts --heldout heldout.jsonl \\
      [--phase0 heldout_paired_step0_step99.jsonl] [--b1-rewrite rewrites.jsonl] --final-dump eval_199.pt --out prompts.jsonl
  python -m examples.experimental.video_gen_agent.keyframe.eval.arms from-dump eval_0.pt --arm K0 --heldout heldout.jsonl \
      --out k0.jsonl
  python -m examples.experimental.video_gen_agent.keyframe.eval.arms generate prompts.jsonl k0.jsonl k.jsonl --out clips.jsonl

``generate`` makes clips with the env's generator, so ``AGENT_VIDEO_GEN_GENERATOR`` and its options pick the router
(or a fake) exactly as in training; a row with an ``image_prompt`` first gets its image from
``AGENT_VIDEO_GEN_IMAGE_GENERATOR``. B0, B1 and B1-rewrite are text-to-video and the mode's one-shot arm is not, so generate them
in separate calls, each against its own video engines (with ``--mode i2v`` or ``reference``, K-blind goes with the
one-shot arm).
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
import torch

from examples.experimental.video_gen_agent.generator import ClipRequest, ImageRequest
from examples.experimental.video_gen_agent.keyframe.eval.compare import ONE_SHOT_ARMS
from examples.experimental.video_gen_agent.phase0.make_planner_data import SYSTEM_PROMPT
from examples.experimental.video_gen_agent.phase0.reward import planner_prompt, request_seed
from miles.utils.function_registry import load_function

SHAPE = {"seconds": 2.0, "width": 832, "height": 480, "fps": 16}


def heldout_requests(path: Path) -> list[str]:
    return [json.loads(line)["messages"][-1]["content"] for line in path.read_text().splitlines()]


def heldout_categories(path: Path) -> dict[str, str]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {
        row["messages"][-1]["content"]: row["metadata"]["category"]
        for row in rows
        if "category" in (row.get("metadata") or {})
    }


def episodes(dump: Path) -> dict[str, dict]:
    """Each request's ``video_gen`` record in an eval dump, from its episode's first sample."""
    data = torch.load(dump, weights_only=False)
    out = {}
    for sample in data["samples"] if isinstance(data, dict) else data:
        if (record := sample["metadata"].get("video_gen")) is not None:
            out.setdefault(record["request"], record)
    return out


def arm_prompts(
    requests: list[str],
    phase0: dict[str, str] | None,
    final: dict[str, dict],
    *,
    mode: str = "t2v",
    rewrites: dict[str, str] | None = None,
) -> list[dict]:
    """B0 and K-blind's prompt per request, B1's with Phase 0's answers and B1-rewrite's with the untrained rewrites,
    and in i2v or reference mode the mode's one-shot arm, with K-blind conditioned on the agent's first image as well;
    a request an arm has no prompt for gets prompt None."""
    rows = []
    for request in requests:
        images = (final.get(request) or {}).get("images") or []
        prompts = {"B0": request}
        for arm, answers in (("B1", phase0), ("B1-rewrite", rewrites)):
            if answers is not None:
                answer = answers.get(request)
                prompts[arm] = planner_prompt(answer) if answer is not None else None
        # Last, so the conditioning below lands on it.
        prompts["K-blind"] = _blind_prompt(mode, images)
        rows += [
            {"arm": arm, "request": request, "prompt": prompt, "seed": request_seed(request)}
            for arm, prompt in prompts.items()
        ]
        if mode == "t2v":
            continue
        if prompts["K-blind"] is not None:
            # i2v starts from the first image; reference mode takes every asset the agent made, which K's video may not.
            paths = [image["path"] for image in images] if mode == "reference" else [images[0]["path"]]
            rows[-1] |= _conditioned(mode, paths)
        rows.append(
            {"arm": ONE_SHOT_ARMS[mode], "request": request, "prompt": request, "image_prompt": request}
            | {"seed": request_seed(request), "mode": mode}
        )
    return rows


def _blind_prompt(mode: str, images: list[dict]) -> str | None:
    """The agent's first image prompt, or in reference mode, where each image is one asset, all of them."""
    if not images:
        return None
    return " ".join(image["prompt"] for image in images) if mode == "reference" else images[0]["prompt"]


def _conditioned(mode: str, image_paths: list[str]) -> dict:
    """A clip request's conditioning: the first image as the start frame in i2v mode, every image as a reference in
    reference mode."""
    return {"image_path": image_paths[0]} if mode == "i2v" else {"reference_paths": image_paths}


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
                row = row | _conditioned(row["mode"], [image.path])
            clip = await generator.generate(
                ClipRequest(
                    prompt=row["prompt"],
                    negative_prompt=row.get("negative_prompt") or "",
                    seed=row["seed"],
                    image_path=row.get("image_path"),
                    reference_paths=tuple(row.get("reference_paths") or ()),
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
        # Reference mode's images, which the replay needs as much as its prompt.
        references = {"reference_paths": video["reference_paths"]} if video.get("reference_paths") else {}
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
            | references
        )
    return rows


async def rewrite(requests: list[str], base_url: str, concurrency: int = 16) -> list[dict]:
    """B1-rewrite's answers: the served planner's one-turn rewrite of each request under Phase 0's system prompt, with
    thinking off as Phase 0 trained it, and greedy as eval decodes."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, request: str) -> dict:
        payload = {
            "model": "default",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}],
            "temperature": 0,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with semaphore:
            response = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return {"request": request, "answer": response.json()["choices"][0]["message"]["content"]}

    async with httpx.AsyncClient(timeout=600) as client:
        return list(await asyncio.gather(*(one(client, request) for request in requests)))


def _with_categories(rows: list[dict], heldout: Path) -> list[dict]:
    categories = heldout_categories(heldout)
    return [row | ({"category": categories[row["request"]]} if row["request"] in categories else {}) for row in rows]


def _answers(path: Path | None, key: str) -> dict[str, str] | None:
    return None if path is None else {r["request"]: r[key] for r in map(json.loads, path.read_text().splitlines())}


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prompts = commands.add_parser("prompts")
    prompts.add_argument("--heldout", type=Path, required=True)
    prompts.add_argument(
        "--phase0", type=Path, help="LIN-2737's heldout_paired_step0_step99.jsonl, for B1 on VBench's requests"
    )
    prompts.add_argument("--b1-rewrite", type=Path, help="rewrite's output, for B1-rewrite")
    prompts.add_argument("--final-dump", type=Path, required=True, help="the final eval's eval_<step>.pt")
    prompts.add_argument("--out", type=Path, required=True)
    prompts.add_argument(
        "--mode",
        choices=sorted(ONE_SHOT_ARMS),
        default="t2v",
        help="the agent's video mode: i2v or reference adds that mode's one-shot arm and conditions K-blind on an image",
    )
    generated = commands.add_parser("generate")
    generated.add_argument("prompts", type=Path, nargs="+")
    generated.add_argument("--out", type=Path, required=True)
    rewritten = commands.add_parser("rewrite")
    rewritten.add_argument("--heldout", type=Path, required=True)
    rewritten.add_argument("--base-url", required=True, help="an OpenAI-compatible server of the untrained planner")
    rewritten.add_argument("--out", type=Path, required=True)
    dumped = commands.add_parser("from-dump")
    dumped.add_argument("dump", type=Path)
    dumped.add_argument("--arm", choices=("K0", "K"), required=True)
    dumped.add_argument("--heldout", type=Path, required=True)
    dumped.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "prompts":
        rows = arm_prompts(
            heldout_requests(args.heldout),
            _answers(args.phase0, "answer99"),
            episodes(args.final_dump),
            mode=args.mode,
            rewrites=_answers(args.b1_rewrite, "answer"),
        )
        _write(args.out, _with_categories(rows, args.heldout))
    elif args.command == "rewrite":
        _write(args.out, asyncio.run(rewrite(heldout_requests(args.heldout), args.base_url)))
    elif args.command == "generate":
        generator = load_function(os.environ["AGENT_VIDEO_GEN_GENERATOR"])()
        image_path = os.environ.get("AGENT_VIDEO_GEN_IMAGE_GENERATOR")
        image_generator = load_function(image_path)() if image_path else None
        rows = [json.loads(line) for path in args.prompts for line in path.read_text().splitlines()]
        _write(args.out, asyncio.run(generate(rows, generator, image_generator)))
    else:
        _write(
            args.out, _with_categories(from_dump(args.dump, args.arm, heldout_requests(args.heldout)), args.heldout)
        )


if __name__ == "__main__":
    main()
