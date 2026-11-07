"""Turn a list of video requests into Phase 0's planner data: `train.jsonl`, `heldout.jsonl` and `heldout_raw.jsonl`.

The split is `make_requests.py`'s, so Phase 0 and the multi-turn env hold out the same requests. Each row is the
planner's chat, `{"messages": [system, user], "metadata": {"video_request": <request>}}`, for `--input-key messages
--apply-chat-template`. A `heldout_raw.jsonl` row also sets `send_request_to_generator`, so the reward step sends the
request itself to the generator: the raw-request baseline, on the same requests and seeds as `heldout.jsonl`.

  python -m examples.experimental.video_gen_agent.phase0.make_planner_data VBench/prompts/all_dimension.txt \
      /root/datasets/video_gen_phase0 --heldout-percent 10
"""

import argparse
import json
from pathlib import Path

from examples.experimental.video_gen_agent.make_requests import split_requests
from examples.experimental.video_gen_agent.phase0.reward import SEND_REQUEST_KEY

from miles.rollout.rm_hub.video.reward import REQUEST_KEY

SYSTEM_PROMPT = """You write prompts for a text-to-video model. The user asks for a short video; rewrite the request into one prompt for a single clip of about two seconds.

Keep everything the request asks for: the subjects, how many there are, their attributes and actions, the setting, the style and the camera movement. Add the concrete visual detail the model needs to get it right: appearance, lighting, composition and how things move. Do not ask for text, captions or watermarks in the video.

Answer with the prompt only, in English, in at most 120 words."""


def planner_row(request: str, *, send_request: bool = False) -> dict:
    metadata = {REQUEST_KEY: request} | ({SEND_REQUEST_KEY: True} if send_request else {})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]
    return {"messages": messages, "metadata": metadata}


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("requests", type=Path, help="A text file with one video request per line.")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--heldout-percent", type=int, default=10)
    args = parser.parse_args()
    train, heldout = split_requests(args.requests.read_text().splitlines(), args.heldout_percent)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write(args.out_dir / "train.jsonl", [planner_row(request) for request in train])
    _write(args.out_dir / "heldout.jsonl", [planner_row(request) for request in heldout])
    _write(args.out_dir / "heldout_raw.jsonl", [planner_row(request, send_request=True) for request in heldout])
    print(f"{len(train)} train and {len(heldout)} held-out requests in {args.out_dir}")


if __name__ == "__main__":
    main()
