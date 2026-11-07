"""Turn a list of video requests into the env's prompt data, split into train and held-out.

The input has one request per line, such as VBench's ``prompts/all_dimension.txt``. Each request lands in
``train.jsonl`` or ``heldout.jsonl`` by a hash of its text, so a request keeps its split across runs, input
orderings and supersets. Each row is ``{"messages": [{"role": "user", "content": <request>}]}``: train with
``--input-key messages`` and no ``--apply-chat-template``; the env adds its own system prompt.

  python examples/experimental/video_gen_agent/make_requests.py prompts.txt /root/datasets/video_requests --heldout-percent 10
"""

import argparse
import hashlib
import json
from pathlib import Path


def is_heldout(request: str, heldout_percent: int) -> bool:
    return int(hashlib.sha256(request.encode()).hexdigest()[:8], 16) % 100 < heldout_percent


def split_requests(lines: list[str], heldout_percent: int) -> tuple[list[str], list[str]]:
    requests = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
    heldout = [r for r in requests if is_heldout(r, heldout_percent)]
    return [r for r in requests if not is_heldout(r, heldout_percent)], heldout


def _write(path: Path, requests: list[str]) -> None:
    with path.open("w") as f:
        for request in requests:
            f.write(json.dumps({"messages": [{"role": "user", "content": request}]}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("requests", type=Path, help="A text file with one video request per line.")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--heldout-percent", type=int, default=10)
    args = parser.parse_args()
    train, heldout = split_requests(args.requests.read_text().splitlines(), args.heldout_percent)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write(args.out_dir / "train.jsonl", train)
    _write(args.out_dir / "heldout.jsonl", heldout)
    print(f"{len(train)} train and {len(heldout)} held-out requests in {args.out_dir}")


if __name__ == "__main__":
    main()
