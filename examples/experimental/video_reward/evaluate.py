"""Score a calibration set with each scorer of a reward pool config, and report agreement and latency per scorer.

usage: python evaluate.py --pools pools.yaml --pairs <out>/pairs.jsonl --clips <dir> [--labels labels.json]
           [--scores scores.json] [--device cuda:0]

`pools.yaml` is the file --reward-pool-config takes; each pool's scorer is loaded on `--device` and fed its pool's
`batch_size` clips per `score_batch` call, as a reward worker would. Scores are cached per scorer and clip in
`--scores`, so a rerun with new labels scores nothing again. `labels.json` maps a pair id to A, B, tie or bad.

Agreement counts a pair when the scorer gives the clip the label (or the pair's known answer) prefers the higher
score; a pair either clip of which scored None counts as missed.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from miles.utils.function_registry import load_function

_KINDS = ("seed", "static", "overlay", "mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--scores", type=Path, default=Path("scores.json"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines()]
    labels = json.loads(args.labels.read_text()) if args.labels else {}
    cache = json.loads(args.scores.read_text()) if args.scores.exists() else {}
    # A clip is scored against the request of each pair it is in; another request's clip, in a mismatch pair, too.
    items = sorted({(pair["request_id"], pair["request"], pair[side]) for pair in pairs for side in ("a", "b")})
    for pool in yaml.safe_load(args.pools.read_text()):
        if pool["name"] not in cache:
            cache[pool["name"]] = _score(pool, items, root=args.pairs.parent, clip_dir=args.clips, device=args.device)
            args.scores.write_text(json.dumps(cache, indent=1))
    print(_report(pairs, labels, cache))


def _score(pool: dict, items: list[tuple[str, str, str]], *, root: Path, clip_dir: Path, device: str) -> dict:
    """Each (request, clip) item's score keyed `<request id>|<clip>`, the scorer's load time, and each batch's seconds."""
    start = time.perf_counter()
    scorer = load_function(pool["scorer_path"])(device=torch.device(device), **pool.get("scorer_kwargs", {}))
    load_seconds = time.perf_counter() - start
    scores, batch_seconds = {}, []
    for begin in range(0, len(items), pool["batch_size"]):
        batch = items[begin : begin + pool["batch_size"]]
        paths = [str(root / clip if clip.startswith("derived/") else clip_dir / clip) for _, _, clip in batch]
        start = time.perf_counter()
        batch_scores = scorer.score_batch(paths, [request for _, request, _ in batch])
        torch.accelerator.synchronize()
        batch_seconds.append((time.perf_counter() - start, len(batch)))
        for (request_id, _, clip), score in zip(batch, batch_scores, strict=True):
            scores[_key(request_id, clip)] = score
    del scorer
    torch.accelerator.empty_cache()
    return dict(scores=scores, load_seconds=load_seconds, batch_seconds=batch_seconds)


def _key(request_id: str, clip: str) -> str:
    return f"{request_id}|{clip}"


def _report(pairs: list[dict], labels: dict, cache: dict) -> str:
    known_kinds = " | ".join(f"{kind} (known)" for kind in _KINDS[1:])
    rows = [f"| scorer | human agreement | {known_kinds} | s/clip p50 | s/clip p90 | load s |"]
    rows.append("|" + " --- |" * (len(_KINDS) + 5))
    for name, result in cache.items():
        scores = result["scores"]
        human = [(pair, labels[pair["pair_id"]]) for pair in pairs if labels.get(pair["pair_id"]) in ("A", "B")]
        cells = [_agreement(human, scores)]
        for kind in _KINDS[1:]:
            known = [(pair, pair["expected"]) for pair in pairs if pair["kind"] == kind]
            cells.append(_agreement(known, scores))
        per_clip = sorted(seconds / count for seconds, count in result["batch_seconds"][1:] or result["batch_seconds"])
        p50, p90 = statistics.median(per_clip), per_clip[int(0.9 * (len(per_clip) - 1))]
        rows.append(f"| {name} | " + " | ".join(cells) + f" | {p50:.3f} | {p90:.3f} | {result['load_seconds']:.0f} |")
    return "\n".join(rows)


def _agreement(judged: list[tuple[dict, str]], scores: dict) -> str:
    if not judged:
        return "–"
    agree = 0
    for pair, winner in judged:
        a, b = scores.get(_key(pair["request_id"], pair["a"])), scores.get(_key(pair["request_id"], pair["b"]))
        if a is not None and b is not None and a != b:
            agree += (a > b) == (winner == "A")
    return f"{agree / len(judged):.2f} ({agree}/{len(judged)})"


if __name__ == "__main__":
    main()
