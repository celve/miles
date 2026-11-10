"""Paired comparisons of the eval arms by request, with a two-sided sign test, and the recipe's win bar.

  python -m examples.experimental.video_gen_agent.keyframe.eval.compare scored.jsonl... [--picks blind_result.json] [--i2v]

Compares K against B0, B1, K0 and K-blind on the training reward, the Qwen3-VL judge, PickScore, mean frame
difference, the OCR count and prompt length; a request either arm lacks a value for is left out of that metric.
The win bar (LIN-2795's recipe §5, the hub's proposal, not yet Linyu's): K beats B0 and B1 on the training reward
and on PickScore with p < 0.05, K's mean frame difference is not below B0's, and in the blind look Linyu prefers K
in at least 15 of 20 decided pairs against B0 (``--picks``, from ``blind.py unblind``). With ``--i2v`` (an agent
trained in i2v mode, the hub's revision) B0-i2v takes B0's place in every condition, since image-to-video fails the
frame-difference check against text-to-video by construction; B0 is still reported, as a reference.
"""

import argparse
import json
from math import comb
from pathlib import Path

METRICS = {
    "reward": lambda row: row["reward"],
    "judge": lambda row: row["scores"].get("judge"),
    "pickscore": lambda row: row["scores"].get("pickscore"),
    "frame_diff": lambda row: row["frame_diff"],
    "ocr_lines": lambda row: row["ocr_lines"],
    "prompt_words": lambda row: row["prompt_words"],
}
AGAINST = ("B0", "B1", "B0-i2v", "K0", "K-blind")


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)


def paired(ours: dict[str, float], theirs: dict[str, float]) -> dict:
    shared = sorted(r for r in ours.keys() & theirs.keys() if ours[r] is not None and theirs[r] is not None)
    diffs = [ours[r] - theirs[r] for r in shared]
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = len(shared)
    return {
        "requests": n,
        "ours_mean": sum(ours[r] for r in shared) / n if n else None,
        "theirs_mean": sum(theirs[r] for r in shared) / n if n else None,
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
        "p": sign_test(wins, losses),
    }


def compare(rows: list[dict]) -> dict[str, dict[str, dict]]:
    by_arm: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[row["request"]] = row
    results = {}
    for other in AGAINST:
        if other in by_arm and "K" in by_arm:
            results[other] = {
                name: paired(
                    {r: metric(row) for r, row in by_arm["K"].items()},
                    {r: metric(row) for r, row in by_arm[other].items()},
                )
                for name, metric in METRICS.items()
            }
    return results


def win_bar(results: dict, picks: dict | None, *, i2v: bool = False) -> dict[str, bool | None]:
    one_shot = one_shot_arm(i2v)

    def beats(other: str, metric: str) -> bool | None:
        result = results.get(other, {}).get(metric)
        return None if result is None else result["wins"] > result["losses"] and result["p"] < 0.05

    frame = results.get(one_shot, {}).get("frame_diff") or {}
    # No paired request has both arms' clips, so neither mean exists: the check is pending, not failed.
    frame_check = None if frame.get("ours_mean") is None else frame["ours_mean"] >= frame["theirs_mean"]
    return {
        f"reward beats {one_shot} and B1": _all(beats(one_shot, "reward"), beats("B1", "reward")),
        f"PickScore beats {one_shot} and B1": _all(beats(one_shot, "pickscore"), beats("B1", "pickscore")),
        f"blind: K in >= 15 of 20 decided vs {one_shot}": None if picks is None else picks["A"]["K"] >= 15,
        f"frame difference not below {one_shot}": frame_check,
    }


def one_shot_arm(i2v: bool) -> str:
    """The one-shot arm the win bar holds K to: image-to-video for an agent trained in i2v mode."""
    return "B0-i2v" if i2v else "B0"


def _all(*checks: bool | None) -> bool | None:
    return None if None in checks else all(checks)


def table(results: dict) -> str:
    lines = [
        "| K vs | metric | n | K mean | other mean | K wins | other wins | ties | sign p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for other, metrics in results.items():
        for name, r in metrics.items():
            means = [f"{v:.4f}" if v is not None else "–" for v in (r["ours_mean"], r["theirs_mean"])]
            lines.append(
                f"| {other} | {name} | {r['requests']} | {means[0]} | {means[1]} | {r['wins']} | {r['losses']} | {r['ties']} | {r['p']:.3g} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scored", type=Path, nargs="+")
    parser.add_argument("--picks", type=Path, help="blind.py unblind's result")
    parser.add_argument("--i2v", action="store_true", help="hold K to B0-i2v, for an agent trained in i2v mode")
    args = parser.parse_args(argv)
    rows = [json.loads(line) for path in args.scored for line in path.read_text().splitlines()]
    results = compare(rows)
    print(table(results))
    picks = json.loads(args.picks.read_text()) if args.picks else None
    for check, passed in win_bar(results, picks, i2v=args.i2v).items():
        print(f"- {check}: {'pending' if passed is None else 'pass' if passed else 'fail'}")


if __name__ == "__main__":
    main()
