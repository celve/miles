"""Paired comparisons of the eval arms by request, with a two-sided sign test, and the recipe's win bar.

  python -m examples.experimental.video_gen_agent.keyframe.eval.compare scored.jsonl... [--picks blind_result.json] \\
      [--mode t2v|i2v|reference] [--b1 B1|B1-rewrite] [--by-category]

Compares K against B0, B1, B1-rewrite, K0 and K-blind on the training reward, the Qwen3-VL judge, PickScore, mean frame
difference, the OCR count and prompt length; a request either arm lacks a value for is left out of that metric.
The win bar (LIN-2795's recipe §5, the hub's proposal, not yet Linyu's): K beats B0 and B1 on the training reward
and on PickScore with p < 0.05, K's mean frame difference is not below B0's, and in the blind look Linyu prefers K
in at least 15 of 20 decided pairs against B0 (``--picks``, from ``blind.py unblind``). With ``--mode`` (the video
mode the agent trained in, the hub's revision) the one-shot arm of that mode, B0-i2v or B0-ref, takes B0's place in
every condition, since an image-conditioned video is not fairly held to text-to-video; B0 is still reported, as a
reference. ``--b1 B1-rewrite`` holds K to the untrained rewrite where Phase 0's trained B1 does not exist (LIN-2834).
``--by-category`` adds the comparisons within each request category.
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
# The one-shot arm the win bar holds K to, by the video mode the agent trained in.
ONE_SHOT_ARMS = {"t2v": "B0", "i2v": "B0-i2v", "reference": "B0-ref"}
AGAINST = ("B0", "B1", "B1-rewrite", "B0-i2v", "B0-ref", "K0", "K-blind")
B1_ARMS = ("B1", "B1-rewrite")


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


def win_bar(results: dict, picks: dict | None, *, mode: str = "t2v", b1: str = "B1") -> dict[str, bool | None]:
    one_shot = ONE_SHOT_ARMS[mode]
    assert b1 in B1_ARMS, f"b1 is one of {B1_ARMS}"

    def beats(other: str, metric: str) -> bool | None:
        result = results.get(other, {}).get(metric)
        return None if result is None else result["wins"] > result["losses"] and result["p"] < 0.05

    if picks is not None and one_shot not in picks["A"]:
        raise ValueError(
            f"the blind look's Set A was not judged against {one_shot}: {picks['A']}; rerun blind.py with "
            f"the same --mode as compare.py"
        )
    frame = results.get(one_shot, {}).get("frame_diff") or {}
    # No paired request has both arms' clips, so neither mean exists: the check is pending, not failed.
    frame_check = None if frame.get("ours_mean") is None else frame["ours_mean"] >= frame["theirs_mean"]
    return {
        f"reward beats {one_shot} and {b1}": _all(beats(one_shot, "reward"), beats(b1, "reward")),
        f"PickScore beats {one_shot} and {b1}": _all(beats(one_shot, "pickscore"), beats(b1, "pickscore")),
        f"blind: K in >= 15 of 20 decided vs {one_shot}": None if picks is None else picks["A"]["K"] >= 15,
        f"frame difference not below {one_shot}": frame_check,
    }


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
    parser.add_argument(
        "--mode",
        choices=sorted(ONE_SHOT_ARMS),
        default="t2v",
        help="the agent's video mode, which picks its one-shot arm",
    )
    parser.add_argument("--b1", choices=B1_ARMS, default="B1", help="the prompt-rewriting arm the win bar holds K to")
    parser.add_argument("--by-category", action="store_true", help="also compare within each request category")
    args = parser.parse_args(argv)
    rows = [json.loads(line) for path in args.scored for line in path.read_text().splitlines()]
    results = compare(rows)
    print(table(results))
    picks = json.loads(args.picks.read_text()) if args.picks else None
    for check, passed in win_bar(results, picks, mode=args.mode, b1=args.b1).items():
        print(f"- {check}: {'pending' if passed is None else 'pass' if passed else 'fail'}")
    if args.by_category:
        for category in sorted({row["category"] for row in rows if row.get("category")}):
            print(f"\n## {category}\n\n{table(compare([row for row in rows if row.get('category') == category]))}")


if __name__ == "__main__":
    main()
