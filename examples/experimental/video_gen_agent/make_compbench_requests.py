"""Turn T2V-CompBench's prompt suite into the env's prompt data, with a held-out set stratified by category.

T2V-CompBench (Sun et al., CVPR 2025) has 1,400 compositional prompts, 200 in each of seven categories, in
``prompts/*.txt``, with each prompt's objects in ``meta_data/*.json``. Its repository carries no licence, so the prompts
are fetched from it at prepare time and never copied into this one. Rows are ``make_requests.py``'s, plus metadata:

  {"messages": [{"role": "user", "content": <request>}],
   "metadata": {"source": "t2v-compbench", "category": <category>, "id": "<category>/<line>", "subjects": [...]}}

``subjects`` lists the objects T2V-CompBench names for the prompt (empty for object interactions and dynamic
attributes, which name none), for eval to check each one's presence; the env does not read it. Each category gives the
held-out set its share of ``--heldout`` (largest remainder), taking the prompts that rank first by a hash of
``--seed`` and the prompt, so the split depends on neither file order nor platform.

  python examples/experimental/video_gen_agent/make_compbench_requests.py /root/datasets/T2V-CompBench \\
      /root/datasets/video_gen_requests_t2v_compbench --heldout 200
"""

import argparse
import hashlib
import json
from pathlib import Path

SOURCE = "t2v-compbench"
# Each category's prompt file and metadata file, in T2V-CompBench's order.
CATEGORIES = {
    "consistent_attribute_binding": ("1_consistent_attr", "consistent_attribute_binding"),
    "dynamic_attribute_binding": ("2_dynamic_attr", "dynamic_attribute_binding"),
    "spatial_relationships": ("3_spatial_relationship", "spatial_relationships"),
    "motion_binding": ("4_motion_binding", "motion_binding"),
    "action_binding": ("5_action_binding", "action_binding"),
    "object_interactions": ("6_interaction", "object_interactions"),
    "generative_numeracy": ("7_numeracy", "generative_numeracy"),
}


def subjects(category: str, meta: dict) -> list[str]:
    if category == "consistent_attribute_binding":
        names = meta["phrases"].split(";")
    elif category in ("spatial_relationships", "motion_binding"):
        names = [meta["object_1"], meta["object_2"]]
    elif category == "action_binding":
        names = [meta["phrase_0"][0], meta["phrase_1"][0]]
    elif category == "generative_numeracy":
        names = meta["objects"].split(",")
    else:
        names = []
    return [name.strip().rstrip("?").strip() for name in names if name.strip()]


def load(root: Path) -> list[dict]:
    """Every distinct prompt with its category, id and subjects; a repeated prompt keeps its first occurrence."""
    rows, seen = [], set()
    for category, (prompt_file, meta_file) in CATEGORIES.items():
        prompts = [line.strip() for line in (root / "prompts" / f"{prompt_file}.txt").read_text().splitlines()]
        prompts = [p for p in prompts if p]
        metas = json.loads((root / "meta_data" / f"{meta_file}.json").read_text())
        # The metadata repeats each prompt, sometimes with other punctuation, in the same order.
        assert len(metas) == len(prompts), f"{meta_file}.json has {len(metas)} rows for {len(prompts)} prompts"
        for line, (prompt, meta) in enumerate(zip(prompts, metas, strict=True)):
            if prompt in seen:
                continue
            seen.add(prompt)
            rows.append(
                {"request": prompt, "category": category, "id": f"{category}/{line:03d}"}
                | {"subjects": subjects(category, meta)}
            )
    return rows


def split(rows: list[dict], heldout: int, seed: int) -> tuple[list[dict], list[dict]]:
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    exact = {c: heldout * len(members) / len(rows) for c, members in by_category.items()}
    quota = {c: int(share) for c, share in exact.items()}
    for c in sorted(exact, key=lambda c: (quota[c] - exact[c], c))[: heldout - sum(quota.values())]:
        quota[c] += 1
    held = set()
    for category, members in by_category.items():
        ranked = sorted(members, key=lambda row: hashlib.sha256(f"{seed}:{row['request']}".encode()).hexdigest())
        held |= {row["id"] for row in ranked[: quota[category]]}
    return [row for row in rows if row["id"] not in held], [row for row in rows if row["id"] in held]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            metadata = {"source": SOURCE} | {key: row[key] for key in ("category", "id", "subjects")}
            f.write(
                json.dumps({"messages": [{"role": "user", "content": row["request"]}], "metadata": metadata}) + "\n"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("compbench_dir", type=Path, help="a checkout of github.com/KaiyueSun98/T2V-CompBench")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--heldout", type=int, default=200, help="held-out requests, spread over the categories")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = load(args.compbench_dir)
    assert 0 <= args.heldout <= len(rows), f"--heldout must be between 0 and {len(rows)}"
    train, heldout = split(rows, args.heldout, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write(args.out_dir / "train.jsonl", train)
    _write(args.out_dir / "heldout.jsonl", heldout)
    print(f"{len(train)} train and {len(heldout)} held-out requests in {args.out_dir}")


if __name__ == "__main__":
    main()
