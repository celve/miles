import json
import subprocess
import sys
from collections import Counter

from examples.experimental.video_gen_agent.make_compbench_requests import CATEGORIES

SCRIPT = "examples/experimental/video_gen_agent/make_compbench_requests.py"


def _meta(category: str, i: int) -> dict:
    return {
        "consistent_attribute_binding": {"phrases": f"a blue car {i}; a white fence {i}"},
        "dynamic_attribute_binding": {"state 0": "green", "state 1": "red", "tag": "color"},
        "spatial_relationships": {"spatial": "left", "object_1": f"dog {i}", "object_2": "bicycle"},
        "motion_binding": {"object_1": f"cat {i}", "d_1": "left", "object_2": "", "d_2": ""},
        "action_binding": {"phrase_0": [f"a dog {i}?", "a dog runs?"], "phrase_1": ["a cat?", "a cat climbs?"]},
        "object_interactions": {},
        "generative_numeracy": {"objects": f"tree {i}, bee", "numbers": "1, 1"},
    }[category]


def _compbench(root, sizes: dict[str, int], order=lambda lines: lines):
    """A T2V-CompBench checkout with ``sizes[category]`` prompts per category; the metadata repeats each prompt with a
    trailing period, as the real one sometimes does, and motion binding repeats its first prompt."""
    (root / "prompts").mkdir(parents=True)
    (root / "meta_data").mkdir()
    for category, (prompt_file, meta_file) in CATEGORIES.items():
        prompts = [f"{category} prompt {i}" for i in range(sizes[category])]
        if category == "motion_binding":
            prompts.append(prompts[0])
        pairs = order(list(zip(prompts, [_meta(category, i) for i in range(len(prompts))], strict=True)))
        (root / "prompts" / f"{prompt_file}.txt").write_text("\n".join(p for p, _ in pairs) + "\n\n")
        metas = [{"prompt": p + "."} | meta for p, meta in pairs]
        (root / "meta_data" / f"{meta_file}.json").write_text(json.dumps(metas))
    return root


def _run(compbench, out, *args):
    subprocess.run([sys.executable, SCRIPT, str(compbench), str(out), *args], check=True)
    return [
        [json.loads(line) for line in (out / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "heldout")
    ]


SIZES = dict.fromkeys(CATEGORIES, 20) | {"consistent_attribute_binding": 40}


def test_the_held_out_set_is_stratified_by_category_and_disjoint_from_train(tmp_path):
    train, heldout = _run(_compbench(tmp_path / "cb", SIZES), tmp_path / "out", "--heldout", "32")

    held = Counter(row["metadata"]["category"] for row in heldout)
    # 160 prompts, 40 in one category: its exact share is 8, the others' 4 each.
    assert held == Counter(dict.fromkeys(CATEGORIES, 4) | {"consistent_attribute_binding": 8})
    requests = [row["messages"][0]["content"] for row in train + heldout]
    assert len(requests) == len(set(requests)) == 160
    assert all(row["messages"][0]["role"] == "user" and len(row["messages"]) == 1 for row in train + heldout)


def test_the_largest_remainders_fill_the_held_out_set_exactly(tmp_path):
    _, heldout = _run(_compbench(tmp_path / "cb", SIZES), tmp_path / "out", "--heldout", "30")

    held = Counter(row["metadata"]["category"] for row in heldout)
    # Shares 7.5 and six of 3.75: the five seats left go to five of the larger remainders, 0.75 over 0.5.
    assert len(heldout) == 30 and held["consistent_attribute_binding"] == 7
    assert sorted(held[c] for c in CATEGORIES if c != "consistent_attribute_binding") == [3, 4, 4, 4, 4, 4]


def test_the_split_follows_the_seed_and_not_the_file_order(tmp_path):
    _, heldout = _run(_compbench(tmp_path / "a", SIZES), tmp_path / "out_a", "--heldout", "32")
    reversed_cb = _compbench(tmp_path / "b", SIZES, order=lambda pairs: pairs[::-1])
    _, reversed_heldout = _run(reversed_cb, tmp_path / "out_b", "--heldout", "32")
    _, reseeded = _run(tmp_path / "a", tmp_path / "out_c", "--heldout", "32", "--seed", "1")

    def requests(rows):
        return {row["messages"][0]["content"] for row in rows}

    assert requests(heldout) == requests(reversed_heldout)
    assert requests(heldout) != requests(reseeded) and len(reseeded) == 32


def test_each_row_names_its_source_category_id_and_subjects(tmp_path):
    train, heldout = _run(_compbench(tmp_path / "cb", SIZES), tmp_path / "out", "--heldout", "0")

    assert heldout == []
    by_id = {row["metadata"]["id"]: row for row in train}
    assert by_id["consistent_attribute_binding/003"] == {
        "messages": [{"role": "user", "content": "consistent_attribute_binding prompt 3"}],
        "metadata": {
            "source": "t2v-compbench",
            "category": "consistent_attribute_binding",
            "id": "consistent_attribute_binding/003",
            "subjects": ["a blue car 3", "a white fence 3"],
        },
    }
    subjects = {category: by_id[f"{category}/002"]["metadata"]["subjects"] for category in CATEGORIES}
    assert subjects == {
        "consistent_attribute_binding": ["a blue car 2", "a white fence 2"],
        "dynamic_attribute_binding": [],
        "spatial_relationships": ["dog 2", "bicycle"],
        "motion_binding": ["cat 2"],
        "action_binding": ["a dog 2", "a cat"],
        "object_interactions": [],
        "generative_numeracy": ["tree 2", "bee"],
    }
    # Motion binding's repeated prompt is kept once, at its first line.
    assert "motion_binding/020" not in by_id and len(train) == 160
