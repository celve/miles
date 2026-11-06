import json
import subprocess
import sys

from examples.experimental.video_gen_agent.make_requests import split_requests

REQUESTS = [f"request number {i}" for i in range(200)]


def test_split_is_stable_under_reordering_and_supersets():
    train, heldout = split_requests(REQUESTS, heldout_percent=10)
    shuffled_train, shuffled_heldout = split_requests(REQUESTS[::-1] + ["one more request"], heldout_percent=10)

    assert set(heldout) <= set(shuffled_heldout) and set(train) <= set(shuffled_train)
    assert not set(train) & set(heldout) and len(train) + len(heldout) == len(REQUESTS)
    assert 5 <= len(heldout) <= 40


def test_writes_prompt_data_without_blank_or_repeated_requests(tmp_path):
    source = tmp_path / "prompts.txt"
    source.write_text("\n".join(["a cat surfing", "", "a cat surfing", "  a dog skiing  "]) + "\n")

    subprocess.run(
        [
            sys.executable,
            "examples/experimental/video_gen_agent/make_requests.py",
            str(source),
            str(tmp_path / "out"),
            "--heldout-percent",
            "0",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert rows == [
        {"messages": [{"role": "user", "content": "a cat surfing"}]},
        {"messages": [{"role": "user", "content": "a dog skiing"}]},
    ]
    assert (tmp_path / "out" / "heldout.jsonl").read_text() == ""
