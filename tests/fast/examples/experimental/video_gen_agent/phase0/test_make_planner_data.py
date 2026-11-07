import json
import subprocess
import sys

from examples.experimental.video_gen_agent.make_requests import is_heldout
from examples.experimental.video_gen_agent.phase0.make_planner_data import SYSTEM_PROMPT

REQUESTS = [f"a clip of request number {i}" for i in range(100)]


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _requests_of(rows):
    return [row["metadata"]["video_request"] for row in rows]


def test_writes_the_planner_chats_split_as_the_env_splits_them_with_a_raw_copy_of_heldout(tmp_path):
    source = tmp_path / "prompts.txt"
    source.write_text("\n".join(REQUESTS + ["", REQUESTS[0]]) + "\n")

    subprocess.run(
        [
            sys.executable,
            *("-m", "examples.experimental.video_gen_agent.phase0.make_planner_data"),
            *(str(source), str(tmp_path / "out"), "--heldout-percent", "20"),
        ],
        check=True,
    )

    train, heldout, raw = (_read(tmp_path / "out" / f"{name}.jsonl") for name in ("train", "heldout", "heldout_raw"))
    assert _requests_of(heldout) == [r for r in REQUESTS if is_heldout(r, 20)] == _requests_of(raw)
    assert _requests_of(train) == [r for r in REQUESTS if not is_heldout(r, 20)]
    assert heldout and train
    assert train[0]["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": train[0]["metadata"]["video_request"]},
    ]
    assert all(row["metadata"]["send_request_to_generator"] for row in raw)
    assert not any("send_request_to_generator" in row["metadata"] for row in train + heldout)
