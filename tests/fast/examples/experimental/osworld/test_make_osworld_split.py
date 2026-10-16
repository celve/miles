"""Offline tests: no GPU, no network, no Gym.

make_osworld_split.py's one claim: the held-out side carries the same domain mix
as training, is disjoint from it, and is reproducible from its recorded seed.
"""

from __future__ import annotations

import json

import pytest

from scripts.tools.make_osworld_split import _read_tasks, _stratified_split, main


def _task(task_id, domain):
    return {
        "responses_create_params": {"max_output_tokens": 1500},
        "verifier_metadata": {"task_id": task_id, "osworld_task": {"domain": domain, "id": task_id}},
    }


def _write_source(path, tasks):
    path.write_text("".join(f"{json.dumps(t)}\n" for t in tasks))
    return path


def test_rows_carry_the_task_for_gym_and_no_instruction_for_the_model(tmp_path):
    source = _write_source(tmp_path / "src.jsonl", [_task("t1", "chrome")])
    (task,) = _read_tasks(source)
    assert task.row["prompt"] == "chrome/t1"
    assert task.row["label"] == "chrome"
    assert task.row["metadata"]["osworld_task"]["verifier_metadata"]["task_id"] == "t1"
    # The instruction reaches the policy inside the environment, so nothing the
    # model reads may sit in `prompt`.
    assert "instruction" not in json.dumps(task.row["prompt"])


def test_every_domain_is_held_out_in_proportion(tmp_path):
    source = _write_source(
        tmp_path / "src.jsonl",
        [_task(f"c{i}", "chrome") for i in range(10)] + [_task(f"g{i}", "gimp") for i in range(10)],
    )
    train, held_out = _stratified_split(_read_tasks(source), held_out_fraction=0.3, seed=1)
    assert len(held_out) == 6
    assert sorted(t.domain for t in held_out) == ["chrome"] * 3 + ["gimp"] * 3
    assert len(train) == 14


def test_a_single_task_domain_stays_in_training(tmp_path):
    """A held-out side of one task measures nothing, and losing the domain from
    training costs more."""
    source = _write_source(tmp_path / "src.jsonl", [_task("only", "thunderbird")])
    train, held_out = _stratified_split(_read_tasks(source), held_out_fraction=0.5, seed=1)
    assert [t.task_id for t in train] == ["only"] and held_out == []


def test_the_split_is_disjoint_and_reproducible_from_its_seed(tmp_path):
    source = _write_source(tmp_path / "src.jsonl", [_task(f"t{i}", f"d{i % 4}") for i in range(40)])
    tasks = _read_tasks(source)
    first = _stratified_split(tasks, 0.25, seed=7)
    second = _stratified_split(tasks, 0.25, seed=7)
    assert [t.task_id for t in first[1]] == [t.task_id for t in second[1]]
    assert not {t.task_id for t in first[0]} & {t.task_id for t in first[1]}

    different = _stratified_split(tasks, 0.25, seed=8)
    assert [t.task_id for t in different[1]] != [t.task_id for t in first[1]]


def test_provenance_records_what_a_before_after_claim_rests_on(tmp_path):
    source = _write_source(tmp_path / "src.jsonl", [_task(f"t{i}", f"d{i % 2}") for i in range(10)])
    out_dir = tmp_path / "out"
    main(source=source, out_dir=out_dir, held_out_fraction=0.4, seed=99, only_domains="")

    provenance = json.loads((out_dir / "provenance.json").read_text())
    assert provenance["seed"] == 99
    assert len(provenance["source_sha256"]) == 64
    assert provenance["counts"] == {"total": 10, "train": 6, "held_out": 4}
    assert not set(provenance["train_task_ids"]) & set(provenance["held_out_task_ids"])

    train_rows = (out_dir / "train.jsonl").read_text().splitlines()
    held_rows = (out_dir / "held_out.jsonl").read_text().splitlines()
    assert len(train_rows) == 6 and len(held_rows) == 4


def test_an_empty_selection_is_refused_rather_than_written(tmp_path):
    source = _write_source(tmp_path / "src.jsonl", [_task("t1", "chrome")])
    with pytest.raises(Exception, match="no tasks read"):
        main(source=source, out_dir=tmp_path / "out", held_out_fraction=0.3, seed=1, only_domains="nonexistent")
