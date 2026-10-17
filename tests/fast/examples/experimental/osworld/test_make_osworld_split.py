"""Offline tests: no GPU, no network, no Gym.

make_osworld_split.py's one claim: the held-out side carries the same domain mix
as training, is disjoint from it, and is reproducible from its recorded seed.
"""

from __future__ import annotations

import json

import pytest

from scripts.tools.make_osworld_split import (
    _read_tasks,
    _stratified_split,
    main,
    read_task_outcomes,
    select_by_variance,
)


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
    main(source=source, out_dir=out_dir, held_out_fraction=0.4, seed=99, only_domains="", variance_from=None)

    provenance = json.loads((out_dir / "provenance.json").read_text())
    assert provenance["seed"] == 99
    assert len(provenance["source_sha256"]) == 64
    assert provenance["counts"] == {"candidates": 10, "train": 6, "held_out": 4}
    assert provenance["selection"] == "domain stratification only"
    assert not set(provenance["train_task_ids"]) & set(provenance["held_out_task_ids"])

    train_rows = (out_dir / "train.jsonl").read_text().splitlines()
    held_rows = (out_dir / "held_out.jsonl").read_text().splitlines()
    assert len(train_rows) == 6 and len(held_rows) == 4


def test_an_empty_selection_is_refused_rather_than_written(tmp_path):
    source = _write_source(tmp_path / "src.jsonl", [_task("t1", "chrome")])
    with pytest.raises(Exception, match="no tasks read"):
        main(
            source=source,
            out_dir=tmp_path / "out",
            held_out_fraction=0.3,
            seed=1,
            only_domains="nonexistent",
            variance_from=None,
        )


def _rollout(task_id, reward, eligible=True, masked=False):
    return {
        "reward": reward,
        "runtime_eligible": eligible,
        "mask_sample": masked,
        "verifier_metadata": {"task_id": task_id},
    }


def _write_probe(path, rows):
    path.write_text("".join(f"{json.dumps(r)}\n" for r in rows))
    return path


def test_outcomes_count_only_episodes_gym_admitted(tmp_path):
    """An episode Gym refused never measured the policy, so letting it stand in
    for a failure would make a task look unanimous when nothing was learned."""
    probe = _write_probe(
        tmp_path / "r.jsonl",
        [
            _rollout("t1", 1.0),
            _rollout("t1", 0.0),
            _rollout("t1", 0.0, eligible=False),
            _rollout("t1", 0.0, masked=True),
            _rollout("t1", True),
        ],
    )
    assert read_task_outcomes(probe) == {"t1": [1.0, 0.0]}


def test_only_tasks_whose_own_rewards_differ_are_kept(tmp_path):
    """A GRPO group is the N samples of one prompt, so a task whose samples all
    score the same gives exactly zero advantage."""
    source = _write_source(tmp_path / "src.jsonl", [_task(f"t{i}", "chrome") for i in range(4)])
    tasks = _read_tasks(source)
    outcomes = {
        "t0": [1.0, 1.0, 1.0, 1.0],
        "t1": [0.0, 0.0, 0.0, 0.0],
        "t2": [1.0, 0.0, 0.0, 1.0],
        "t3": [0.0, 1.0, 0.0, 0.0],
    }
    kept, report = select_by_variance(tasks, outcomes)
    assert [t.task_id for t in kept] == ["t2", "t3"]
    assert {entry["task_id"] for entry in report["no_spread"]} == {"t0", "t1"}
    assert {entry["task_id"] for entry in report["with_spread"]} == {"t2", "t3"}


def test_partial_raw_rewards_that_differ_are_kept(tmp_path):
    """reward_mode: raw preserves partial scores, so [0.2, 0.8] has real spread
    and a non-zero advantage. Recoding every positive reward as "success" threw
    exactly these tasks away."""
    source = _write_source(tmp_path / "src.jsonl", [_task("t1", "chrome"), _task("t2", "gimp"), _task("t3", "vlc")])
    kept, report = select_by_variance(
        _read_tasks(source),
        {"t1": [0.2, 0.8], "t2": [0.4, 1.0], "t3": [0.6, 0.6]},
    )
    assert [t.task_id for t in kept] == ["t1", "t2"]
    assert [entry["task_id"] for entry in report["no_spread"]] == ["t3"]
    spreads = {entry["task_id"]: entry["spread"] for entry in report["with_spread"]}
    assert spreads["t1"] == pytest.approx(0.6)
    # The binary count would have called both of these unanimous successes.
    assert all(entry["successes"] == entry["samples"] for entry in report["with_spread"])


def test_a_task_probed_once_cannot_show_variance(tmp_path):
    source = _write_source(tmp_path / "src.jsonl", [_task("t1", "chrome")])
    kept, report = select_by_variance(_read_tasks(source), {"t1": [1.0]})
    assert kept == [] and report["not_probed"] == ["t1"]


def test_a_split_with_no_spread_anywhere_is_refused(tmp_path):
    """Training on it would make every optimizer step a no-op, which is worth
    an error rather than a run that looks like it worked."""
    source = _write_source(tmp_path / "src.jsonl", [_task(f"t{i}", "chrome") for i in range(4)])
    probe = _write_probe(tmp_path / "r.jsonl", [_rollout(f"t{i}", r) for i in range(4) for r in (0.0, 0.0)])
    with pytest.raises(Exception, match="differing rewards"):
        main(
            source=source,
            out_dir=tmp_path / "out",
            held_out_fraction=0.25,
            seed=1,
            only_domains="",
            variance_from=probe,
        )


def test_the_held_out_side_survives_variance_filtering(tmp_path):
    """Filtering before splitting shrank the pool the split drew from and
    produced an empty held_out.jsonl -- written successfully, so the run went
    ahead and could report no before/after score at all. The holdout is frozen
    from the unfiltered candidates instead."""
    source = _write_source(
        tmp_path / "src.jsonl",
        [_task(f"c{i}", "chrome") for i in range(5)] + [_task(f"g{i}", "gimp") for i in range(5)],
    )
    rows = []
    for name in ("c0", "g0"):
        rows += [_rollout(name, 1.0), _rollout(name, 0.0)]
    for name in ("c1", "c2", "c3", "c4", "g1", "g2", "g3", "g4"):
        rows += [_rollout(name, 0.0), _rollout(name, 0.0)]
    probe = _write_probe(tmp_path / "r.jsonl", rows)
    out_dir = tmp_path / "out"
    main(source=source, out_dir=out_dir, held_out_fraction=0.3, seed=99, only_domains="", variance_from=probe)
    provenance = json.loads((out_dir / "provenance.json").read_text())
    assert provenance["counts"]["held_out"] > 0, "the evaluation set must not be emptied by training-side filtering"
    assert (out_dir / "held_out.jsonl").read_text().strip(), "held_out.jsonl must not be written empty"
    assert not set(provenance["train_task_ids"]) & set(provenance["held_out_task_ids"])


def test_a_requested_holdout_that_yields_nothing_is_refused(tmp_path):
    source = _write_source(tmp_path / "src.jsonl", [_task("only", "chrome")])
    with pytest.raises(Exception, match="no evaluation tasks"):
        main(
            source=source,
            out_dir=tmp_path / "out",
            held_out_fraction=0.3,
            seed=1,
            only_domains="",
            variance_from=None,
        )


def test_provenance_records_the_variance_selection(tmp_path):
    source = _write_source(tmp_path / "src.jsonl", [_task(f"t{i}", f"d{i % 2}") for i in range(6)])
    probe = _write_probe(
        tmp_path / "r.jsonl",
        [_rollout(f"t{i}", r) for i in range(6) for r in ((1.0, 0.0) if i % 2 == 0 else (0.0, 0.0))],
    )
    out_dir = tmp_path / "out"
    main(source=source, out_dir=out_dir, held_out_fraction=0.0, seed=5, only_domains="", variance_from=probe)
    provenance = json.loads((out_dir / "provenance.json").read_text())
    assert provenance["selection"] == "reward spread on the training side only"
    assert provenance["counts"]["candidates"] == 6
    assert len(provenance["variance_report"]["no_spread"]) == 3
