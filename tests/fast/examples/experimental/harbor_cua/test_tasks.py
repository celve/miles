"""The task set: a disjoint split, substituted pages, and a verifier contract."""

import json

import pytest

from examples.experimental.harbor_cua.make_harbor_cua_tasks import generate, substitute, write_task
from examples.experimental.harbor_cua.tasks.families import FAMILIES, instance_id, split_variants


@pytest.fixture
def generated(tmp_path):
    counts = generate(
        tasks_dir=tmp_path / "tasks",
        data_dir=tmp_path / "data",
        train_per_family=4,
        eval_per_family=2,
    )
    return tmp_path, counts


def test_the_split_shares_no_instance(generated):
    tmp_path, counts = generated
    train = [json.loads(line) for line in (tmp_path / "data" / "train.jsonl").read_text().splitlines()]
    evaluation = [json.loads(line) for line in (tmp_path / "data" / "eval.jsonl").read_text().splitlines()]

    assert counts == {"train": 20, "eval": 10, "families": 5}
    assert {row["label"] for row in train} & {row["label"] for row in evaluation} == set()
    assert {row["metadata"]["split"] for row in train} == {"train"}
    assert {row["metadata"]["split"] for row in evaluation} == {"eval"}


def test_every_row_records_its_provenance(generated):
    tmp_path, _ = generated
    rows = [json.loads(line) for line in (tmp_path / "data" / "train.jsonl").read_text().splitlines()]

    for row in rows:
        metadata = row["metadata"]
        assert metadata["family"] in {family.name for family in FAMILIES}
        assert metadata["variant"]
        assert metadata["provenance"].endswith("families.py")
        assert metadata["agent_name"] == "computer-1"


def test_each_row_names_a_task_directory_that_exists(generated):
    tmp_path, _ = generated
    rows = [json.loads(line) for line in (tmp_path / "data" / "train.jsonl").read_text().splitlines()]

    for row in rows:
        task_dir = tmp_path / "tasks" / row["metadata"]["instance_id"]
        assert (task_dir / "task.toml").is_file()
        assert (task_dir / "instruction.md").is_file()
        assert (task_dir / "environment" / "Dockerfile").is_file()
        assert (task_dir / "environment" / "page.html").is_file()
        assert (task_dir / "tests" / "test.sh").is_file()


def test_no_placeholder_survives_generation(generated):
    tmp_path, _ = generated

    for path in (tmp_path / "tasks").rglob("*"):
        if path.suffix in {".html", ".md", ".toml"}:
            assert "__" not in path.read_text().replace("__pycache__", ""), path


def test_the_verifier_gets_the_stage_count_and_any_expected_answer(tmp_path):
    for family in FAMILIES:
        variant = family.variants[0]
        task_dir = write_task(tmp_path, family, variant)
        toml = (task_dir / "task.toml").read_text()
        assert f'STAGES_TOTAL = "{variant.stages_total}"' in toml
        assert f'EXPECTED_ANSWER = "{variant.expected_answer or ""}"' in toml


def test_the_graded_family_can_score_between_zero_and_one():
    """A partially solved episode has to be representable, or GRPO sees no spread."""
    two_step = next(family for family in FAMILIES if family.name == "two-step")

    assert all(variant.stages_total == 2 for variant in two_step.variants)


def test_asking_for_more_instances_than_a_family_has_is_refused():
    family = FAMILIES[0]

    with pytest.raises(ValueError, match="fewer than"):
        split_variants(family, train=len(family.variants), evaluation=1)


def test_substitution_is_a_plain_token_replacement():
    assert substitute("a __X__ b __Y__", {"X": "1", "Y": "2"}) == "a 1 b 2"


def test_instance_ids_are_safe_for_harbor_task_paths():
    for family in FAMILIES:
        for variant in family.variants:
            name = instance_id(family, variant)
            assert name.replace("-", "").replace("_", "").isalnum()
