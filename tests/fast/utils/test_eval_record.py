import base64

from miles.utils.eval_record import (
    EpisodeWriter,
    decode_data_url,
    episode_dir,
    finish_episode,
    iter_episodes,
    load_episode,
    start_episode,
)


def _open(tmp_path, task="lo-calc@bold_header", index=3):
    return start_episode(episode_dir(tmp_path, task, index), stack="harbor", task_id=task, sample_index=index)


def test_an_episode_reads_back_as_it_was_written(tmp_path):
    directory = _open(tmp_path)
    writer = EpisodeWriter(directory)
    writer.set_instruction("Make the header bold.")
    writer.add_step(output='{"action": "click"}', screenshot=b"png-0", action={"type": "click", "x": 10})
    writer.add_step(output="garbled", screenshot=b"webp-1", media_type="image/webp", error="ERROR: no JSON")
    finish_episode(directory, reward=1.0, status="Submitted", turns=2)

    episode = load_episode(directory)

    assert (episode.stack, episode.task_id, episode.sample_index) == ("harbor", "lo-calc@bold_header", 3)
    assert episode.instruction == "Make the header bold."
    assert [step.screenshot.read_bytes() for step in episode.steps] == [b"png-0", b"webp-1"]
    assert episode.steps[1].screenshot.suffix == ".webp"
    assert [step.action for step in episode.steps] == [{"type": "click", "x": 10}, None]
    assert episode.steps[1].error == "ERROR: no JSON"
    assert (episode.reward, episode.status, episode.outcome["turns"]) == (1.0, "Submitted", 2)


def test_an_episode_that_never_finished_reads_as_unfinished(tmp_path):
    directory = _open(tmp_path)
    EpisodeWriter(directory).add_step(output="a")

    episode = load_episode(directory)

    assert (episode.reward, episode.status, len(episode.steps)) == (None, "unfinished", 1)


def test_a_retried_episode_replaces_the_last_try(tmp_path):
    EpisodeWriter(_open(tmp_path)).add_step(output="first try", screenshot=b"x")

    directory = _open(tmp_path)
    EpisodeWriter(directory).add_step(output="second try")

    assert [step.output for step in load_episode(directory).steps] == ["second try"]
    assert not (directory / "step_000.png").exists()


def test_the_task_id_becomes_one_safe_path_component(tmp_path):
    assert episode_dir(tmp_path, "lite.osworld/../etc@x", 0) == tmp_path / "lite.osworld_.._etc_x" / "0"
    assert episode_dir(tmp_path, "..", 1) == tmp_path / "task" / "1"


def test_every_episode_under_a_root_is_found(tmp_path):
    for task, index in [("a", 0), ("a", 1), ("b", 0)]:
        start_episode(episode_dir(tmp_path / "eval_0" / "heldout", task, index), stack="cua-lite", task_id=task, sample_index=index)

    assert sorted((e.task_id, e.sample_index) for e in iter_episodes(tmp_path)) == [("a", 0), ("a", 1), ("b", 0)]


def test_only_base64_data_urls_decode():
    url = "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()

    assert decode_data_url(url) == (b"\x89PNG", "image/png")
    assert decode_data_url("https://example.com/a.png") is None
    assert decode_data_url("data:text/plain,hello") is None
