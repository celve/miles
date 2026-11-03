import io

import typer
from PIL import Image
from scripts.tools.render_eval_records import main
from typer.testing import CliRunner

from miles.utils.eval_record import EpisodeWriter, episode_dir, finish_episode, start_episode


def _png(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 36), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _record(root, task, index, *, reward, steps, status="Submitted"):
    directory = start_episode(episode_dir(root, task, index), stack="harbor", task_id=task, sample_index=index)
    writer = EpisodeWriter(directory)
    writer.set_instruction(f"Solve {task} <now>")
    for step in range(steps):
        writer.add_step(
            output=f"{task} reply {step}", screenshot=_png((step * 40, 0, 0)), action={"type": "click", "x": step}
        )
    finish_episode(directory, reward=reward, status=status)


def _recordings(tmp_path):
    base, checkpoint = tmp_path / "base" / "eval_0", tmp_path / "ckpt" / "eval_0"
    for index in range(4):
        _record(base, "learned", index, reward=0.0, steps=3)
        _record(checkpoint, "learned", index, reward=1.0 if index else 0.0, steps=2)
        _record(base, "both_pass", index, reward=1.0, steps=1)
        _record(checkpoint, "both_pass", index, reward=1.0, steps=1)
    _record(base, "base_only", 0, reward=0.0, steps=1)
    return base, checkpoint


def _render(*args):
    app = typer.Typer()
    app.command()(main)
    result = CliRunner().invoke(app, [str(arg) for arg in args])
    assert result.exit_code == 0, result.output
    return result


def test_the_page_puts_both_sides_of_every_shared_task_in_aligned_rows(tmp_path):
    base, checkpoint = _recordings(tmp_path)
    out = tmp_path / "page.html"

    _render("--base", base, "--checkpoint", checkpoint, "--out", out)

    page = out.read_text()
    assert "id='learned'" in page and "id='both_pass'" in page and "base_only" not in page
    assert page.count("data:image/jpeg;base64,") == 3 + 2 + 1 + 1
    assert "Solve learned &lt;now&gt;" in page
    assert "<td>0/4</td><td>3/4</td>" in page
    assert page.count("class='empty'") + page.count('class="empty"') == 1


def test_contrast_keeps_tasks_base_failed_and_shows_a_passing_checkpoint_episode(tmp_path):
    base, checkpoint = _recordings(tmp_path)
    out = tmp_path / "page.html"

    _render("--base", base, "--checkpoint", checkpoint, "--out", out, "--select", "contrast", "--pair", "contrast")

    page = out.read_text()
    assert "id='learned'" in page and "id='both_pass'" not in page
    assert "sample 0 · 3 steps" in page and "sample 1 · 2 steps" in page


def test_a_gif_steps_both_episodes_together(tmp_path):
    base, checkpoint = _recordings(tmp_path)

    _render(
        "--base",
        base,
        "--checkpoint",
        checkpoint,
        "--out",
        tmp_path / "p.html",
        "--task",
        "learned",
        "--gif-dir",
        tmp_path / "gifs",
    )

    gif = Image.open(tmp_path / "gifs" / "learned.gif")
    assert gif.n_frames == 3
    assert gif.size[0] == 2 * 64


def test_no_shared_task_is_an_error(tmp_path):
    base, checkpoint = _recordings(tmp_path)
    app = typer.Typer()
    app.command()(main)

    result = CliRunner().invoke(
        app,
        [
            "--base",
            str(base),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(tmp_path / "p.html"),
            "--task",
            "base_only",
        ],
    )

    assert result.exit_code != 0


def test_an_unscored_base_is_not_a_failure_contrast_can_pick(tmp_path):
    base, checkpoint = tmp_path / "base" / "eval_0", tmp_path / "ckpt" / "eval_0"
    _record(base, "lost", 0, reward=None, steps=0, status="SandboxLost")
    _record(checkpoint, "lost", 0, reward=1.0, steps=1)
    _record(base, "learned", 0, reward=0.0, steps=1)
    _record(base, "learned", 1, reward=None, steps=0, status="SandboxLost")
    _record(checkpoint, "learned", 0, reward=1.0, steps=1)
    out = tmp_path / "page.html"

    _render("--base", base, "--checkpoint", checkpoint, "--out", out, "--select", "contrast", "--pair", "contrast")

    page = out.read_text()
    assert "id='lost'" not in page and "id='learned'" in page
    assert "<td>0/1 (1 unscored)</td><td>1/1</td>" in page
    assert "badge fail" in page and "badge unscored" not in page


def test_a_root_holding_several_eval_points_is_refused(tmp_path):
    run = tmp_path / "run"
    _record(run / "eval_0", "t", 0, reward=0.0, steps=1)
    _record(run / "eval_49", "t", 0, reward=1.0, steps=1)
    app = typer.Typer()
    app.command()(main)

    result = CliRunner().invoke(
        app, ["--base", str(run), "--checkpoint", str(run / "eval_49"), "--out", str(tmp_path / "p.html")]
    )

    assert result.exit_code != 0
    assert "several eval points" in result.output
