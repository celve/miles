"""Records CUA-Lite's evaluated episodes under ``--eval-record-dir`` (format: ``miles.utils.eval_record``).

CUA-Lite's Miles port evaluates with its own loop, ``lite.train.rollout.core.engine._eval_rollout``,
so Miles's eval loop never hands its samples a record directory. ``install()`` wraps three of
CUA-Lite's names instead: ``_eval_rollout`` marks an eval point, ``generate_and_rm`` opens an
episode for each sample it runs there, and ``AdapterBasedAgent.sample`` adds a hook that writes
each step's observation, the model's raw reply and the actions it executed, then the episode
return. Outside an eval point, or with ``--eval-record-dir`` unset, each wrapper only calls through.

Episodes land in ``<eval_record_dir>/eval_<rollout_id>/<env_key>/<sample index>``. Turn it on from
the rollout module the launcher names, before its first rollout:

    from examples.experimental.cua_lite import record
    record.install()
"""

import functools
import io
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from miles.utils.eval_record import EpisodeWriter, episode_dir, eval_point_dir, finish_episode, start_episode

_eval_point: ContextVar[str | None] = ContextVar("cua_lite_eval_point", default=None)
_episode: ContextVar[Path | None] = ContextVar("cua_lite_episode", default=None)


def install() -> None:
    from lite.agents.core.agent.base import AdapterBasedAgent
    from lite.train.rollout.core import engine

    if getattr(engine, "_miles_eval_record_installed", False):
        return
    engine._eval_rollout = _marking_eval_point(engine._eval_rollout)
    engine.generate_and_rm = _opening_episode(engine.generate_and_rm)
    AdapterBasedAgent.sample = _recording_sample(AdapterBasedAgent.sample)
    engine._miles_eval_record_installed = True


def _marking_eval_point(eval_rollout):
    @functools.wraps(eval_rollout)
    async def wrapper(args, rollout_id, *rest, **kwargs):
        if args.eval_record_dir is None:
            return await eval_rollout(args, rollout_id, *rest, **kwargs)
        token = _eval_point.set(eval_point_dir(args.eval_record_dir, rollout_id))
        try:
            return await eval_rollout(args, rollout_id, *rest, **kwargs)
        finally:
            _eval_point.reset(token)

    return wrapper


def _opening_episode(generate_and_rm):
    @functools.wraps(generate_and_rm)
    async def wrapper(args, sample, *rest, **kwargs):
        if (point := _eval_point.get()) is None:
            return await generate_and_rm(args, sample, *rest, **kwargs)
        task_id = str(sample.metadata["env_key"])
        directory = start_episode(
            episode_dir(point, task_id, sample.index), stack="cua-lite", task_id=task_id, sample_index=sample.index
        )
        token = _episode.set(directory)
        try:
            return await generate_and_rm(args, sample, *rest, **kwargs)
        finally:
            _episode.reset(token)

    return wrapper


def _recording_sample(sample):
    @functools.wraps(sample)
    async def wrapper(self, env, *rest, hooks=None, **kwargs):
        if (directory := _episode.get()) is not None:
            hooks = [*(hooks or []), RecordHook(directory)]
        return await sample(self, env, *rest, hooks=hooks, **kwargs)

    return wrapper


class RecordHook:
    """A CUA-Lite ``SampleHook``: ``agent.sample`` calls it off the event loop, once per step and once at the end."""

    def __init__(self, directory: Path) -> None:
        self._writer = EpisodeWriter(directory)

    def on_step(self, data: Any) -> None:
        screenshot = None
        if data.image is not None:
            buffer = io.BytesIO()
            data.image.save(buffer, format="PNG")
            screenshot = buffer.getvalue()
        self._writer.add_step(
            output=data.predict_result.step.response,
            screenshot=screenshot,
            action=data.actions,
            error=data.predict_result.model_output_error,
        )

    def on_complete(self, lite_rl_sample: Any) -> None:
        # CUA-Lite passes None when the episode raised; its rollout then scores it as errored.
        if lite_rl_sample is None:
            finish_episode(self._writer.directory, reward=None, status="failed")
            return
        from lite.core.messages.selectors import instruction_text

        self._writer.set_instruction(instruction_text(lite_rl_sample.lite_sample.messages))
        status = "truncated" if lite_rl_sample.truncated else "terminated" if lite_rl_sample.terminated else "ended"
        finish_episode(self._writer.directory, reward=float(lite_rl_sample.episode_return), status=status)
