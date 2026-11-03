"""The CUA-Lite recorder against stand-ins for the three CUA-Lite names it wraps, shaped as CUA-Lite calls them."""

import asyncio
import sys
import textwrap
import types
from types import SimpleNamespace

import pytest
from PIL import Image

from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.eval_record import iter_episodes

# The call chain of lite.train.rollout.core.engine and lite.agents.core.agent.base, cut to what the recorder sees.
_ENGINE = textwrap.dedent("""
    import asyncio
    from lite.agents.core.agent.base import AdapterBasedAgent

    seen_metadata = []

    async def generate_and_rm(args, sample, sampling_params=None, evaluation=False):
        seen_metadata.append(dict(sample.metadata))
        return await AdapterBasedAgent().sample(SimpleNamespace(task=sample.metadata["env_key"]))

    async def _eval_rollout(args, rollout_id):
        # One dataset after another, each numbering its samples from 0.
        for dataset_cfg in args.eval_datasets:
            tasks = []
            for index, metadata in enumerate(args.prompts):
                sample = SimpleNamespace(index=index, metadata=dataset_cfg.inject_metadata(metadata))
                tasks.append(asyncio.create_task(generate_and_rm(args, sample, sampling_params={}, evaluation=True)))
            await asyncio.gather(*tasks)

    async def _generate_rollout_async(args, rollout_id):
        samples = [SimpleNamespace(index=i, metadata=dict(m)) for i, m in enumerate(args.prompts)]
        return await asyncio.gather(*(generate_and_rm(args, s, {}) for s in samples))
    """)
_BASE = textwrap.dedent("""
    import asyncio

    class AdapterBasedAgent:
        async def sample(self, env, max_steps=10_000, hooks=None):
            hooks = hooks or []
            result = SimpleNamespace(
                episode_return=1.0 if env.task.endswith("@win") else 0.0,
                terminated=True,
                truncated=False,
                lite_sample=SimpleNamespace(messages=[f"do {env.task}"]),
            )
            for step_idx in range(2):
                data = SimpleNamespace(
                    step_idx=step_idx,
                    image=Image.new("RGB", (8, 8), (step_idx * 100, 0, 0)),
                    predict_result=SimpleNamespace(
                        step=SimpleNamespace(response=f"reply {step_idx}"), model_output_error=None
                    ),
                    actions=[{"name": "click", "arguments": {"x": step_idx}}],
                )
                for hook in hooks:
                    await asyncio.to_thread(hook.on_step, data)
            for hook in hooks:
                await asyncio.to_thread(hook.on_complete, result)
            return result
    """)


def _module(name: str, source: str = "", **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(SimpleNamespace=SimpleNamespace, Image=Image, **attrs)
    exec(source, module.__dict__)
    return module


@pytest.fixture
def lite(monkeypatch):
    modules = {name: types.ModuleType(name) for name in ("lite", "lite.agents", "lite.agents.core", "lite.train")}
    modules["lite.agents.core.agent.base"] = base = _module("lite.agents.core.agent.base", _BASE)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    modules["lite.train.rollout.core.engine"] = _module("lite.train.rollout.core.engine", _ENGINE)
    modules["lite.train.rollout.core"] = core = types.ModuleType("lite.train.rollout.core")
    core.engine = modules["lite.train.rollout.core.engine"]
    modules["lite.core.messages.selectors"] = _module(
        "lite.core.messages.selectors", instruction_text=lambda messages: messages[0]
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(EvalDatasetConfig, "inject_metadata", EvalDatasetConfig.inject_metadata)
    from examples.experimental.cua_lite import record

    record.install()
    return core.engine


def _args(tmp_path, record=True, datasets=("heldout",)):
    return SimpleNamespace(
        eval_record_dir=str(tmp_path / "records") if record else None,
        eval_datasets=[EvalDatasetConfig(name=name, path=f"{name}.parquet") for name in datasets],
        prompts=[{"env_key": f"lite.osworld/calc@{task}"} for task in ["win", "win", "lose"]],
    )


def test_an_eval_point_records_every_episode_with_its_steps_and_return(lite, tmp_path):
    asyncio.run(lite._eval_rollout(_args(tmp_path), 7))

    episodes = sorted(iter_episodes(tmp_path / "records" / "eval_7" / "heldout"), key=lambda e: e.sample_index)
    assert [(e.stack, e.dataset, e.task_id, e.sample_index, e.reward) for e in episodes] == [
        ("cua-lite", "heldout", "lite.osworld/calc@win", 0, 1.0),
        ("cua-lite", "heldout", "lite.osworld/calc@win", 1, 1.0),
        ("cua-lite", "heldout", "lite.osworld/calc@lose", 2, 0.0),
    ]
    first = episodes[0]
    assert first.instruction == "do lite.osworld/calc@win"
    assert first.status == "terminated"
    assert [step.output for step in first.steps] == ["reply 0", "reply 1"]
    assert first.steps[1].action == [{"name": "click", "arguments": {"x": 1}}]
    assert Image.open(first.steps[1].screenshot).getpixel((0, 0)) == (100, 0, 0)


def test_datasets_that_number_their_samples_alike_keep_their_own_episodes(lite, tmp_path):
    asyncio.run(lite._eval_rollout(_args(tmp_path, datasets=("probe", "heldout")), 7))

    recorded = sorted((e.dataset, e.sample_index) for e in iter_episodes(tmp_path / "records" / "eval_7"))
    assert recorded == [("heldout", 0), ("heldout", 1), ("heldout", 2), ("probe", 0), ("probe", 1), ("probe", 2)]


def test_the_evaluated_sample_carries_no_recording_tag(lite, tmp_path):
    asyncio.run(lite._eval_rollout(_args(tmp_path), 7))

    assert lite.seen_metadata == [{"env_key": f"lite.osworld/calc@{t}"} for t in ["win", "win", "lose"]]


def test_training_rollouts_are_not_recorded(lite, tmp_path):
    asyncio.run(lite._generate_rollout_async(_args(tmp_path), 7))

    assert not (tmp_path / "records").exists()


def test_without_a_record_dir_the_eval_records_nothing(lite, tmp_path):
    asyncio.run(lite._eval_rollout(_args(tmp_path, record=False), 7))

    assert not (tmp_path / "records").exists()


def test_installing_twice_records_each_step_once(lite, tmp_path):
    from examples.experimental.cua_lite import record

    record.install()
    asyncio.run(lite._eval_rollout(_args(tmp_path), 0))

    assert all(len(episode.steps) == 2 for episode in iter_episodes(tmp_path / "records"))
