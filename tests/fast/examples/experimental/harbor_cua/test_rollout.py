"""The generate function: identity, write-offs and the call-accounting guard."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from examples.experimental.harbor_cua import rollout as harbor_cua
from examples.experimental.harbor_cua.trial_driver import TrialOutcome
from miles.utils.types import Sample
from tests.fast.examples.experimental.harbor_cua.conftest import png_bytes


def args(**overrides):
    return Namespace(
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=31000,
        rollout_max_context_len=65536,
        harbor_cua_max_turns=4,
        harbor_cua_max_sandboxes=2,
        **overrides,
    )


def task_sample(index, **metadata):
    return Sample(index=index, metadata={"instance_id": "cua/click-red", **metadata})


def fn_input(sample, state):
    """The three attributes the generate function reads, without pulling in base_types."""
    return SimpleNamespace(
        state=state, sample=sample, sampling_params={"temperature": 1.0}, evaluation=False, args=state.args
    )


@pytest.fixture
def state(fake_processor, request):
    return SimpleNamespace(args=getattr(request, "param", args()), processor=fake_processor, tokenizer=fake_processor.tokenizer)


@pytest.fixture
def stub(monkeypatch, episode):
    """Stand in for the recorder and the trial, leaving conversion real."""
    holder = SimpleNamespace(record=episode.record, outcome=None, api_base=None)

    class Recorder:
        port = 1234

        def open_rollout(self):
            return "tok"

        def base_url(self, token):
            return f"http://127.0.0.1:1234/r/{token}/v1"

        def take(self, token):
            return holder.record

    async def shared_recorder(_args, _options):
        return Recorder()

    async def run_trial(**kwargs):
        holder.api_base = kwargs["api_base"]
        return holder.outcome

    monkeypatch.setattr(harbor_cua, "_shared_recorder", shared_recorder)
    monkeypatch.setattr(harbor_cua, "run_trial", run_trial)
    return holder


def outcome(reward=1.0, exit_status="Submitted", episodes=2):
    return TrialOutcome(
        reward=reward,
        exit_status=exit_status,
        metadata={"reward": reward, "exit_status": exit_status, "agent_metrics": {"n_episodes": episodes}},
        trial_dir="/tmp/trial",
    )


async def test_an_episode_becomes_siblings_sharing_identity_and_reward(stub, state, episode):
    episode.turn(image=png_bytes((1, 0, 0)))
    episode.rewrite(media=[], prompt=[1, 2, 3, 400])
    episode.turn(image=png_bytes((2, 0, 0)))
    stub.outcome = outcome(reward=0.5, episodes=2)

    output = await harbor_cua.generate(fn_input(task_sample(4), state))

    samples = output.samples
    assert len(samples) == 2
    assert {s.rollout_id for s in samples} == {4}
    assert {s.reward for s in samples} == {0.5}
    assert samples[0].metadata["segments"] == 2
    assert samples[0].metadata["harbor_episodes"] == 2


async def test_the_recorder_url_is_what_the_trial_is_pointed_at(stub, state, episode):
    episode.turn(image=png_bytes((3, 0, 0)))
    stub.outcome = outcome(episodes=1)

    await harbor_cua.generate(fn_input(task_sample(1), state))

    assert stub.api_base == "http://127.0.0.1:1234/r/tok/v1"


async def test_an_infrastructure_failure_aborts_the_group_by_default(stub, state, episode):
    stub.outcome = outcome(reward=0.0, exit_status="AgentError", episodes=0)

    output = await harbor_cua.generate(fn_input(task_sample(2), state))

    (sample,) = output.samples
    assert sample.status is Sample.Status.ABORTED
    assert sample.remove_sample is True
    assert sample.multimodal_train_inputs is not None
    assert sample.loss_mask == []
    assert "AgentError" in sample.metadata["write_off_reason"]


async def test_masking_instead_of_aborting_is_available(stub, state, episode):
    state.args = args(harbor_cua_write_off_mode="mask_sample")
    stub.outcome = outcome(reward=0.0, exit_status="AgentError", episodes=0)

    output = await harbor_cua.generate(fn_input(task_sample(2), state))

    (sample,) = output.samples
    assert sample.status is Sample.Status.FAILED
    assert sample.remove_sample is True


async def test_an_unknown_write_off_mode_is_refused(stub, state):
    state.args = args(harbor_cua_write_off_mode="shrug")
    stub.outcome = outcome(reward=0.0, exit_status="AgentError", episodes=0)

    with pytest.raises(ValueError, match="harbor-cua-write-off-mode"):
        await harbor_cua.generate(fn_input(task_sample(2), state))


async def test_the_options_defaults_reach_the_run(stub, state, episode):
    """Every flag has one default, in HarborCuaOptions, and the rollout reads it there."""
    episode.turn(image=png_bytes((6, 0, 0)))
    stub.outcome = outcome(episodes=1)
    bare = Namespace(partial_rollout=False, sglang_router_ip="127.0.0.1", sglang_router_port=31000)
    state.args = bare

    output = await harbor_cua.generate(fn_input(task_sample(9), state))

    assert len(output.samples) == 1


async def test_fewer_action_calls_than_episodes_is_a_write_off(stub, state, episode):
    episode.turn(image=png_bytes((4, 0, 0)), auxiliary=True)
    stub.outcome = outcome(episodes=3)

    output = await harbor_cua.generate(fn_input(task_sample(5), state))

    (sample,) = output.samples
    assert sample.status is Sample.Status.ABORTED
    assert "action calls for 3 Harbor episodes" in sample.metadata["write_off_reason"]


async def test_a_text_only_episode_is_a_write_off_not_a_padded_sample(stub, state, episode):
    episode.turn()
    stub.outcome = outcome(episodes=1)

    output = await harbor_cua.generate(fn_input(task_sample(6), state))

    (sample,) = output.samples
    assert sample.status is Sample.Status.ABORTED
    assert sample.metadata["write_off_reason"] == "episode produced no trainable segment"
    assert sample.metadata["dropped_text_only_segments"] == 1


async def test_a_missing_instance_id_fails_loudly(stub, state):
    with pytest.raises(ValueError, match="instance_id"):
        await harbor_cua.generate(fn_input(Sample(index=7, metadata={}), state))


async def test_the_processor_fingerprint_is_recorded_on_every_sample(stub, state, episode):
    episode.turn(image=png_bytes((5, 0, 0)))
    stub.outcome = outcome(episodes=1)

    output = await harbor_cua.generate(fn_input(task_sample(8), state))

    assert "merge=2" in output.samples[0].metadata["processor_fingerprint"]


def test_generate_fn_output_contract():
    """Where base_types imports, the stand-in above must match the real type."""
    base_types = pytest.importorskip(
        "miles.rollout.base_types", reason="needs the sglang build a training node has"
    )
    assert set(base_types.GenerateFnOutput.__dataclass_fields__) == {"samples"}


async def test_an_explicit_upstream_is_used_verbatim():
    from examples.experimental.harbor_cua.options import HarborCuaOptions

    options = HarborCuaOptions(upstream="http://10.0.0.9:20035")

    assert await harbor_cua._resolve_upstream(args(), options) == "http://10.0.0.9:20035"


async def test_asking_for_the_router_forwards_there_despite_the_evidence_gap():
    from examples.experimental.harbor_cua.options import HarborCuaOptions

    options = HarborCuaOptions(upstream="router")

    assert await harbor_cua._resolve_upstream(args(), options) == "http://127.0.0.1:31000"


async def test_evaluation_reports_one_sample_per_episode(stub, state, episode):
    """The eval path averages a flat reward list, so N siblings would count N times."""
    episode.turn(image=png_bytes((7, 0, 0)))
    episode.rewrite(media=[], prompt=[1, 2, 3, 900])
    episode.turn(image=png_bytes((8, 0, 0)))
    stub.outcome = outcome(reward=1.0, episodes=2)
    fn = fn_input(task_sample(12), state)
    fn.evaluation = True

    output = await harbor_cua.generate(fn)

    assert len(output.samples) == 1
    assert output.samples[0].metadata["segments"] == 2
    assert output.samples[0].metadata["evaluation_siblings_dropped"] == 1


async def test_training_keeps_every_sibling(stub, state, episode):
    episode.turn(image=png_bytes((7, 0, 0)))
    episode.rewrite(media=[], prompt=[1, 2, 3, 900])
    episode.turn(image=png_bytes((8, 0, 0)))
    stub.outcome = outcome(reward=1.0, episodes=2)

    output = await harbor_cua.generate(fn_input(task_sample(13), state))

    assert len(output.samples) == 2
    assert output.samples[0].metadata["evaluation_siblings_dropped"] == 0
