import pytest

from miles.dashboard import hooks
from miles.dashboard.hooks import BATCH_MAX_EVENTS, TrajectorySink
from miles.dashboard.store import Stream, TrajectoryEvent, TrajectoryEventKind
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


class FakeRemoteMethod:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def remote(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("collector unreachable")
        self.calls.append((args, kwargs))


class FakeHandle:
    def __init__(self, fail_push=False):
        self.push_trajectories = FakeRemoteMethod(fail=fail_push)


@pytest.fixture(autouse=True)
def clean_sink():
    TrajectoryLifecycle().sink = None
    yield
    TrajectoryLifecycle().sink = None


def _sample(index=7, group=2, versions=("3", "4")):
    return Sample(
        index=index,
        group_index=group,
        weight_versions=[
            WeightVersionsPerCall(spans=[WeightVersionSpan(version, i, i + 1)]) for i, version in enumerate(versions)
        ],
    )


def _pushed(handle):
    return [event for (args, _) in handle.push_trajectories.calls for event in args[0]]


# --------------------------------- emitters ----------------------------------


def test_attempt_and_gen_events_carry_identity_and_version():
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    sample = _sample()
    sink.attempt_start(sample)
    sink.gen_start(sample)
    sink.attempt_end(sample)
    sink.flush()

    kinds = [e.kind for e in _pushed(handle)]
    assert kinds == [
        TrajectoryEventKind.ATTEMPT_START,
        TrajectoryEventKind.GEN_START,
        TrajectoryEventKind.ATTEMPT_END,
    ]
    for event in _pushed(handle):
        assert (event.sample_index, event.group_index) == (7, 2)
        assert event.weight_version == "4"
    assert _pushed(handle)[-1].detail == Sample.Status.PENDING.value


def test_event_without_weight_version_spans_uses_empty_version():
    """A sample whose generation reported no weight version is still emitted, with an empty version."""
    handle = FakeHandle()
    sink = TrajectorySink(handle)

    sink.attempt_start(_sample(versions=()))
    sink.flush()

    [event] = _pushed(handle)
    assert (event.kind, event.sample_index, event.weight_version) == (TrajectoryEventKind.ATTEMPT_START, 7, "")


def test_event_uses_last_span_from_multi_span_generation_call():
    """The reported version is the last span of the last call, not that call's first span."""
    handle = FakeHandle()
    sample = Sample(
        index=7,
        group_index=2,
        weight_versions=[
            WeightVersionsPerCall(spans=[WeightVersionSpan("3", 0, 1)]),
            WeightVersionsPerCall(spans=[WeightVersionSpan("4", 1, 2), WeightVersionSpan("5", 2, 3)]),
        ],
    )
    sink = TrajectorySink(handle)

    sink.gen_start(sample)
    sink.flush()

    [event] = _pushed(handle)
    assert event.weight_version == "5"


def test_spans_use_explicit_timestamps_and_turns():
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    sample = _sample()
    sink.gen_span(sample, 10.0, 14.5, turn=3, detail="120")
    sink.tool_span(sample, 14.5, 16.0, turn=3, detail="2 calls")
    sink.flush()

    events = _pushed(handle)
    assert [(e.kind, e.ts, e.turn) for e in events] == [
        (TrajectoryEventKind.GEN_START, 10.0, 3),
        (TrajectoryEventKind.GEN_END, 14.5, 3),
        (TrajectoryEventKind.TOOL_START, 14.5, 3),
        (TrajectoryEventKind.TOOL_END, 16.0, 3),
    ]
    assert events[1].detail == "120"


def test_attempt_end_drains_lifecycle_metadata_from_turn_samples():
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    turns = []
    for i in range(2):
        sample = _sample()
        sample.metadata["lifecycle"] = dict(t0=100.0 + i, t1=110.0 + i, turn=i + 1)
        turns.append(sample)
    turns[0].metadata["lifecycle"]["t0"] = None  # unknown start: only the end emits
    turns[1].metadata["lifecycle"]["prev_t1"] = 110.0  # session gap = agent/tool work
    turns[1].metadata["lifecycle"]["req_ts"] = 110.6  # server-edge arrival bounds it

    sink.attempt_end(turns)
    sink.flush()
    events = _pushed(handle)
    assert [(e.kind, e.turn) for e in events] == [
        (TrajectoryEventKind.GEN_END, 1),
        (TrajectoryEventKind.TOOL_START, 2),
        (TrajectoryEventKind.TOOL_END, 2),
        (TrajectoryEventKind.GEN_START, 2),
        (TrajectoryEventKind.GEN_END, 2),
        (TrajectoryEventKind.ATTEMPT_END, -1),
    ]
    assert (events[1].ts, events[2].ts) == (110.0, 110.6)  # gap ends at arrival, not gen start
    assert events[1].detail == "agent gap"


def test_batching_and_failure_swallowed():
    handle = FakeHandle()
    sink = TrajectorySink(handle)
    sample = _sample()
    for _ in range(BATCH_MAX_EVENTS - 1):
        sink.gen_start(sample)
    assert handle.push_trajectories.calls == []
    sink.gen_start(sample)  # crosses the batch threshold
    assert len(_pushed(handle)) == BATCH_MAX_EVENTS

    failing = TrajectorySink(FakeHandle(fail_push=True))
    for _ in range(BATCH_MAX_EVENTS + 1):
        failing.gen_start(sample)  # must never raise into the rollout path
    failing.flush()


def test_attach_detach_via_lifecycle_seam():
    assert TrajectoryLifecycle().sink is None
    handle = FakeHandle()
    hooks.attach_trajectory_sink(handle)
    sink = TrajectoryLifecycle().sink
    assert isinstance(sink, TrajectorySink)
    hooks.attach_trajectory_sink(FakeHandle())
    assert TrajectoryLifecycle().sink is sink  # one sink per process

    sink.gen_start(_sample())
    hooks.detach_and_flush()
    assert TrajectoryLifecycle().sink is None
    assert len(_pushed(handle)) == 1  # detach flushed the partial batch


def test_trajectory_events_roundtrip_store(tmp_path):
    from miles.dashboard.store import MetricStore

    writer = MetricStore(tmp_path)
    for ts in (10.0, 3700.0):
        writer.append(
            TrajectoryEvent(
                ts=ts,
                kind=TrajectoryEventKind.GEN_START,
                sample_index=5,
                group_index=-1,
                turn=1,
                weight_version="2",
                detail="",
            )
        )
    writer.flush()
    store = MetricStore.load(tmp_path)
    assert store.has_stream(Stream.TRAJECTORIES)
    assert len((tmp_path / "trajectories").glob("*.jsonl") and list((tmp_path / "trajectories").glob("*.jsonl"))) == 2
    assert [e.ts for e in store.trajectory_events()] == [10.0, 3700.0]
    assert [e.ts for e in store.trajectory_events(t0=3600.0)] == [3700.0]
    assert store.iter_records(Stream.TRAJECTORIES)[0].weight_version == "2"


@pytest.mark.asyncio
async def test_custom_phases_keep_concurrent_episodes_nested_spans_and_errors(tmp_path):
    import asyncio

    from miles.dashboard.store import MetricStore
    from miles.utils.lifecycle import report_trajectory_phase, trajectory_episode, trajectory_phase

    handle = FakeHandle()
    sink = TrajectorySink(handle)
    TrajectoryLifecycle().sink = sink

    async def episode(index):
        sample = _sample(index=index)
        with trajectory_episode(sample):
            with trajectory_phase(sample, "setup"):
                await asyncio.sleep(0)
            with trajectory_phase(sample, "env_step", turn=1):
                with trajectory_phase(sample, "settle", turn=1):
                    await asyncio.sleep(0)
            report_trajectory_phase(sample, "model_call", 50.0, 51.0, turn=2)
            with pytest.raises(ValueError, match="desktop failed"), trajectory_phase(sample, "reset"):
                raise ValueError("desktop failed")

    await asyncio.gather(episode(10), episode(20))
    store = MetricStore(tmp_path)
    for event in _pushed(handle):
        store.append(event)
    store.flush()
    lanes = store.trajectory_lanes()
    assert {lane["sample_index"] for lane in lanes} == {10, 20}
    for lane in lanes:
        assert len(lane["attempts"]) == 1
        assert {s["kind"] for s in lane["segments"]} == {"setup", "env_step", "settle", "model_call", "reset"}
        assert all(s["t0"] is not None and s["t1"] is not None for s in lane["segments"])
        model = next(s for s in lane["segments"] if s["kind"] == "model_call")
        assert (model["t0"], model["t1"], model["turn"]) == (50.0, 51.0, 2)
    palette = store.heatmap("lifecycle", t0=49, t1=52, x_buckets=6)["palette"]
    assert "model_call" in palette


def test_custom_phase_helpers_are_noops_when_disabled_or_evaluating():
    from miles.utils.lifecycle import report_trajectory_phase, trajectory_episode, trajectory_phase

    sample = _sample()
    with trajectory_episode(sample), trajectory_phase(sample, "setup"):
        report_trajectory_phase(sample, "reset", 1.0, 2.0)
    handle = FakeHandle()
    TrajectoryLifecycle().sink = TrajectorySink(handle)
    with trajectory_episode(sample, evaluation=True), trajectory_phase(sample, "env_step", evaluation=True):
        report_trajectory_phase(sample, "reset", 1.0, 2.0, evaluation=True)
    TrajectoryLifecycle().sink.flush()
    assert _pushed(handle) == []


def test_reported_phase_closes_without_waiting_for_another_event():
    from miles.utils.lifecycle import report_trajectory_phase

    handle = FakeHandle()
    TrajectoryLifecycle().sink = TrajectorySink(handle)
    report_trajectory_phase(_sample(), "settle", 10.0, 12.0, turn=1)
    assert [(event.kind, event.ts) for event in _pushed(handle)] == [("phase_start", 10.0), ("phase_end", 12.0)]
