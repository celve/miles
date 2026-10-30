"""Trajectory-lifecycle seam callable from core rollout code.

Core generation paths read ``TrajectoryLifecycle().sink`` unconditionally, so
this module must never import from the dashboard extra. The dashboard installs
the sink when enabled; while it is ``None`` every probe site is a no-op.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

from miles.rollout.agentic.agent_function import report_to_caller
from miles.utils.misc import SingletonMeta


class TrajectoryLifecycle(metaclass=SingletonMeta):
    def __init__(self):
        self.sink = None  # the dashboard TrajectorySink (duck-typed); None = off


def attach_lifecycle_metadata(sample, record, prev_record, turn: int) -> None:
    """Fold one chat record's timing onto its sample via
    ``metadata["lifecycle"]`` so it rides the Sample across the session
    boundary. ``timestamp`` is the segment end; the start is the engine's
    ``e2e_latency`` when reported."""
    meta_info = record.response["choices"][0].get("meta_info", {})
    latency = meta_info.get("e2e_latency")
    t1 = record.timestamp
    t0 = t1 - latency if latency is not None else None
    segment = dict(t0=t0, t1=t1, turn=turn)
    if record.request_timestamp is not None:
        segment["req_ts"] = record.request_timestamp  # server-edge arrival bounds the agent gap
    if prev_record is not None:
        segment["prev_t1"] = prev_record.timestamp  # gap since the previous call end = agent-side work
    sample.metadata["lifecycle"] = segment


@dataclass(frozen=True)
class TrajectoryIdentity:
    sample_index: int
    group_index: int = -1
    weight_version: str = ""

    @classmethod
    def from_sample(cls, sample):
        versions = [span.version for span in sample.all_weight_version_spans]
        return cls(
            (
                sample.lineage.source_sample_index
                if sample.lineage is not None
                else sample.index if sample.index is not None else -1
            ),
            sample.group_index if sample.group_index is not None else -1,
            str(versions[-1]) if versions else "",
        )


@dataclass(frozen=True)
class TrajectoryPhaseReport:
    identity: TrajectoryIdentity
    name: str
    boundary: str
    ts: float
    turn: int


def _report_phase(sample, name: str, boundary: str, *, turn: int, ts: float | None, evaluation: bool):
    if evaluation:
        return
    try:
        identity = sample if isinstance(sample, TrajectoryIdentity) else TrajectoryIdentity.from_sample(sample)
        report = TrajectoryPhaseReport(identity, name, boundary, time.time() if ts is None else ts, turn)
        if report_to_caller(report):
            return
        sink = TrajectoryLifecycle().sink
        if sink is not None:
            sink.phase_report(report)
    except Exception:
        logging.getLogger(__name__).warning("trajectory phase report failed", exc_info=True)


@contextmanager
def trajectory_phase(sample, name: str, *, turn: int = -1, evaluation: bool = False):
    """Time one phase using a Sample or the dispatcher-provided TrajectoryIdentity in a child agent."""
    _report_phase(sample, name, "start", turn=turn, ts=None, evaluation=evaluation)
    try:
        yield
    finally:
        _report_phase(sample, name, "end", turn=turn, ts=None, evaluation=evaluation)


def report_trajectory_phase(sample, name: str, t0: float, t1: float, *, turn: int = -1, evaluation: bool = False):
    """Report an externally measured phase using wall-clock timestamps in seconds."""
    _report_phase(sample, name, "start", turn=turn, ts=t0, evaluation=evaluation)
    _report_phase(sample, name, "end", turn=turn, ts=t1, evaluation=evaluation)


@contextmanager
def trajectory_episode(sample, *, evaluation: bool = False):
    """Record an episode in a custom rollout function that bypasses Miles's per-sample dispatcher."""
    sink = None if evaluation else TrajectoryLifecycle().sink
    if sink is not None:
        sink.attempt_start(sample)
    try:
        yield
    finally:
        if sink is not None:
            sink.attempt_end(sample)
