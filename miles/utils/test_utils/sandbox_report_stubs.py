"""An agent function that reports its sandbox, kept apart from ``agent_function_stubs``
so the other ``subproc`` tests do not import the dashboard in every child process."""

import asyncio

from miles.dashboard.hooks import report_sandbox


async def report_sandbox_health(sandbox_key: str, **kwargs) -> str:
    report_sandbox(sandbox_key, provider="local")
    report_sandbox(sandbox_key, status="finished")
    return "reported"


async def report_then_hang(sandbox_key: str, **kwargs) -> None:
    report_sandbox(sandbox_key, provider="local")
    try:
        await asyncio.sleep(3600)
    finally:
        report_sandbox(sandbox_key, status="finished")


async def report_custom_phases(metadata: dict, **kwargs) -> str:
    from miles.utils.lifecycle import TrajectoryIdentity, report_trajectory_phase, trajectory_phase

    identity = TrajectoryIdentity(**metadata["miles_lifecycle"])
    with trajectory_phase(identity, "setup"):
        await asyncio.sleep(0)
    with trajectory_phase(identity, "env_step", turn=1):
        with trajectory_phase(identity, "settle", turn=1):
            await asyncio.sleep(0)
    report_trajectory_phase(identity, "model_call", 10.0, 12.0, turn=2)
    report_sandbox("phase-desktop", provider="local")
    return "reported"
