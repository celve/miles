"""Exceptions an agent-function call raises wherever it runs: on the rollout loop, in a child process, or on a worker."""


class InfraAbort(Exception):
    """Raised by an agent function to discard its sample, for a failure the policy cannot have caused."""

    def __init__(self, exit_status: str, message: str | None = None):
        super().__init__(message or exit_status)
        self.exit_status = exit_status


class RemoteTraceback(Exception):
    """The agent process's traceback, chained under the error it raised, as concurrent.futures does."""

    def __init__(self, tb: str):
        super().__init__(tb)
        self.tb = tb

    def __str__(self) -> str:
        return self.tb
