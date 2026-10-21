"""The run-wide options of a Harbor CUA rollout, in one place.

Each one is fixed for a whole run -- the turn cap, the sandbox cap, the sample
budget, where the recorder forwards -- so they are launch args rather than
per-request fields, and their defaults live here once so the parser and the
rollout cannot disagree about them.

There is deliberately no image-resize option. The processor settings have to be
identical on both sides of the exactness gate, and only the checkpoint's own
config is, since this side cannot configure the engine's processor.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# A sandbox or conversion failure is not a task failure. Aborting drops the whole GRPO
# group, which is correct statistics; masking keeps the group but lets a zero reward the
# policy never earned move its baseline.
WRITE_OFF_ABORT_GROUP = "abort_group"
WRITE_OFF_MASK_SAMPLE = "mask_sample"
WRITE_OFF_MODES = (WRITE_OFF_ABORT_GROUP, WRITE_OFF_MASK_SAMPLE)

# Resolve one of the router's engines at startup, rather than forwarding to the router.
UPSTREAM_AUTO = "auto"
UPSTREAM_ROUTER = "router"


@dataclass(frozen=True)
class HarborCuaOptions:
    max_turns: int = 12
    max_sandboxes: int = 16
    max_sample_tokens: int | None = None
    call_timeout_s: float = 1800.0
    write_off_mode: str = WRITE_OFF_ABORT_GROUP
    # Measured on the Miles sglang router at sglang 0.5.22: a chat reply comes back
    # without prompt_token_ids, response_token_ids or meta_info, so the recorder has no
    # exact evidence and refuses the call. ``auto`` asks the router for one of its
    # engines and forwards there instead; an explicit URL names one directly.
    upstream: str = UPSTREAM_AUTO
    # Empty runs ``computer-1`` in this process; a URL asks ``osworld_service.py`` on the
    # machine with /dev/kvm to run each OSWorld trial, whose agent then calls the recorder
    # from there, so the recorder must bind an address that machine can reach.
    runner_url: str = ""
    recorder_host: str = "127.0.0.1"
    recorder_advertise_host: str = ""
    history_n: int = 4
    trial_timeout_s: float = 3600.0

    def __post_init__(self) -> None:
        if self.write_off_mode not in WRITE_OFF_MODES:
            raise ValueError(
                f"--harbor-cua-write-off-mode must be one of {WRITE_OFF_MODES}, got {self.write_off_mode!r}"
            )
        if self.max_turns < 1:
            raise ValueError(f"--harbor-cua-max-turns must be positive, got {self.max_turns}")
        if self.max_sandboxes < 1:
            raise ValueError(f"--harbor-cua-max-sandboxes must be positive, got {self.max_sandboxes}")
        if self.runner_url and (self.recorder_advertise_host or self.recorder_host) in ("127.0.0.1", "localhost"):
            raise ValueError(
                "--harbor-cua-runner-url needs a recorder the runner's machine can reach: "
                "set --harbor-cua-recorder-host and --harbor-cua-recorder-advertise-host"
            )

    @classmethod
    def from_args(cls, args) -> HarborCuaOptions:
        return cls(**{f.name: getattr(args, f"harbor_cua_{f.name}", f.default) for f in fields(cls)})

    def router_url(self, args) -> str:
        return f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

    def sample_token_budget(self, args) -> int | None:
        if self.max_sample_tokens is not None:
            return self.max_sample_tokens
        context_len = getattr(args, "rollout_max_context_len", None)
        return int(context_len) if context_len else None


def add_arguments(parser) -> None:
    """Attached to the generate function so ``train.py`` grows these flags."""
    defaults = HarborCuaOptions()
    group = parser.add_argument_group("harbor cua")
    group.add_argument("--harbor-cua-max-turns", type=int, default=defaults.max_turns)
    group.add_argument("--harbor-cua-max-sandboxes", type=int, default=defaults.max_sandboxes)
    group.add_argument("--harbor-cua-max-sample-tokens", type=int, default=defaults.max_sample_tokens)
    group.add_argument("--harbor-cua-call-timeout-s", type=float, default=defaults.call_timeout_s)
    group.add_argument("--harbor-cua-write-off-mode", choices=list(WRITE_OFF_MODES), default=defaults.write_off_mode)
    group.add_argument("--harbor-cua-upstream", type=str, default=defaults.upstream)
    group.add_argument("--harbor-cua-runner-url", type=str, default=defaults.runner_url)
    group.add_argument("--harbor-cua-recorder-host", type=str, default=defaults.recorder_host)
    group.add_argument("--harbor-cua-recorder-advertise-host", type=str, default=defaults.recorder_advertise_host)
    group.add_argument("--harbor-cua-history-n", type=int, default=defaults.history_n)
    group.add_argument("--harbor-cua-trial-timeout-s", type=float, default=defaults.trial_timeout_s)
