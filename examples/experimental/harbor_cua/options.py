"""The run-wide options of a Harbor CUA rollout, in one place.

Each one is fixed for a whole run -- the turn cap, the sandbox cap, the sample
budget, the image resize settings -- so they are launch args rather than
per-request fields, and their defaults live here once so the parser and the
rollout cannot disagree about them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# A sandbox or conversion failure is not a task failure. Aborting drops the whole GRPO
# group, which is correct statistics; masking keeps the group but lets a zero reward the
# policy never earned move its baseline.
WRITE_OFF_ABORT_GROUP = "abort_group"
WRITE_OFF_MASK_SAMPLE = "mask_sample"
WRITE_OFF_MODES = (WRITE_OFF_ABORT_GROUP, WRITE_OFF_MASK_SAMPLE)


@dataclass(frozen=True)
class HarborCuaOptions:
    max_turns: int = 12
    max_sandboxes: int = 16
    max_sample_tokens: int | None = None
    call_timeout_s: float = 1800.0
    write_off_mode: str = WRITE_OFF_ABORT_GROUP
    min_pixels: int | None = None
    max_pixels: int | None = None
    # Measured on the Miles sglang router at 0.5.22: a chat reply comes back without
    # prompt_token_ids, response_token_ids or meta_info, so the recorder has no exact
    # evidence and refuses the call. Naming an engine's own URL here restores it.
    upstream: str = ""

    def __post_init__(self) -> None:
        if self.write_off_mode not in WRITE_OFF_MODES:
            raise ValueError(f"--harbor-cua-write-off-mode must be one of {WRITE_OFF_MODES}, got {self.write_off_mode!r}")
        if self.max_turns < 1:
            raise ValueError(f"--harbor-cua-max-turns must be positive, got {self.max_turns}")
        if self.max_sandboxes < 1:
            raise ValueError(f"--harbor-cua-max-sandboxes must be positive, got {self.max_sandboxes}")

    @classmethod
    def from_args(cls, args) -> HarborCuaOptions:
        return cls(**{f.name: getattr(args, f"harbor_cua_{f.name}", f.default) for f in fields(cls)})

    @property
    def resize_settings(self) -> dict[str, int] | None:
        """Pinned image-processor settings, recorded with every sample's fingerprint."""
        settings = {
            name: value
            for name, value in (("min_pixels", self.min_pixels), ("max_pixels", self.max_pixels))
            if value is not None
        }
        return settings or None

    def engine_url(self, args) -> str:
        """Where the recorder forwards: an explicit engine, else the router."""
        return self.upstream or f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

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
    group.add_argument(
        "--harbor-cua-write-off-mode", choices=list(WRITE_OFF_MODES), default=defaults.write_off_mode
    )
    group.add_argument("--harbor-cua-upstream", type=str, default=defaults.upstream)
    group.add_argument("--harbor-cua-min-pixels", type=int, default=defaults.min_pixels)
    group.add_argument("--harbor-cua-max-pixels", type=int, default=defaults.max_pixels)
