"""The interface a diffusion RL algorithm implements across the rollout and conversion stages; its loss is the
trainer's, picked by the same `--loss-type` in `miles.backends.fsdp_utils.diffusion.losses`."""

import abc
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from miles.utils.types import Sample


class DiffusionAlgorithm(abc.ABC):
    # (what is unsupported, whether this run asks for it), checked with the run-wide list at startup.
    unsupported: list[tuple[str, Callable[[Namespace], bool]]] = []
    # Flags the algorithm never reads; each must keep its default.
    unread_flags: tuple[str, ...] = ()
    # Whether the trainer keeps an EMA of the adapter, which the engines then sample with.
    uses_ema: bool = False

    @abc.abstractmethod
    def rollout_request(self, sde_steps: list[int] | None, returned_latents: list[int] | None) -> dict[str, Any]:
        """Per-request fields of ``POST /rollout/generate`` from the step strategy's choice."""

    @abc.abstractmethod
    def build_train_pairs(self, args: Namespace, sample: Sample, *, advantage: float) -> list[dict[str, Any]]:
        """One image's train pairs, in the order the tile schedule takes them."""
