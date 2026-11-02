"""The interface a diffusion RL algorithm implements across the rollout, conversion and train stages."""

import abc
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch

from miles.utils.types import Sample


class DiffusionAlgorithm(abc.ABC):
    # (what is unsupported, whether this run asks for it), checked with the run-wide list at startup.
    unsupported: list[tuple[str, Callable[[Namespace], bool]]] = []
    # Flags the algorithm never reads; each must keep its default.
    unread_flags: tuple[str, ...] = ()

    @abc.abstractmethod
    def rollout_request(self, sde_steps: list[int] | None, returned_latents: list[int] | None) -> dict[str, Any]:
        """Per-request fields of ``POST /rollout/generate`` from the step strategy's choice."""

    @abc.abstractmethod
    def build_train_pairs(self, args: Namespace, sample: Sample, *, advantage: float) -> list[dict[str, Any]]:
        """One image's train pairs, in the order the tile schedule takes them."""

    @abc.abstractmethod
    def loss(self, actor, pairs: list[dict], *, metrics) -> torch.Tensor:
        """Summed loss of one micro-batch; the actor normalizes by the optimizer step's pair count."""
