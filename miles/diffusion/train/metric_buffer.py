"""Per-optimizer-step scalar metrics of the Flow-GRPO loop, reduced over DP item-weighted."""

from enum import Enum, auto

import torch
import torch.distributed as dist


class MetricReduce(Enum):
    MEAN = auto()
    REPLICATED = auto()


SCHEMA = {
    "loss": MetricReduce.MEAN,
    "policy_loss": MetricReduce.MEAN,
    "kl_loss": MetricReduce.MEAN,
    "loss_abs_mean": MetricReduce.MEAN,
    "adv_abs_mean": MetricReduce.MEAN,
    "ratio_abs_minus_1": MetricReduce.MEAN,
    "approx_kl": MetricReduce.MEAN,
    "clipfrac": MetricReduce.MEAN,
    "log_prob_new_idx_0": MetricReduce.MEAN,
    "log_prob_old_idx_0": MetricReduce.MEAN,
    "log_prob_mean_abs_diff": MetricReduce.MEAN,
    "grad_norm": MetricReduce.REPLICATED,
}


class MetricBuffer:
    """MEAN takes a local sum and the item count behind it, so the reduction weights items, not ranks.
    The schema fixes the packing order every rank agrees on."""

    def __init__(self, *, group, device: torch.device) -> None:
        self._group = group
        self._device = device
        self._mean_keys = sorted(key for key, op in SCHEMA.items() if op is MetricReduce.MEAN)
        self._sums = {key: self._scalar(0.0) for key in self._mean_keys}
        self._counts = dict.fromkeys(self._mean_keys, 0.0)
        self._replicated: dict[str, torch.Tensor] = {}

    def _scalar(self, value: torch.Tensor | float) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            return torch.tensor(value, device=self._device, dtype=torch.float64)
        return value.detach().reshape(()).to(device=self._device, dtype=torch.float64)

    def emit_mean(self, key: str, total: torch.Tensor, count: int) -> None:
        assert SCHEMA[key] is MetricReduce.MEAN, key
        self._sums[key] = self._sums[key] + self._scalar(total)
        self._counts[key] += float(count)

    def emit_replicated(self, key: str, value: torch.Tensor) -> None:
        assert SCHEMA[key] is MetricReduce.REPLICATED, key
        self._replicated[key] = self._scalar(value)

    def reduce(self) -> dict[str, float]:
        """Every rank in the group must call this."""
        counts = torch.tensor([self._counts[key] for key in self._mean_keys], device=self._device, dtype=torch.float64)
        packed = torch.cat([torch.stack([self._sums[key] for key in self._mean_keys]), counts])
        if dist.get_world_size(self._group) > 1:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self._group)
        totals = packed.tolist()
        num_keys = len(self._mean_keys)
        out = {key: totals[i] / totals[num_keys + i] for i, key in enumerate(self._mean_keys) if totals[num_keys + i]}
        out.update({key: float(value) for key, value in self._replicated.items()})
        return out
