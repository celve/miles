from argparse import Namespace

import pytest
import torch

from miles.backends.fsdp_utils.lr_scheduler import get_lr_scheduler


def _scheduler(*, num_rollout: int, rollouts_per_round: int, global_batch_size: int, decay_style: str = "cosine"):
    args = Namespace(
        num_rollout=num_rollout,
        rollout_batch_size=rollouts_per_round,
        n_samples_per_prompt=1,
        global_batch_size=global_batch_size,
        lr=1.0,
        min_lr=0.0,
        lr_warmup_init=0.0,
        lr_warmup_iters=0,
        lr_warmup_fraction=None,
        lr_decay_iters=None,
        lr_decay_style=decay_style,
        lr_wsd_decay_iters=None,
        lr_wsd_decay_style=None,
        use_checkpoint_lr_scheduler=False,
        override_lr_scheduler=False,
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    return args, optimizer, get_lr_scheduler(args, optimizer)


def test_rounds_of_multi_sample_rollouts_decay_over_one_step_per_round():
    # 4 rollouts per round cut into 8 samples: the training-side split takes one step whose global batch is 8
    args, optimizer, scheduler = _scheduler(num_rollout=4, rollouts_per_round=4, global_batch_size=8)
    rates = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert args.train_iters == 4
    assert rates[:3] == sorted(rates[:3], reverse=True) and min(rates[:3]) > 0
    assert rates[3] == pytest.approx(0.0, abs=1e-12)


def test_one_round_of_multi_sample_rollouts_builds_a_scheduler():
    args, _, _ = _scheduler(num_rollout=1, rollouts_per_round=4, global_batch_size=8, decay_style="constant")

    assert args.train_iters == 1


def test_a_global_batch_dividing_the_round_keeps_its_step_count():
    args, _, _ = _scheduler(num_rollout=3, rollouts_per_round=8, global_batch_size=4)

    assert args.train_iters == 6
