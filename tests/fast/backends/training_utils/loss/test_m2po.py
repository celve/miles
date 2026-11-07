from __future__ import annotations

import math

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.losses import policy_loss_function
from miles.backends.training_utils.loss.hub.m2po import compute_m2po_mask
from miles.utils.ft_utils.process_group_utils import GroupInfo

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state

register_cpu_ci(est_time=40, suite="stage-b-cpu", labels=["precision"])

_BASE_ARGS = dict(entropy_coef=0.0, kl_coef=0.0, observe_training_entropy=False, true_on_policy_mode=False)


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("m2po") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _algorithm_1(log_ratio: torch.Tensor, advantages: torch.Tensor, tau: float) -> torch.Tensor:
    keep = torch.ones_like(log_ratio)
    trust_region = [
        i
        for i, (r, a) in enumerate(zip(log_ratio.tolist(), advantages.tolist(), strict=True))
        if (a > 0 and r > 0) or (a < 0 and r < 0)
    ]
    squared = [r * r for r in log_ratio.tolist()]
    while trust_region and math.fsum(squared[i] for i in trust_region) / len(trust_region) > tau:
        largest = max(trust_region, key=lambda i: squared[i])
        keep[largest] = 0
        trust_region.remove(largest)
    return keep


@pytest.mark.parametrize("tau", [1e-9, 1e-3, 0.04, 1.0])
@pytest.mark.parametrize("seed", range(10))
def test_mask_matches_algorithm_1(seed, tau):
    g = torch.Generator().manual_seed(seed)
    log_ratio = torch.randn(300, generator=g) * 0.1
    log_ratio[torch.rand(300, generator=g) < 0.03] *= 30
    advantages = torch.randn(300, generator=g)
    advantages[torch.rand(300, generator=g) < 0.2] = 0

    assert torch.equal(compute_m2po_mask(log_ratio, advantages, tau), _algorithm_1(log_ratio, advantages, tau))


def test_mask_leaves_tokens_ppo_could_not_clip():
    log_ratio = torch.tensor([5.0, -5.0, 5.0, -5.0, 0.01, 0.0])
    advantages = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0])

    assert compute_m2po_mask(log_ratio, advantages, tau=0.04).tolist() == [1, 1, 0, 0, 1, 1]


def _run(args, inputs):
    batch = make_batch(inputs, "policy_loss")
    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.backend.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    loss, metrics = policy_loss_function(args, batch, logits, reducer)
    loss.backward()
    return loss.detach(), metrics, logits.grad, reducer


def _inputs(args):
    make_parallel_state().tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    return make_inputs(
        seed=2701, batch_size=3, prompt_lens=[20, 64, 40], response_lens=[10, 48, 32], vocab_size=128, args=args
    )


@pytest.mark.parametrize("use_tis", [False, True])
def test_an_empty_mask_leaves_the_unclipped_surrogate(process_group, use_tis):
    args = make_args(**_BASE_ARGS, use_tis=use_tis)
    inputs = _inputs(args)
    _, clipped, clipped_grad, _ = _run(args, inputs)
    _, _, unclipped_grad, _ = _run(make_args(**_BASE_ARGS, use_tis=use_tis, eps_clip=1e9, eps_clip_high=1e9), inputs)

    _, m2po, m2po_grad, _ = _run(make_args(**_BASE_ARGS, use_tis=use_tis, use_m2po=True, m2po_tau=1e9), inputs)

    torch.testing.assert_close(m2po_grad, unclipped_grad)
    assert not torch.allclose(m2po_grad, clipped_grad)
    assert m2po["pg_clipfrac"].item() == 0.0
    assert clipped["pg_clipfrac"].item() > 0.0
    torch.testing.assert_close(m2po["m2po_ppo_clipfrac"], clipped["pg_clipfrac"])


def test_the_mask_follows_the_rollout_policy_and_zeroes_its_tokens(process_group):
    args = make_args(**_BASE_ARGS, use_m2po=True, m2po_tau=0.04)
    inputs = _inputs(args)
    old = torch.cat(inputs["log_probs"])
    rollout = torch.cat(inputs["rollout_log_probs"])
    advantages = torch.cat(inputs["advantages"])

    loss, metrics, grad, reducer = _run(args, inputs)

    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    current = torch.cat(
        get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
            total_lengths=list(inputs["total_lens"]),
            response_lengths=list(inputs["response_lens"]),
            with_entropy=False,
        )["log_probs"]
    )
    keep = _algorithm_1(current.detach() - rollout, advantages, 0.04)
    expected = reducer(-(current - old).exp() * advantages * keep)
    expected.backward()
    assert 0 < (1 - keep).sum() < keep.numel()
    torch.testing.assert_close(loss, expected.detach())
    torch.testing.assert_close(grad, logits.grad)
    torch.testing.assert_close(metrics["pg_clipfrac"], reducer(1 - keep))
    torch.testing.assert_close(metrics["m2po_log_ratio_sq"], reducer((current.detach() - rollout).square()))
