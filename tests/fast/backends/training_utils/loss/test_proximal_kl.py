from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.losses import policy_loss_function
from miles.utils.ft_utils.process_group_utils import GroupInfo

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("proximal-kl") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _setup(proximal_kl_coef):
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    args = make_args(
        advantage_estimator="grpo",
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        use_tis=True,
        proximal_kl_coef=proximal_kl_coef,
    )
    inputs = make_inputs(
        seed=7, batch_size=3, prompt_lens=[20, 64, 40], response_lens=[10, 48, 32], vocab_size=128, args=args
    )
    return args, inputs


def _current_log_probs(args, inputs):
    return [
        log_prob.detach()
        for log_prob in get_log_probs_and_entropy(
            deep_clone(inputs["policy_logits"]),
            args=args,
            unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
            total_lengths=list(inputs["total_lens"]),
            response_lengths=list(inputs["response_lens"]),
            with_entropy=False,
        )["log_probs"]
    ]


def _run(args, batch, inputs):
    logits = deep_clone(inputs["policy_logits"]).requires_grad_(True)
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    loss, metrics = policy_loss_function(args, batch, logits, reducer)
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


def test_the_anchor_is_inert_at_a_batchs_first_step(process_group):
    args, inputs = _setup(proximal_kl_coef=1.0)
    batch = make_batch(inputs, "policy_loss")
    batch["log_probs"] = _current_log_probs(args, inputs)
    anchored = _run(args, batch, inputs)

    args.proximal_kl_coef = 0.0
    plain = _run(args, batch, inputs)

    assert anchored[1]["proximal_kl"].item() == pytest.approx(0.0, abs=1e-12)
    torch.testing.assert_close(anchored[0], plain[0])
    torch.testing.assert_close(anchored[2], plain[2])
    assert "proximal_kl" not in plain[1]


def test_the_anchor_adds_its_kl_once_the_policy_has_moved(process_group):
    args, inputs = _setup(proximal_kl_coef=0.5)
    batch = make_batch(inputs, "policy_loss")
    batch["log_probs"] = [lp + 0.1 * torch.randn_like(lp) for lp in _current_log_probs(args, inputs)]
    anchored_loss, metrics, anchored_grad = _run(args, batch, inputs)

    args.proximal_kl_coef = 0.0
    plain_loss, _, plain_grad = _run(args, batch, inputs)

    assert metrics["proximal_kl"].item() > 0
    torch.testing.assert_close(anchored_loss - plain_loss, 0.5 * metrics["proximal_kl"])
    assert not torch.allclose(anchored_grad, plain_grad)
