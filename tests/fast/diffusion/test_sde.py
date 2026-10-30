"""The trainer scores a rollout transition exactly as flow_grpo's `sde_step_with_logprob` does.

    rollout grid (one image)      sigmas = [1.0, s1, s2, ..., 0.0]      sigma_max = s1
    train pair at step i          sigma = sigmas[i], next_sigma = sigmas[i + 1]
"""

import math

import torch

from miles.diffusion.train.sde import sde_step_log_prob


def _flow_grpo_sde_step_with_logprob(sigmas, step, model_output, sample, prev_sample, noise_level):
    view = (-1, *([1] * (sample.ndim - 1)))
    sigma = sigmas[step].view(view)
    sigma_prev = sigmas[step + 1].view(view)
    sigma_max = sigmas[1].item()
    dt = sigma_prev - sigma
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
    mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )
    log_prob = (
        -((prev_sample - mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim))), mean, std_dev_t


def _shifted_sigmas(shift=3.0, num_steps=10):
    base = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
    return torch.cat([shift * base / (1 + (shift - 1) * base), torch.zeros(1)])


def test_matches_flow_grpo_bitwise_including_sigma_one_and_the_last_step():
    torch.manual_seed(0)
    sigmas = _shifted_sigmas()
    step = torch.tensor([0, 4, 9])
    v, x, nxt = (torch.randn(3, 4, 6) for _ in range(3))

    got = sde_step_log_prob(
        v,
        x,
        prev_sample=nxt,
        noise_level=0.7,
        sigmas=sigmas[step],
        next_sigmas=sigmas[step + 1],
        sigma_max=sigmas[1].expand(3),
    )
    want = _flow_grpo_sde_step_with_logprob(sigmas, step, v, x, nxt, 0.7)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g, w, rtol=0.0, atol=0.0)


def test_a_first_sigma_within_isclose_of_one_takes_sigma_max():
    sigmas = torch.tensor([0.999993, 0.8, 0.5, 0.0])
    v, x, nxt = (torch.ones(1, 2) for _ in range(3))

    _, _, std_dev_t = sde_step_log_prob(
        v, x, prev_sample=nxt, noise_level=1.0, sigmas=sigmas[:1], next_sigmas=sigmas[1:2], sigma_max=sigmas[1:2]
    )

    torch.testing.assert_close(std_dev_t.flatten(), torch.sqrt(sigmas[:1] / (1 - sigmas[1])))
