"""Micro-batch schedule of one rank's Flow-GRPO train pairs: image x SDE-step tiles."""

import torch
import torch.distributed as dist


def tile_train_pairs(
    rows: list[list[dict]], *, num_optim_steps: int, sample_microbatch: int, tstep_microbatch: int
) -> list[list[list[dict]]]:
    """``steps[step][micro_batch]`` lists of pairs.

    ``rows`` holds each image's pairs in step order. Each optimizer step takes the next block of images; its
    micro-batches are tiles of `sample_microbatch` images x `tstep_microbatch` steps, the step chunk outermost,
    each tile flattened image-major.
    """
    window = len(rows[0])
    assert all(len(row) == window for row in rows), "every image must contribute the same number of SDE steps"
    assert len(rows) % num_optim_steps == 0, f"{len(rows)} images do not split into {num_optim_steps} steps"
    images_per_step = len(rows) // num_optim_steps
    sample_mb = min(sample_microbatch, images_per_step)
    tstep_mb = min(tstep_microbatch, window)
    assert (
        images_per_step % sample_mb == 0 and window % tstep_mb == 0
    ), f"tiles must be uniform: {images_per_step} images per step / {sample_mb}, {window} steps / {tstep_mb}"

    steps = []
    for step in range(num_optim_steps):
        step_rows = rows[step * images_per_step : (step + 1) * images_per_step]
        steps.append(
            [
                [row[t] for row in step_rows[s : s + sample_mb] for t in range(t0, t0 + tstep_mb)]
                for t0 in range(0, window, tstep_mb)
                for s in range(0, images_per_step, sample_mb)
            ]
        )
    return steps


def check_same_component_on_every_rank(
    component: str, *, components: tuple[str, ...], group, device: torch.device
) -> None:
    """Every DP rank must forward the same component at each micro-batch: the components share one FSDP mesh, so a
    mismatch would pair one expert's collectives with another's."""
    local = torch.tensor([components.index(component)], device=device)
    gathered = torch.empty(dist.get_world_size(group), dtype=local.dtype, device=device)
    dist.all_gather_into_tensor(gathered, local, group=group)
    if len(set(gathered.tolist())) > 1:
        raise RuntimeError(
            f"DP ranks forward different components at one micro-batch: {[components[i] for i in gathered.tolist()]}; "
            "every rank's rollout must draw the same SDE steps (epoch_global_random_choice)"
        )
