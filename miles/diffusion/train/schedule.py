"""Micro-batch schedule of one rank's Flow-GRPO train pairs: image x SDE-step tiles."""


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
