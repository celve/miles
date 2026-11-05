---
title: Flow-GRPO in Miles
---

The rest of these pages describe the standalone [miles_diffusion](https://github.com/radixark/miles_diffusion)
repository. One recipe also runs from Miles itself: Flow-GRPO on SD3.5-medium with the OCR reward, ported from
miles_diffusion at `221f965f`.

```bash
export HF_TOKEN=...  # SD3.5 is gated; the launcher installs CPU PaddleOCR when missing
python scripts/run_sd35_medium_flow_grpo_ocr.py --num-rollout 600
```

`--model-kind diffusion` selects the diffusion path inside Miles' own driver and control plane, and sets the rollout
function, data source and converter hooks; Miles' default rollout and perf panels log the rest, without token metrics:

| Piece | Where |
| --- | --- |
| Engine | `sglang.multimodal_gen`, launched by the inference engine spec (`miles/diffusion/engine.py`) and served through the Miles router |
| Rollout | `miles.diffusion.rollout.rollout_fn.DiffusionRolloutFn`: SDE-sampled trajectories from `/rollout/generate`, scored through `rm_hub` by `--rm-type ocr` (CPU PaddleOCR), `--rm-type pickscore` (CLIP-H PickScore) or a `--custom-rm-path`, per microgroup or per group under `--group-rm` |
| Reward pools | `miles/ray/specs/reward.py`: each built-in reward runs on the workers of a reward pool the rollout executor owns, one worker per cell. A GPU pool's workers take whole GPUs appended to the run's placement group after the rollout GPUs, the way `--eval-num-gpus` is placed; a CPU pool's take none. A colocated pool (`--diffusion-pickscore-placement colocated`) instead shares the first rollout GPUs and sleeps, by the engines' offload rule, while training holds them. The pool client restarts a worker that died and resends its batch. `--rm-type` declares its own pool; `--diffusion-reward-pools` declares the others a prompt's metadata `rm_type` may select. Ray deployments only |
| Train data | `miles.diffusion.conversion`: rewards through the text runs' post-process boundary (`--reward-key`, `--custom-reward-post-process-path`, `--disable-rewards-normalization`), Flow-GRPO advantages, one train pair per SDE step, one row per image in the DP split |
| Trainer | `miles.diffusion.train.actor.DiffusionFSDPTrainRayActor`: one FSDP2 DiT per denoiser component with a PEFT LoRA adapter, Flow-GRPO loss, KL to the LoRA base |
| Weight sync | LoRA A/B pairs over CUDA IPC, merged by the engine (`lora_merge`) |

Supported today: the SD3 and Wan2.2 families, LoRA, colocated GPUs, FSDP, the `sde` sampler and the OCR and PickScore
rewards. Under `--deterministic-mode` a 2-rollout run reproduces miles_diffusion's metric series bit for bit on the same image.
`tests/e2e/fsdp/test_sd35_medium_flow_grpo_ocr_2xGPU.py` checks that series against a recorded standard, with
tolerances that absorb engine, torch and PaddleOCR differences between images; `MILES_DIFFUSION_PARITY_EXACT=1`
makes it exact, the parity check on the recording image.

## Wan2.2-T2V-A14B

Flow-GRPO with LoRA on both experts of Wan2.2-T2V-A14B, 5-frame 480x480 videos, PickScore reward, ported from
miles_diffusion's 5-GPU recipe at `d608b4ec`: four GPUs run the trainer and the engines, and the PickScore reward
pool's one worker runs on a fifth, slot 4 of the placement group.

```bash
python scripts/run_wan22_a14b_flow_grpo_pickscore.py --num-rollout 300
```

The family (`wan2_2`) trains two components: `transformer` denoises timesteps from 875 up, guided by
`--diffusion-guidance-scale`, and `transformer_2` the rest, guided by `--diffusion-guidance-scale-2`. Each rollout draws
one SDE step (`epoch_global_random_choice`), so each trains one expert; `train/log_prob_mean_abs_diff_<component>`
names it. The cross-attention norms are read in fp32 through their own FSDP2 groups. PickScore (`--rm-type pickscore`)
scores every frame and averages them. `tests/e2e/fsdp/test_wan22_a14b_flow_grpo_pickscore_4xGPU.py` runs a 4-GPU proxy
of the recipe that scores PickScore on CPU.
