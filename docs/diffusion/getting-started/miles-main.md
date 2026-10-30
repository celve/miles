---
title: Flow-GRPO in Miles
---

The rest of these pages describe the standalone [miles_diffusion](https://github.com/radixark/miles_diffusion)
repository. One recipe also runs from Miles itself: Flow-GRPO on SD3.5-medium with the OCR reward, ported from
miles_diffusion at `221f965f`.

```bash
pip install paddleocr==2.9.1 paddlepaddle==2.6.2 python-Levenshtein
export HF_TOKEN=...  # SD3.5 is gated
python scripts/run_sd35_medium_flow_grpo_ocr.py --num-rollout 600
```

`--model-kind diffusion` selects the diffusion path inside Miles' own driver and control plane:

| Piece | Where |
| --- | --- |
| Engine | `sglang.multimodal_gen`, launched by the inference engine spec (`miles/diffusion/engine.py`) and served through the Miles router |
| Rollout | `miles.diffusion.rollout.rollout_fn.DiffusionRolloutFn`: SDE-sampled trajectories from `/rollout/generate`, scored by CPU PaddleOCR |
| Train data | `miles.diffusion.conversion`: GRPO advantages and one train pair per SDE step, one row per image in the DP split |
| Trainer | `miles.diffusion.train.actor.DiffusionFSDPTrainRayActor`: FSDP2 DiT with a PEFT LoRA adapter, Flow-GRPO loss, KL to the LoRA base |
| Weight sync | LoRA A/B pairs over CUDA IPC, merged by the engine (`lora_merge`) |

Supported today: SD3 family, LoRA, colocated GPUs, FSDP, the `sde` sampler and the OCR reward. Under
`--deterministic-mode` a 2-rollout run reproduces miles_diffusion's metric series bit for bit on the same image;
`tests/e2e/fsdp/test_sd35_medium_flow_grpo_ocr_2xGPU.py` checks it.
