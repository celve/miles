# Keyframe planner RL: watched images, then one video

A Qwen3.5-9B (or Qwen3.8-27B) planner turns a video request into one video in two steps. It calls `generate_image` up to three times and sees each image; then it calls `generate_video` once with the id of the image it picks and a prompt for what moves and how the camera moves. That clip is the submission and ends the episode; the planner never watches it. GRPO trains the planner on a Qwen3-VL judge's score of the clip against the request, through the same [reward step](../reward.py) as the [multi-turn recipe](../multiturn/README.md).

## How an episode runs

The episode is [the keyframe env](env.py) under session server v2, reached through [`generate.py`](generate.py), which hands the env both generators' routers (`--video-generator-model video`, `--image-generator-model image`) and is otherwise `agentic_tool_call.generate`. Each image comes back in the tool reply as an `image_url` part, which the session server carries in token space. [`generator.py`](generator.py)'s `RouterImageGenerator` makes images through `DiffusionApiClient.generate_image`, which stores each as `<sha256>.png`; the video is the multi-turn recipe's `RouterClipGenerator`. A generator outage or timeout discards the group, as does any other env fault; an episode that ends without a video, out of turns or cut at the token limit, scores 0. The env's options are in [`env.py`](env.py).

`--video-mode preview` (the default) makes the clip text-to-video from the chosen image's prompt followed by the motion prompt, so the images are cheap previews and the clip need not look like the one picked. `--video-mode i2v` makes it image-to-video from the chosen image; [LIN-2809](https://github.com/softlash/LIN-2809/blob/main/docs/report.md) measured those clips moving far less than text-to-video ones at the same cost. `--video-mode reference` makes the images assets: the planner makes one image per subject or object, each isolated on a plain background (no setting or whole-scene images), then `generate_video` takes `image_ids`, the watched assets, and a prompt that places them in a setting it describes. Wan2.1-VACE-14B makes a clip that shows the assets without starting from any of them, from that prompt alone. A setting image as a reference made clips flicker and lose subjects ([LIN-2830's asset measurement](https://github.com/softlash/LIN-2830/blob/main/docs/assets/README.md)), and on T2V-CompBench the untrained planner makes about two isolated assets an episode. Reference mode's image budget is 4. [LIN-2830](https://github.com/softlash/LIN-2830/blob/main/docs/measure/README.md) measured it at conditioning scale 0.5 (`--reference-scale`): it holds the image far better than preview mode and scores the same on the reward and its motion gate, at twice T2V's GPU time a clip. sglang serves VACE through its diffusers backend, from a directory `prepare` builds beside the checkpoint with [`reference_model.py`](reference_model.py), whose [pipeline](reference_pipeline.py) fuses lightx2v's Wan2.1 4-step LoRA at load. Each episode records its video mode and generator names, so a comparison never mixes them.

## Running it

[`run_qwen35_9b_keyframe.py`](run_qwen35_9b_keyframe.py) runs on one node, not colocated, every engine on one GPU; in `reference` mode the 9B's 8-GPU layout moves an image GPU to video (1/1/1/4/1), since VACE is the bottleneck:

| Planner | GPUs | Trainer | Planner engine + judge | Image engines | Video engines | Eval engine |
|---|---|---|---|---|---|---|
| Qwen3.5-9B (default) | 4 | 1 | 1 | 1 | 1 | none: eval runs on the planner engine |
| Qwen3.5-9B (default) | 8 | 1 | 1 | 2 | 3 | 1, on exported snapshots |
| Qwen3.8-27B (`--planner qwen3.8-27b`) | 8 | 4, at TP4 with Adam's state in host memory | 1 | 1 | 2 | none: eval runs on the planner engine |

The 27B renders with `--tito-model qwen38small`, whose image turns the session server counts like Qwen3.5's.

```bash
python examples/experimental/video_gen_agent/keyframe/run_qwen35_9b_keyframe.py --num-gpus-per-node 4
# a smoke run
python examples/experimental/video_gen_agent/keyframe/run_qwen35_9b_keyframe.py --num-gpus-per-node 4 --num-rollout 2 --rollout-batch-size 2 --eval-interval 1
# the Qwen3.8-27B planner
python examples/experimental/video_gen_agent/keyframe/run_qwen35_9b_keyframe.py --planner qwen3.8-27b
# another split, image model or video mode
python examples/experimental/video_gen_agent/keyframe/run_qwen35_9b_keyframe.py --image-gpus 3 --video-gpus 2 --image-model Qwen/Qwen-Image-2.1-Turbo --video-mode i2v
```

The generators are frozen diffusion model groups. Images: Z-Image-Turbo, 9 steps without CFG (or Qwen-Image-2.1-Turbo, 8 steps at CFG 1), at the video's size (832×480). Video: Wan2.2-T2V-A14B (I2V-A14B in `i2v` mode, Wan2.1-VACE-14B in `reference` mode), 2 s at 16 fps (33 frames), with lightx2v's 4-step distill LoRAs and no CFG. Each A14B expert takes its own LoRA, so the video group lists both under `loras` in `--sglang-config`, and Miles merges them into each engine through `/v1/set_lora` before the engine joins its router.

The launcher downloads the models and the mode's two LoRAs, converts the planner to `torch_dist`, writes VBench's requests with [`make_requests.py`](../make_requests.py) (the held-out split Phase 0 and the multi-turn recipe use), and keeps checkpoints, images, clips and submitted clips under `--output-dir/<run id>`.

## What to read in a run

The multi-turn recipe's [episode metrics](../multiturn/episode_metrics.py) apply, with `episode/images` in place of folding counts: `eval/heldout` for the held-out mean episode reward (step 0 is the untrained planner), `rollout/episode_raw_reward` in training, and `episode/prompt_tokens_per_turn`, `episode/peak_prompt_tokens` and `episode/turn_seconds`.
