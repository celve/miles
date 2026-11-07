# Multi-turn planner RL with clip observations

A Qwen3.5-9B planner turns a video request into a clip with a frozen generator: it writes a prompt, generates, watches the clip, critiques and retries, at most three generations, then submits one. GRPO trains it on a Qwen3-VL judge's score of the submitted clip against the request. It is the multi-turn counterpart of [Phase 0](../phase0/README.md), on the same requests, clips and judge.

## How an episode runs

The episode is [the env](../env.py) under session server v2, reached through [`generate.py`](generate.py), which hands the generator's router to the env and is otherwise `agentic_tool_call.generate`. With `AGENT_VIDEO_GEN_CLIP_OBSERVATIONS=1` each generated clip comes back in the tool reply as a `video_url` part, which the session server carries in token space (`--multimodal-token-space-processing`, `--tito-model qwen35`, the Miles router). The engines parse Qwen3.5's native tool calls with the `qwen35` family's parsers (`qwen3_coder`, `qwen3`); without them a call stays text and the env, which acts on parsed tool calls only, never generates. [`generator.py`](generator.py)'s `RouterClipGenerator` is the env's generator: one `DiffusionApiClient.generate_clip` per `generate` call, its frame count rounded to the 4k+1 the generator takes, an HTTP failure raised as `GeneratorUnavailable`, which discards the episode instead of training on it.

[The env's reward step](../reward.py) scores each episode's submitted clip on the judge pool (`--video-reward-pool judge`); an episode that submits nothing scores 0, and a pool failure leaves the reward missing.

## Running it

[`run_qwen35_9b_fastwan.py`](run_qwen35_9b_fastwan.py) trains Qwen3.5-9B (the plugin multimodal model under raw Megatron, TP 4, vision tower frozen, as in TimeLens) against FastWan2.2-TI2V-5B and a Qwen3-VL-8B judge on one 8-GPU node: 4 trainer GPUs, 1 planner engine with the judge worker beside it on the same GPU, 2 generator engines and 1 eval engine. A frozen generator cannot share GPUs with training, so the run is not colocated. The eval engine loads an HF snapshot the trainer exports at each eval step (`--eval-hf-dir`), so evaluation runs off the training path.

```bash
python examples/experimental/video_gen_agent/multiturn/run_qwen35_9b_fastwan.py
# a smoke run
python examples/experimental/video_gen_agent/multiturn/run_qwen35_9b_fastwan.py --num-rollout 2 --rollout-batch-size 4 --eval-interval 1
# the same run without the shared feature cache, to measure the trainer's media wait without it
python examples/experimental/video_gen_agent/multiturn/run_qwen35_9b_fastwan.py --no-feature-cache
```

The launcher downloads the three models, converts Qwen3.5-9B to `torch_dist`, writes VBench's requests with [`make_requests.py`](../make_requests.py) (Phase 0's split, so both recipes hold out the same requests), and keeps checkpoints, clips, submitted clips and the feature cache under `--output-dir/<run id>`. The engines need an SGLang with the token-space preprocess cache (sgl-project/sglang#47903 → #48305, on `sglang-miles` since 2026-11-06).

Clips are Phase 0's: 832×480, 2 s at 24 fps (49 frames), FastWan's 3-step schedule without CFG, so `eval/heldout` here and Phase 0's `eval/heldout_raw` (the request sent to the generator as it is) share requests, clips settings and judge. The planner watches each clip at 4 fps.

## What to read in a run

- `eval/heldout`: the planner on the held-out requests, greedy; step 0 is the untrained planner.
- `perf/multimodal_rebuild_time`: the trainer's wait for its media in a step, the number to compare with and without `--no-feature-cache`.
- `train/train_rollout_logprob_abs_diff`: the train/rollout log-prob gap, which TimeLens keeps near 0.006.

## Disk use

The feature cache never evicts, so the recipe prunes it: [`feature_cache.py`](feature_cache.py), the rollout log hook, removes every entry written before the rollout just generated, which training has already used (a removed entry reads back as a miss and is computed again). The cache then holds about two rollouts' clips.

- One clip's features at the planner's sampling (832×480, 8 frames at 4 fps) are `pixel_values_videos` of 6240 × 1536 float32, 36.6 MiB, for 1560 visual tokens (measured with Qwen3.5's video processor).
- A step makes at most 32 × 8 episodes × 3 clips = 768 clips, 27.4 GiB of features; an eval at most 95 × 3 = 285 clips, 10.2 GiB.
- Budget the cache directory for two steps and one eval in flight: about 65 GiB. Without the prune a 200-step run would need up to 153,600 clips' features, 5.4 TiB.
- The clips themselves (`clips/`, mp4, under 1 MB each) and the submitted copies (`submitted/`) are kept for the whole run: at most 160 k files, under 160 GB.
