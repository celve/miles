# Phase 0: single-turn planner RL with a frozen video generator

A text planner rewrites each video request into one prompt for a frozen video generator, and GRPO trains it on a judge's score of the resulting clip. It is the baseline the multi-turn agent has to beat.

## How a sample is scored

[`reward.py`](reward.py)'s `planner_rm` is the `--custom-rm-path`. It takes the planner's answer (the text after any reasoning block) as the generator prompt, makes one clip through the generator's router with `DiffusionApiClient.generate_clip`, and scores the clip against the user's request with `video_rm` on the judge pool. Every clip of a request uses the same seed, derived from the request, so the samples of a group and the two baselines differ only in their prompt.

- A generator that fails over HTTP, or a clip the judge cannot read or score, gives `None`, and the missing-reward filter drops the group (counted as `rollout/dynamic_filter/drop_group_has_missing_reward`).
- An empty answer scores 0: it is the planner's own failure. The answer is read as text, so the run needs `--rollout-skip-special-tokens`; without it the response keeps its stop token and `planner_rm` stops the run.
- A clip whose frame count or size the generator changed stops the run (`NonRetryableError`), since every later clip would fail the same way.

`planner_rm` works with or without `--group-rm`. The recipe runs without it: Miles' eval refuses `--group-rm`, and the judge scores clip by clip, so a group call batches nothing.

`reward.log_rollout_data`, the `--custom-rollout-log-function-path`, adds `rollout/phase0/generate_seconds` and `score_seconds` (mean per reward call), `empty_answer_ratio` and `generator_prompt_words` to the default rollout metrics. With `perf/rollout_time` and `perf/train_time` they give the step-time split between generation and training.

## Data and baselines

[`make_planner_data.py`](make_planner_data.py) turns a list of requests (VBench's `prompts/all_dimension.txt`) into the planner's chats, split into train and held-out by [`make_requests.py`](../make_requests.py), the multi-turn env's split, so both recipes hold out the same requests. It writes the held-out set twice: `heldout.jsonl` for the planner, and `heldout_raw.jsonl`, whose rows send the request itself to the generator.

Every eval, including the one before training, scores both with greedy decoding: `eval/heldout` is the planner (at step 0 the untrained one) and `eval/heldout_raw` is the raw-request baseline.

## Running it

[`run_qwen3_8b_fastwan.py`](run_qwen3_8b_fastwan.py) trains Qwen3-8B (non-thinking, FSDP) against FastWan2.2-TI2V-5B-FullAttn (832×480, 49 frames at 24 fps, 3 steps) and a Qwen3-VL-8B judge (`VlmJudgeScorer`), on one 8-GPU node: 2 trainer GPUs, 1 planner engine, 3 generator engines and 2 judge workers. A frozen generator cannot share GPUs with training, so the run is not colocated.

```bash
python examples/experimental/video_gen_agent/phase0/run_qwen3_8b_fastwan.py
# a smoke run
python examples/experimental/video_gen_agent/phase0/run_qwen3_8b_fastwan.py --num-rollout 2 --rollout-batch-size 4 --eval-interval 1
```

The launcher downloads the models and VBench's prompts, writes the data under `--data-dir`, and keeps checkpoints, clips and rollout dumps under `--output-dir/<run id>`. `--train-gpus`, `--policy-gpus`, `--generator-gpus` and `--judge-gpus` set the node's split.
