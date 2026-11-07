# Video-generation agent

A planner model learns, with RL, to turn a user's video request into a good clip from a frozen video generator, which it calls as a tool. This directory holds the environment and two recipes: [Phase 0](phase0/README.md) (single turn, the baseline) and the [multi-turn planner](multiturn/README.md).

## The environment

[`env.py`](env.py) is an agent function for `agentic_tool_call.generate` under session v2. Each episode gives the planner the user's request and two tools:

- `generate(prompt, negative_prompt, seed, seconds)` calls the generator and answers with the clip's id and the remaining budget;
- `submit(clip_id)` names the answer and ends the episode.

The env fixes every clip's resolution and fps, cuts clips to a maximum length, and allows a fixed number of generations and turns per episode, all stated to the planner. A generator outage or timeout raises `InfraAbort("GeneratorUnavailable")`, so the sample is discarded instead of being trained on as a failure.

The env does not score. It records the submitted clip in the sample metadata, and [`reward.py`](reward.py) scores it in the rollout's reward step, once per distinct clip, on the reward pool `--video-reward-pool` names (declared with `--reward-pool-config`; `reward.StubClipScorer` is a CPU scorer for smoke runs). An episode without a submission scores 0, and a pool failure leaves the reward missing, which drops the group.

The generator sits behind the `ClipGenerator` interface in [`generator.py`](generator.py). `FakeClipGenerator` writes a placeholder file per request, for tests and CPU smoke runs.

## Running it

```bash
python examples/experimental/video_gen_agent/make_requests.py prompts.txt /root/datasets/video_requests --heldout-percent 10

export AGENT_VIDEO_GEN_GENERATOR=examples.experimental.video_gen_agent.generator.FakeClipGenerator
export AGENT_VIDEO_GEN_SUBMITTED_DIR=/root/shared_data/video_gen/submitted
# with the usual model, cluster and algorithm args:
--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
--custom-agent-function-path examples.experimental.video_gen_agent.env.run_agent
--custom-rm-path examples.experimental.video_gen_agent.reward.reward_func --video-reward-pool judge --reward-pool-config pools.yaml
--use-session-server v2
--prompt-data /root/datasets/video_requests/train.jsonl --input-key messages
```

`make_requests.py` reads one request per line, such as VBench's `prompts/all_dimension.txt`, and splits them into `train.jsonl` and `heldout.jsonl` by a hash of each request, so the split is the same in every run.

Score without `--group-rm`: under session v2 it hands the reward step a group of episodes, each a list of samples, which neither `reward.py` nor Miles' batched reward setter takes.

The env reads its options from `AGENT_VIDEO_GEN_*` environment variables, listed in `env.py`. `--custom-agent-function-mode worker` and `sandbox` forward them by default; there, the generator's clip directory and the submitted-clip directory must be shared paths that the agent's host can reach.

Clips go back to the planner as text only by default. `AGENT_VIDEO_GEN_CLIP_OBSERVATIONS=1` adds each clip as a `video_url` part, which the session server carries in token space with `--multimodal-token-space-processing`; the multi-turn recipe runs that way.
