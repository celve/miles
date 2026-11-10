# Video-generation agent

A planner model learns, with RL, to turn a user's video request into a good clip from a frozen video generator, which it calls as a tool. This directory holds the environment and three recipes: [Phase 0](phase0/README.md) (single turn, the baseline), the [multi-turn planner](multiturn/README.md), and the [keyframe planner](keyframe/README.md) (watched images, then one video).

## The environment

[`env.py`](env.py) is an agent function for `agentic_tool_call.generate` under session v2. Each episode gives the planner the user's request and two tools:

- `generate(prompt, negative_prompt, seed, seconds)` calls the generator and answers with the clip's id and the remaining budget;
- `submit(clip_id)` names the answer and ends the episode.

The env fixes every clip's resolution and fps, cuts clips to a maximum length, and allows a fixed number of generations and turns per episode, all stated to the planner. Two more tools are opt-in. `extend(clip_id, prompt, ...)` (`AGENT_VIDEO_GEN_MAX_EXTENDS`, its own budget beside the generations) continues a clip as a new shot, image-to-video from its last frame, and submitting the last shot submits the shots joined in order as one video. `AGENT_VIDEO_GEN_MIN_SHOTS` makes submit refuse a video of fewer shots. With clip observations on, `AGENT_VIDEO_GEN_KEEP_CLIPS=K` keeps only the last K clips in view: every older clip is folded, in the request the env sends, to a note naming its prompt (`AGENT_VIDEO_GEN_FOLD=text`) or to its last frame (`keyframe`), and `recall(clip_id)` shows it again at its original path. The planner's own turns, its critiques among them, are never folded. Session v2 branches at every fold, and each branch becomes a sample with the episode's reward. A generator outage or timeout raises `InfraAbort("GeneratorUnavailable")`, so the sample is discarded instead of being trained on as a failure.

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

`make_compbench_requests.py` does the same for [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench)'s 1,400 compositional prompts, holding out 200 spread over its seven categories, with each request's category and named subjects in its `metadata`. T2V-CompBench's repository carries no licence, so the keyframe launcher's `--requests t2v-compbench` fetches it at a pinned commit rather than copying it here.

Score without `--group-rm`: under session v2 it hands the reward step a group of episodes, each a list of samples, which neither `reward.py` nor Miles' batched reward setter takes.

The env reads its options from `AGENT_VIDEO_GEN_*` environment variables, listed in `env.py`. `--custom-agent-function-mode worker` and `sandbox` forward them by default; there, the generator's clip directory and the submitted-clip directory must be shared paths that the agent's host can reach.

Clips go back to the planner as text only by default. `AGENT_VIDEO_GEN_CLIP_OBSERVATIONS=1` adds each clip as a `video_url` part, which the session server carries in token space with `--multimodal-token-space-processing`; the multi-turn recipe runs that way.
