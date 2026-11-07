# Video reward

Scores clips a frozen video generator wrote against the user's request, on the run's reward pools, from a text-policy run. Three scorers read the clip from its file:

| Scorer | Model | Score |
| --- | --- | --- |
| `miles.rollout.rm_hub.video.videoalign.VideoAlignScorer` | [VideoAlign](https://github.com/KwaiVGI/VideoAlign)'s VideoReward (Qwen2-VL-2B) | sum of its VQ, MQ and TA heads, each normalized by the checkpoint |
| `miles.rollout.rm_hub.video.vlm_judge.VlmJudgeScorer` | a Qwen-VL model, Qwen3-VL-8B-Instruct by default | prompt adherence, visual quality and motion, each the expected 1–5 answer mapped to [0, 1], averaged |
| `miles.rollout.rm_hub.video.pickscore.VideoPickScoreScorer` | PickScore | mean over 8 evenly spaced frames |

A clip a scorer cannot read scores `None`, and the missing-reward filter drops its group.

## In a run

Declare the pools with `--reward-pool-config pools.yaml` (any run, not only `--model-kind diffusion`). Each GPU pool takes one GPU per worker after the rollout GPUs, or with `colocated: true` shares the first rollout GPUs and sleeps while training holds them.

Score with `--custom-rm-path miles.rollout.rm_hub.video.reward.video_rm --video-reward-pool <name>`, with or without `--group-rm`. Whatever generates the clip records it in the sample's metadata as `video_clip_path`, with the user's request as `video_request`: pass the request, never the planner's generator prompt, so a judge cannot score the planner's words. A sample with no clip scores `None`.

A custom reward function can call a pool itself: it receives every pool as a `<name>_pool` keyword argument, and `await videoalign_pool.score(clip_paths, requests)` returns one score per clip.

## Calibration

`build_pairs.py` turns two generated clips per request into a set of A/B pairs, including reward-hacking pairs whose answer is known (a frozen clip, another request's clip with this request's text drawn over it, another request's clip). `evaluate.py` runs each scorer of a pools file over the set and reports, per scorer, agreement with human labels, agreement on the known pairs, and seconds per clip:

```bash
python build_pairs.py --requests requests.jsonl --clips clips/ --out calib/
python evaluate.py --pools pools.yaml --pairs calib/pairs.jsonl --clips clips/ --labels labels.json --scores calib/scores.json
```
