"""Run real ``computer-1`` trials through the generate function, without a trainer.

This is the rollout-only check: a real sandbox, a real desktop, a real policy,
and the samples the trainer would have been handed -- but no optimizer step, so
a conversion fault shows up before any GPU time goes into training.

    python -m examples.experimental.harbor_cua.check_trial_live \\
        --model /personal/lin2045/models/Qwen3-VL-4B-Instruct \\
        --router-port 30000 --instances cua-basics--click-target--red --repeat 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from argparse import Namespace
from types import SimpleNamespace

from examples.experimental.harbor_cua.rollout import generate
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample


def build_state(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        args=Namespace(
            partial_rollout=False,
            sglang_router_ip=args.router_ip,
            sglang_router_port=args.router_port,
            rollout_max_context_len=args.max_context_len,
            harbor_cua_max_turns=args.max_turns,
            harbor_cua_max_sandboxes=args.max_sandboxes,
            harbor_cua_max_pixels=args.max_pixels,
            harbor_cua_write_off_mode=args.write_off_mode,
        ),
        processor=load_processor(args.model, trust_remote_code=True),
        tokenizer=load_tokenizer(args.model, trust_remote_code=True),
    )


async def one_episode(state, instance_id: str, index: int, temperature: float) -> dict:
    sample = Sample(index=index, metadata={"instance_id": instance_id, "agent_name": "computer-1"})
    fn_input = SimpleNamespace(
        state=state,
        sample=sample,
        sampling_params={"temperature": temperature, "max_new_tokens": 512},
        evaluation=False,
        args=state.args,
    )
    output = await generate(fn_input)
    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    for produced in samples:
        produced.validate()
    first = samples[0]
    return {
        "instance_id": instance_id,
        "index": index,
        "reward": first.reward,
        "status": str(first.status),
        "samples": len(samples),
        "rollout_ids": sorted({s.rollout_id for s in samples}),
        "response_lengths": [s.response_length for s in samples],
        "trained_tokens": [sum(s.loss_mask) for s in samples],
        "image_rows": [int(s.multimodal_train_inputs["image_grid_thw"].shape[0]) for s in samples],
        "metadata": {
            key: first.metadata.get(key)
            for key in (
                "exit_status",
                "harbor_episodes",
                "recorded_action_calls",
                "auxiliary_calls_excluded",
                "segments",
                "kept_segments",
                "dropped_text_only_segments",
                "image_occurrences",
                "distinct_images",
                "boundary_reasons",
                "rejected_calls",
                "write_off_reason",
                "processor_fingerprint",
            )
        },
    }


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    state = build_state(args)
    jobs = [
        one_episode(state, instance_id, index, args.temperature)
        for index, instance_id in enumerate(args.instances * args.repeat)
    ]
    results = await asyncio.gather(*jobs, return_exceptions=True)

    report = []
    failures = 0
    for result in results:
        if isinstance(result, BaseException):
            failures += 1
            report.append({"error": f"{type(result).__name__}: {result}"})
            continue
        report.append(result)

    rewards = [row["reward"] for row in report if isinstance(row.get("reward"), float)]
    summary = {
        "episodes": len(report),
        "errors": failures,
        "rewards": rewards,
        "reward_spread": (max(rewards) - min(rewards)) if rewards else None,
        "aborted": sum(1 for row in report if row.get("status", "").endswith("ABORTED")),
        "rows": report,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, default=str))
    for row in report:
        print(json.dumps(row, default=str)[:700])
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--router-ip", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=30000)
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-sandboxes", type=int, default=4)
    parser.add_argument("--max-context-len", type=int, default=32768)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--write-off-mode", default="abort_group")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
