"""Prompt data source of diffusion runs: jsonl prompts in file order, no tokenizer."""

import copy
import json
from pathlib import Path

from miles.rollout.data_source import DataSource
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.types import Sample


class DiffusionPromptDataSource(DataSource):
    """Each `get_samples` call takes the next prompts, wrapping around the file, and repeats each
    `--n-samples-per-prompt` times; group and sample indices count up over the whole run."""

    def __init__(self, args) -> None:
        self.args = args
        self.prompts = _read_prompts(args.prompt_data, prompt_key=args.input_key, metadata_key=args.metadata_key)
        assert self.prompts, f"no prompts under key {args.input_key!r} in {args.prompt_data}"
        self.prompt_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = []
        for _ in range(num_samples):
            prompt, metadata = self.prompts[self.prompt_offset]
            self.prompt_offset = (self.prompt_offset + 1) % len(self.prompts)
            groups.append(self._make_group(prompt, metadata))
            self.sample_group_index += 1
        return groups

    def _make_group(self, prompt: str, metadata: dict) -> list[Sample]:
        group = []
        for _ in range(self.args.n_samples_per_prompt):
            group.append(
                Sample(
                    group_index=self.sample_group_index,
                    index=self.sample_index,
                    prompt=prompt,
                    metadata=copy.deepcopy(metadata),
                )
            )
            self.sample_index += 1
        return group

    def save(self, directory: Path) -> None:
        save_simple_checkpoint(
            directory=directory,
            data=dict(
                prompt_offset=self.prompt_offset,
                sample_group_index=self.sample_group_index,
                sample_index=self.sample_index,
            ),
        )

    def load(self, directory: Path) -> None:
        state = load_simple_checkpoint(directory=directory)
        self.prompt_offset = state["prompt_offset"]
        self.sample_group_index = state["sample_group_index"]
        self.sample_index = state["sample_index"]


def _read_prompts(path: str, *, prompt_key: str, metadata_key: str) -> list[tuple[str, dict]]:
    prompts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get(prompt_key)
            if isinstance(prompt, str) and prompt.strip():
                prompts.append((prompt.strip(), row.get(metadata_key) or {}))
    return prompts
