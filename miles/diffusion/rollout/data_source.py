"""Prompt data source of diffusion runs: jsonl prompts in file order, or shuffled per epoch, no tokenizer."""

import copy
import json
import random
from pathlib import Path

from miles.rollout.data_source import DataSource
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.types import Sample


class DiffusionPromptDataSource(DataSource):
    """Each `get_samples` call takes the next prompts, wrapping around the file, and repeats each
    `--n-samples-per-prompt` times; group and sample indices count up over the whole run. `--rollout-shuffle`
    permutes each epoch with `--rollout-seed` + epoch, as the text data source does."""

    def __init__(self, args) -> None:
        self.args = args
        self.prompts = _read_prompts(args.prompt_data, prompt_key=args.input_key, metadata_key=args.metadata_key)
        assert self.prompts, f"no prompts under key {args.input_key!r} in {args.prompt_data}"
        self.prompt_offset = 0
        self.epoch = 0
        self.order = self._epoch_order()
        self.sample_group_index = 0
        self.sample_index = 0

    @property
    def dataset(self) -> list[tuple[str, dict]]:
        """The prompt list; its length is what `--num-epoch` counts."""
        return self.prompts

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = []
        for _ in range(num_samples):
            prompt, metadata = self.prompts[self.order[self.prompt_offset]]
            self.prompt_offset += 1
            if self.prompt_offset == len(self.prompts):
                self.prompt_offset, self.epoch = 0, self.epoch + 1
                self.order = self._epoch_order()
            groups.append(self._make_group(prompt, metadata))
            self.sample_group_index += 1
        return groups

    def _epoch_order(self) -> list[int]:
        order = list(range(len(self.prompts)))
        if self.args.rollout_shuffle:
            random.Random(self.args.rollout_seed + self.epoch).shuffle(order)
        return order

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
                epoch=self.epoch,
                sample_group_index=self.sample_group_index,
                sample_index=self.sample_index,
            ),
        )

    def load(self, directory: Path) -> None:
        state = load_simple_checkpoint(directory=directory)
        self.prompt_offset = state["prompt_offset"]
        # Checkpoints saved before --rollout-shuffle hold no epoch; their prompt order was the file order of epoch 0.
        self.epoch = state.get("epoch", 0)
        self.order = self._epoch_order()
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
