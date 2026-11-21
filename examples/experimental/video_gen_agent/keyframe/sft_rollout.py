"""SFT samples from recorded keyframe episodes, rendered for the policy being trained.

Each row of ``--prompt-data`` holds an episode's messages as the env sent and received them (``sft_data.py`` writes
them), whichever planner played it. They are rendered with the trained planner's own fixed chat template, the way its
session server tokenizes an RL episode turn by turn: every assistant turn is trained from just after its generation
prompt through its end-of-turn token, and nothing else is. Each image placeholder is then expanded by SGLang's
token-space processor, as the rollout engines expand it, so the trainer's rebuild finds the tokens it expects.
"""

import asyncio
import logging

import torch

from miles.backends.training_utils.data.token_space_media_process import (
    RolloutEngineProcessorArgs,
    build_trainer_processor,
)
from miles.utils.chat_template_utils.template import normalize_tool_arguments
from miles.utils.media_refs import collect_media_refs
from miles.utils.processing_utils import load_tokenizer

__all__ = ["generate_rollout", "render_episode"]

logger = logging.getLogger(__name__)

IMAGE_PAD = "<|image_pad|>"
END_OF_TURN = "<|im_end|>"

_STATE: dict = {}


def render_episode(tokenizer, messages: list[dict], tools: list[dict], **template_kwargs) -> tuple[list[int], list[int]]:
    """The episode's tokens, one placeholder per image, and a loss mask of 1 on every assistant turn's own tokens."""
    # The session server's own normalization: tool-call arguments as the template indexes them.
    messages = normalize_tool_arguments(messages, "dict")
    pieces: list[tuple[str, int]] = []
    done = ""
    for k, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = _render(tokenizer, messages[:k], tools, add_generation_prompt=True, **template_kwargs)
        turn = _render(tokenizer, messages[: k + 1], tools, **template_kwargs)
        if not prompt.startswith(done) or not turn.startswith(prompt):
            raise ValueError(f"turn {k} does not extend the rendered conversation; the template rewrites history")
        # The planner stops at its end-of-turn token; what the template writes after it is the next prompt's.
        end = turn.rindex(END_OF_TURN, len(prompt)) + len(END_OF_TURN)
        pieces += [(prompt[len(done) :], 0), (turn[len(prompt) : end], 1)]
        done = turn[:end]
    tokens, mask = [], []
    for text, trained in pieces:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens += ids
        mask += [trained] * len(ids)
    return tokens, mask


def _render(tokenizer, messages, tools, **kwargs) -> str:
    return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, **kwargs)


def expand_images(core, tokens: list[int], mask: list[int], images: list[str], pad_id: int) -> tuple[list, list]:
    """Each image's single placeholder expanded to its token count, as the engines expand it; untrained."""
    if not images:
        return tokens, mask
    output = asyncio.run(core.process(image_data=images, input_ids=tokens, mm_token_expansion_start_len=0))
    expanded = output["input_ids"][0].tolist()
    expanded_mask, i = [], 0
    for token in expanded:
        if token == pad_id and (i >= len(tokens) or tokens[i] != pad_id):
            # An extra copy of the placeholder before it, which an image never carries a loss on.
            assert expanded_mask and expanded_mask[-1] == 0, "an image placeholder sits in a trained turn"
            expanded_mask.append(0)
            continue
        assert token == tokens[i], f"the processor changed token {i}, not only an image placeholder"
        expanded_mask.append(mask[i])
        i += 1
    assert i == len(tokens), "the processor dropped tokens"
    return expanded, expanded_mask


def _setup(args) -> dict:
    if not _STATE:
        tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True)
        engine_args = RolloutEngineProcessorArgs.from_sglang_config(args.sglang, hf_checkpoint=args.hf_checkpoint)
        core = build_trainer_processor(
            engine_args, mm_processor_worker_num=1, mm_io_worker_num=0, device=torch.device("cpu")
        ).core
        _STATE.update(
            tokenizer=tokenizer,
            core=core,
            pad_id=tokenizer.convert_tokens_to_ids(IMAGE_PAD),
            kwargs=dict(args.apply_chat_template_kwargs or {}),
        )
    return _STATE


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """``--rollout-function-path`` for an SFT run on rows of recorded episodes."""
    assert not evaluation, "an SFT run on recorded episodes has no eval"
    state = _setup(args)
    samples = data_buffer.get_samples(args.rollout_batch_size)
    out = []
    for (sample,) in samples:
        messages, tools = sample.prompt, sample.metadata.get("tools")
        tokens, mask = render_episode(state["tokenizer"], messages, tools, **state["kwargs"])
        images = (collect_media_refs(messages, None) or {}).get("images") or []
        tokens, mask = expand_images(state["core"], tokens, mask, images, state["pad_id"])
        response_length = len(mask) - mask.index(1)
        sample.tokens = tokens
        sample.response_length = response_length
        sample.loss_mask = mask[-response_length:]
        sample.multimodal_inputs = {"images": images} if images else None
        sample.reward = 0
        out.append(sample)
    if rollout_id == 0:
        logger.info(f"keyframe SFT: {len(out)} episodes, first has {len(out[0].tokens)} tokens")
    return out
