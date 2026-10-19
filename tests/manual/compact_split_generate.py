"""E2E fixture: compact generate fns splitting one rollout execution into several
training samples that share the parent's rollout_id. Use via
--custom-generate-function-path tests.manual.compact_split_generate.generate"""

import copy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    return await _generate_split(input, num_pieces=2)


async def generate_one_to_three(input: GenerateFnInput) -> GenerateFnOutput:
    return await _generate_split(input, num_pieces=1 + input.sample.index % 3)


async def _generate_split(input: GenerateFnInput, *, num_pieces: int) -> GenerateFnOutput:
    output = await single_turn_generate(input)
    sample = output.samples
    assert isinstance(sample, Sample)

    sample.rollout_id = sample.index
    num_pieces = min(num_pieces, sample.response_length)
    if num_pieces < 2 or sample.status == Sample.Status.ABORTED:
        return GenerateFnOutput(samples=[sample])

    prompt_len = len(sample.tokens) - sample.response_length
    # A None mask becomes all ones in conversion, which would train earlier pieces' tokens again.
    loss_mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
    cuts = [sample.response_length * k // num_pieces for k in range(num_pieces + 1)]
    pieces = []
    for start, end in zip(cuts, cuts[1:], strict=False):
        piece = copy.deepcopy(sample)
        piece.tokens = sample.tokens[: prompt_len + end]
        piece.response_length = end
        piece.loss_mask = [0] * start + list(loss_mask[start:end])
        if sample.rollout_log_probs is not None:
            piece.rollout_log_probs = sample.rollout_log_probs[:end]
        if end < sample.response_length:
            piece.status = Sample.Status.COMPLETED
        pieces.append(piece)
    return GenerateFnOutput(samples=pieces)
