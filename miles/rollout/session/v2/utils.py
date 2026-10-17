import logging
from argparse import Namespace
from typing import Any

from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.errors import TokenizationError
from miles.rollout.session.image_processing import expand_input_ids, merge_train_inputs, retained_image_count
from miles.rollout.session.request_args import filter_turn_args
from miles.rollout.session.samples.merge import (
    compute_samples_from_openai_records,
    merge_samples_with_addition_r3,
    truncate_samples_by_total_tokens,
)
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.session_state import SessionRegistryV2, SessionStateV2
from miles.rollout.session.v2.tree_trajectory import TrajectoryNode
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def tree_metadata(state: SessionStateV2) -> dict:
    """The structural layer: node and leaf tables, index-aligned with commits.

    ``response_id`` is the branch<->leaf join key — the agent saw the same id
    in each chat response, so the semantic layer can key per-branch data on it.
    """
    nodes = [
        {
            "id": node.seq,
            "parent": node.parent.seq if node.parent is not None else None,
            "seq": node.seq,
            "truncated": node.truncated,
            "committed_at": node.committed_at,
            "completion_span": list(node.completion_span),
            "num_tokens": len(node.token_ids),
            "response_id": node.response_id,
            "turn_args": filter_turn_args(node.turn_args),
        }
        for node in state.tree.nodes
    ]
    leaves = [
        {"node_id": leaf.seq, "path_node_ids": [n.seq for n in leaf.path_nodes()]} for leaf in state.tree.leaves()
    ]
    return {"nodes": nodes, "leaves": leaves}


def build_leaf_material(
    args: Namespace,
    state: SessionStateV2,
    registry: SessionRegistryV2,
    *,
    session_id: str,
    max_seq_len: int | None,
    use_addition_r3: bool = False,
    computed_mismatch: dict[int, list[dict] | None] | None = None,
) -> list[Sample]:
    """Merge each leaf's root-to-leaf node records into one raw sample, in commit order.

    Leaves whose turns all truncate away are dropped. Each sample's metadata
    carries the ``leaf`` descriptor plus the flat TITO bookkeeping keys for
    the downstream pick/post-process hooks (the per-leaf counterpart of the
    session metadata a v1 sample receives whole).

    With ``use_addition_r3``, each record along a path carries only its
    additional R3 rows; the per-leaf assembler materializes the required prefix
    because a path is the linear record chain its offsets were computed on.

    ``computed_mismatch`` maps a leaf's ``seq`` to the ``tito_session_mismatch``
    already computed for its path, so that path is not re-rendered and re-tokenized.
    """
    material: list[Sample] = []
    for leaf in state.tree.leaves():
        path = leaf.path_nodes()
        records, accumulated_token_ids, leaf_counts = _expanded_path(path, leaf, state, registry)
        turns = compute_samples_from_openai_records(
            args,
            records,
            registry.tokenizer,
            accumulated_token_ids=accumulated_token_ids,
            max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
            use_addition_r3=use_addition_r3,
            evaluation=state.evaluation,
        )
        if max_seq_len is not None:
            turns = truncate_samples_by_total_tokens(turns, max_seq_len, registry.tokenizer)
        if not turns:
            continue
        if use_addition_r3:
            sample = merge_samples_with_addition_r3(args, turns, records, registry.tokenizer)
        else:
            sample = merge_samples(turns, registry.tokenizer)
        if registry.image_processor is not None and not _add_pixels_or_drop(
            sample, leaf, leaf_counts, state, registry
        ):
            continue
        flat: dict[str, Any] = {
            "accumulated_token_ids": list(leaf.token_ids),
            "turn_args": filter_turn_args(leaf.turn_args),
            "leaf": {
                "node_id": leaf.seq,
                "parent": leaf.parent.seq if leaf.parent is not None else None,
                "path_node_ids": [n.seq for n in path],
                "response_id": leaf.response_id,
            },
        }
        if registry.image_processor is not None:
            flat["prompt_image_ids"] = {str(node.seq): list(node.images.image_ids) for node in path}
            flat["image_provenance"] = registry.image_processor.provenance()
        if computed_mismatch is not None and leaf.seq in computed_mismatch:
            mismatch = computed_mismatch[leaf.seq]
        else:
            try:
                mismatch = registry.compute_mismatch(leaf.path_messages(), leaf.token_ids, turn_args=leaf.turn_args)
            except TokenizationError:
                logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
                mismatch = None
        if mismatch is not None:
            flat["tito_session_mismatch"] = mismatch
        sample.metadata = {**(sample.metadata or {}), **flat}
        material.append(sample)
    return material


def _expanded_path(
    path: list[TrajectoryNode],
    leaf: TrajectoryNode,
    state: SessionStateV2,
    registry: SessionRegistryV2,
) -> tuple[list[SessionRecord], list[int], list[int]]:
    """The path's records and accumulated ids as inference saw them, image tokens expanded.

    Serving keeps one placeholder per image so the tree, the matcher, the retry
    picker and the trim arithmetic all work on the sequence the renderer produced.
    The engine expanded those placeholders, so training has to as well — and doing
    it here, before per-turn assembly, means the existing cursor, mask and
    weight-version arithmetic describes the real sequence with no index remapping.
    """
    records = [node.record for node in path]
    if registry.image_processor is None or state.images is None or not leaf.images.occurrences:
        return records, leaf.token_ids, []
    placeholder_id = registry.image_processor.placeholder_id
    expanded_records = [
        node.record.model_copy(
            update={
                "request": {
                    **node.record.request,
                    "input_ids": expand_input_ids(
                        node.record.request["input_ids"], state.images.counts_for(node.images), placeholder_id
                    ),
                }
            }
        )
        for node in path
    ]
    leaf_counts = state.images.counts_for(leaf.images)
    return expanded_records, expand_input_ids(leaf.token_ids, leaf_counts, placeholder_id), leaf_counts


def _add_pixels_or_drop(
    sample: Sample,
    leaf: TrajectoryNode,
    leaf_counts: list[int],
    state: SessionStateV2,
    registry: SessionRegistryV2,
) -> bool:
    """Bind the surviving token sequence to its images; False drops the sample.

    The images comes from the tokens that survived truncation, not from the leaf's
    last request: a cut that lands inside an image leaves pixels whose token span
    is gone.
    """
    image_processor = registry.image_processor
    retained = (
        retained_image_count(sample.tokens, leaf_counts, image_processor.placeholder_id)
        if leaf_counts and state.images is not None
        else 0
    )
    if retained == 0:
        if image_processor.text_only_leaf == "drop":
            logger.info(
                "Dropping leaf seq=%d: no image survives in its %d tokens (--session-image-text-only-leaf=drop)",
                leaf.seq,
                len(sample.tokens),
            )
            return False
        return True
    processed = image_processor.compute_pixels(
        list(leaf.images.image_ids[:retained]), leaf_counts[:retained], state.images
    )
    sample.multimodal_train_inputs = merge_train_inputs(processed)
    return True
