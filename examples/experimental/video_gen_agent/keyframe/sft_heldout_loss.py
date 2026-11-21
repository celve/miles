"""A model's per-token loss on SFT rows, rendered and masked as ``sft_rollout.py`` trains them.

The SFT run has no eval, so the held-out split ``sft_data.py`` writes is scored here, before and after: once on the
base planner and once on the SFT run's HF export. Images go through the model's own HF processor, whose placeholder
counts the trainer's rebuild checks against too.

usage (one GPU): python -m examples.experimental.video_gen_agent.keyframe.sft_heldout_loss <hf dir> <rows.jsonl>
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from examples.experimental.video_gen_agent.keyframe.sft_rollout import IMAGE_PAD, render_episode
from transformers import AutoModelForImageTextToText, AutoProcessor

from miles.utils.chat_template_utils.tito_tokenizer import resolve_fixed_chat_template
from miles.utils.media_refs import collect_media_refs, load_media_for_processor
from miles.utils.processing_utils import load_tokenizer


def _expand(tokens: list[int], mask: list[int], counts: list[int], pad_id: int) -> tuple[list[int], list[int]]:
    """Each image placeholder repeated to its image's token count, untrained."""
    counts = iter(counts)
    out_tokens, out_mask = [], []
    for token, trained in zip(tokens, mask, strict=True):
        repeat = next(counts) if token == pad_id else 1
        out_tokens += [token] * repeat
        out_mask += [trained] * repeat
    assert next(counts, None) is None, "more images than placeholders"
    return out_tokens, out_mask


@torch.no_grad()
def _row_loss(model, processor, tokenizer, row: dict, template_kwargs: dict) -> tuple[float, int]:
    tokens, mask = render_episode(tokenizer, row["messages"], row["tools"], **template_kwargs)
    media = collect_media_refs(row["messages"], None) or {}
    inputs = {}
    counts = []
    if media.get("images"):
        vision = processor.image_processor(images=load_media_for_processor(media)["images"], return_tensors="pt")
        merge = processor.image_processor.merge_size
        counts = (vision["image_grid_thw"].prod(-1) // merge**2).tolist()
        inputs = {key: value.to(model.device) for key, value in vision.items()}
    tokens, mask = _expand(tokens, mask, counts, tokenizer.convert_tokens_to_ids(IMAGE_PAD))
    ids = torch.tensor([tokens], device=model.device)
    logits = model(input_ids=ids, **inputs).logits[0, :-1].float()
    trained = torch.tensor(mask[1:], device=model.device, dtype=torch.bool)
    loss = F.cross_entropy(logits[trained], ids[0, 1:][trained], reduction="sum")
    return loss.item(), int(trained.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", help="an HF model directory: the base planner or an SFT run's hf export")
    parser.add_argument("rows", type=Path, help="a jsonl sft_data.py wrote, normally heldout.jsonl")
    parser.add_argument("--tito-model", default="qwen35", help="the planner family whose fixed template renders rows")
    args = parser.parse_args()

    template, template_kwargs = resolve_fixed_chat_template(args.tito_model)
    tokenizer = load_tokenizer(args.model, chat_template_path=template, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    losses = [_row_loss(model, processor, tokenizer, row, template_kwargs) for row in rows]
    total, tokens = sum(loss for loss, _ in losses), sum(n for _, n in losses)
    print(json.dumps({"model": args.model, "rows": len(rows), "tokens": tokens, "loss_per_token": total / tokens}))


if __name__ == "__main__":
    main()
