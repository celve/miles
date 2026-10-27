import torch

from miles.backends.training_utils.data import token_space_media_process


class _RecordingProcessor:
    def __init__(self):
        self.calls = []

    async def process(self, **kwargs):
        self.calls.append(kwargs)
        num_images = len(kwargs["image_data"])
        return {
            "input_ids": torch.tensor([kwargs["input_ids"]]),
            "attention_mask": torch.ones(1, len(kwargs["input_ids"])),
            "pixel_values": torch.full((num_images, 3), float(num_images)),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * num_images),
            "mm_modalities": ["image"] * num_images,
            "mm_spans": [(1, 3), (3, 5)],
        }


def test_trainer_rebuilds_tensors_without_expanding_the_rollout_tokens(monkeypatch):
    processor = _RecordingProcessor()
    monkeypatch.setattr(
        token_space_media_process, "_token_space_processor", lambda hf_checkpoint, mm_process_config_json, keep_on_device=False: processor
    )
    multimodal_inputs = [{"images": ["/shared/a.png", "/shared/b.png"]}, None]
    tokens = [torch.tensor([1, 9, 9, 9, 9, 2, 5]), torch.tensor([1, 2])]

    train_inputs = token_space_media_process.compute_multimodal_train_inputs(
        "ckpt", None, multimodal_inputs, tokens, device=torch.device("cpu")
    )

    (call,) = processor.calls
    # The rollout tokens are already expanded, so expansion starts past their end.
    assert call["input_ids"] == [1, 9, 9, 9, 9, 2, 5]
    assert call["mm_token_expansion_start_len"] == 7
    assert call["image_data"] == ["/shared/a.png", "/shared/b.png"]
    assert call["image_device"] == "cpu"
    assert train_inputs[1] is None
    assert set(train_inputs[0]) == {"pixel_values", "image_grid_thw"}
    assert torch.equal(train_inputs[0]["image_grid_thw"], torch.tensor([[1, 2, 2], [1, 2, 2]]))


def test_per_micro_batch_rebuild_matches_the_round_rebuild_and_skips_expansion(monkeypatch):
    from types import SimpleNamespace

    from miles.backends.training_utils.data import rollout as data_rollout
    from miles.backends.training_utils.data.multimodal import expand_multimodal_rollout_data_in_place

    processor = _RecordingProcessor()
    monkeypatch.setattr(
        token_space_media_process, "_token_space_processor", lambda hf_checkpoint, mm_process_config_json, keep_on_device=False: processor
    )
    monkeypatch.setattr(data_rollout.current_platform, "current_device", lambda: torch.device("cpu"))
    args = SimpleNamespace(
        hf_checkpoint="ckpt",
        fp16=False,
        sglang_mm_process_config=None,
        multimodal_rebuild_decode_workers=0,
        multimodal_rebuild_prefetch=False,
        multimodal_rebuild_cache_mb=0,
        multimodal_rebuild_keep_on_device=False,
    )
    multimodal_inputs = [{"images": ["/shared/a.png"]}, None, {"images": ["/shared/b.png", "/shared/c.png"]}]
    tokens = [torch.tensor([1, 9, 2]), torch.tensor([1, 2]), torch.tensor([1, 9, 9, 2])]

    lazy = data_rollout.LazyMultimodalTrainInputs(args, multimodal_inputs, tokens)
    rollout_data = {"tokens": tokens, "multimodal_train_inputs": lazy}
    expand_multimodal_rollout_data_in_place(rollout_data)
    assert processor.calls == []

    batch = data_rollout.DataIterator(rollout_data, micro_batch_indices=[[2, 1]]).get_next(["multimodal_train_inputs"])
    rebuilt, text_only = batch["multimodal_train_inputs"]
    assert text_only is None
    assert rebuilt["pixel_values"].dtype == torch.bfloat16
    assert torch.equal(rebuilt["image_grid_thw"], torch.tensor([[1, 2, 2], [1, 2, 2]]))
    assert [call["image_data"] for call in processor.calls] == [["/shared/b.png", "/shared/c.png"]]


def test_decoded_images_decode_once_and_trim_to_the_byte_cap():
    from miles.backends.training_utils.data.decoded_media import DecodedImages

    decoded = []

    def decode(path):
        decoded.append(path)
        return torch.zeros(1024, dtype=torch.uint8)

    images = DecodedImages(decode, workers=2, cache_bytes=2048)
    images.request(["a", "b", "a"])
    assert images.get("a").numel() == 1024 and images.get("b").numel() == 1024
    images.request(["c"])
    images.get("c")
    images.trim(keep={"c"})
    images.get("c")
    images.get("b")
    images.get("a")
    assert sorted(decoded) == ["a", "a", "b", "c"]


def test_parallel_decode_and_prefetch_feed_the_processor_decoded_images(monkeypatch):
    from types import SimpleNamespace

    from miles.backends.training_utils.data import rollout as data_rollout

    processor = _RecordingProcessor()
    monkeypatch.setattr(
        token_space_media_process, "_token_space_processor", lambda hf_checkpoint, mm_process_config_json, keep_on_device=False: processor
    )
    monkeypatch.setattr(data_rollout, "image_decoder", lambda hf_checkpoint, mm_config: lambda path: SimpleNamespace(nbytes=10, path=path))
    monkeypatch.setattr(data_rollout.current_platform, "current_device", lambda: torch.device("cpu"))
    args = SimpleNamespace(
        hf_checkpoint="ckpt",
        fp16=False,
        sglang_mm_process_config=None,
        multimodal_rebuild_decode_workers=2,
        multimodal_rebuild_prefetch=True,
        multimodal_rebuild_cache_mb=1,
        multimodal_rebuild_keep_on_device=False,
    )
    multimodal_inputs = [{"images": ["/s/a.png", "/s/b.png"]}, {"images": ["/s/b.png", "/s/c.png"]}]
    tokens = [torch.tensor([1, 9, 9, 2]), torch.tensor([1, 9, 9, 2])]
    lazy = data_rollout.LazyMultimodalTrainInputs(args, multimodal_inputs, tokens)
    iterator = data_rollout.DataIterator({"multimodal_train_inputs": lazy}, micro_batch_indices=[[0], [1]])

    iterator.get_next(["multimodal_train_inputs"])
    assert "/s/c.png" in lazy._images._entries
    iterator.get_next(["multimodal_train_inputs"])

    assert [[image.path for image in call["image_data"]] for call in processor.calls] == [
        ["/s/a.png", "/s/b.png"],
        ["/s/b.png", "/s/c.png"],
    ]
    assert lazy._images.decodes == 3
