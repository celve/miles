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
        token_space_media_process,
        "_token_space_processor",
        lambda hf_checkpoint, mm_process_config_json, keep_on_device=False: processor,
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


def _lazy_args(**overrides):
    from types import SimpleNamespace

    defaults = dict(
        hf_checkpoint="ckpt",
        fp16=False,
        sglang_mm_process_config=None,
        multimodal_rebuild_decode_workers=0,
        multimodal_rebuild_prefetch=False,
        multimodal_rebuild_cache_mb=0,
        multimodal_rebuild_feature_cache_mb=0,
        multimodal_rebuild_keep_on_device=False,
    )
    return SimpleNamespace(**{**defaults, **overrides})


def test_per_micro_batch_rebuild_matches_the_round_rebuild_and_skips_expansion(monkeypatch):
    from miles.backends.training_utils.data import rollout as data_rollout
    from miles.backends.training_utils.data.multimodal import expand_multimodal_rollout_data_in_place

    processor = _RecordingProcessor()
    monkeypatch.setattr(
        token_space_media_process,
        "_token_space_processor",
        lambda hf_checkpoint, mm_process_config_json, keep_on_device=False: processor,
    )
    monkeypatch.setattr(data_rollout.current_platform, "current_device", lambda: torch.device("cpu"))
    args = _lazy_args()
    multimodal_inputs = [{"images": ["/shared/a.png"]}, None, {"images": ["/shared/b.png", "/shared/c.png"]}]
    token_ids = [[1, 9, 2], [1, 2], [1, 9, 9, 2]]
    tokens = [torch.tensor(ids) for ids in token_ids]

    from miles.utils.timer import Timer

    phases = []
    monkeypatch.setattr(Timer(), "event_sinks", [lambda name, t0, t1: phases.append((name, t0, t1))])
    lazy = data_rollout.LazyMultimodalTrainInputs(args, multimodal_inputs, token_ids)
    rollout_data = {"tokens": tokens, "multimodal_train_inputs": lazy}
    expand_multimodal_rollout_data_in_place(rollout_data)
    assert processor.calls == []

    batch = data_rollout.DataIterator(rollout_data, micro_batch_indices=[[2, 1]]).get_next(["multimodal_train_inputs"])
    rebuilt, text_only = batch["multimodal_train_inputs"]
    assert text_only is None
    assert rebuilt["pixel_values"].dtype == torch.bfloat16
    assert torch.equal(rebuilt["image_grid_thw"], torch.tensor([[1, 2, 2], [1, 2, 2]]))
    assert [call["image_data"] for call in processor.calls] == [["/shared/b.png", "/shared/c.png"]]
    assert processor.calls[0]["input_ids"] == [1, 9, 9, 2]
    assert len(phases) == 1
    assert all(name == "multimodal_rebuild" and t1 >= t0 for name, t0, t1 in phases)


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


def test_parallel_decode_feeds_the_processor_decoded_images(monkeypatch):
    from types import SimpleNamespace

    from miles.backends.training_utils.data import rollout as data_rollout

    processor = _RecordingProcessor()
    monkeypatch.setattr(
        token_space_media_process,
        "_token_space_processor",
        lambda hf_checkpoint, mm_process_config_json, keep_on_device=False: processor,
    )
    monkeypatch.setattr(
        data_rollout,
        "image_decoder",
        lambda hf_checkpoint, mm_config: lambda path: SimpleNamespace(nbytes=10, path=path),
    )
    monkeypatch.setattr(data_rollout.current_platform, "current_device", lambda: torch.device("cpu"))
    args = _lazy_args(multimodal_rebuild_decode_workers=2, multimodal_rebuild_prefetch=True, multimodal_rebuild_cache_mb=1)
    multimodal_inputs = [{"images": ["/s/a.png", "/s/b.png"]}, {"images": ["/s/b.png", "/s/c.png"]}]
    lazy = data_rollout.LazyMultimodalTrainInputs(args, multimodal_inputs, [[1, 9, 9, 2], [1, 9, 9, 2]])
    iterator = data_rollout.DataIterator({"multimodal_train_inputs": lazy}, micro_batch_indices=[[0], [1]])

    iterator.get_next(["multimodal_train_inputs"])
    iterator.get_next(["multimodal_train_inputs"])

    assert [[image.path for image in call["image_data"]] for call in processor.calls] == [
        ["/s/a.png", "/s/b.png"],
        ["/s/b.png", "/s/c.png"],
    ]
    assert lazy._images.decodes == 3


def test_prefetch_rebuilds_the_next_micro_batch_on_one_thread_before_it_is_asked_for(monkeypatch):
    import threading

    from miles.backends.training_utils.data import rollout as data_rollout

    processor = _RecordingProcessor()
    original_process = processor.process
    threads = []

    async def process(**kwargs):
        threads.append(threading.get_ident())
        return await original_process(**kwargs)

    processor.process = process
    monkeypatch.setattr(
        token_space_media_process,
        "_token_space_processor",
        lambda hf_checkpoint, mm_process_config_json, keep_on_device=False: processor,
    )
    monkeypatch.setattr(data_rollout.current_platform, "current_device", lambda: torch.device("cpu"))
    multimodal_inputs = [{"images": ["/s/a.png"]}, {"images": ["/s/b.png", "/s/c.png"]}, {"images": ["/s/c.png"]}]
    token_ids = [[1, 9, 2], [1, 9, 9, 2], [1, 9, 2]]
    schedule = [[0], [1], [2]]

    inline = data_rollout.LazyMultimodalTrainInputs(_lazy_args(), multimodal_inputs, token_ids)
    expected = [inline.get_many(indices) for indices in schedule]
    processor.calls.clear()
    threads.clear()

    lazy = data_rollout.LazyMultimodalTrainInputs(
        _lazy_args(multimodal_rebuild_prefetch=True), multimodal_inputs, token_ids
    )
    iterator = data_rollout.DataIterator({"multimodal_train_inputs": lazy}, micro_batch_indices=schedule)
    first = iterator.get_next(["multimodal_train_inputs"])["multimodal_train_inputs"]
    lazy._pending[1].result()
    assert [call["input_ids"] for call in processor.calls] == [[1, 9, 2], [1, 9, 9, 2]]

    batches = [first] + [iterator.get_next(["multimodal_train_inputs"])["multimodal_train_inputs"] for _ in range(2)]
    # A request that is not the prefetched micro-batch is rebuilt on the same thread.
    batches.append(lazy.get_many([0]))

    assert len(processor.calls) == 4
    assert len(set(threads)) == 1 and threads[0] != threading.get_ident()
    for got, want in zip(batches, expected + [expected[0]], strict=True):
        for got_mm, want_mm in zip(got, want, strict=True):
            assert got_mm.keys() == want_mm.keys()
            assert all(torch.equal(got_mm[key], want_mm[key]) for key in got_mm)


def test_decode_threads_run_the_initializer_before_decoding():
    import threading

    from miles.backends.training_utils.data.decoded_media import DecodedImages

    initialized = set()
    images = DecodedImages(
        lambda path: (threading.get_ident() in initialized, torch.zeros(1)),
        workers=2,
        cache_bytes=0,
        thread_initializer=lambda: initialized.add(threading.get_ident()),
    )
    assert images.get("a")[0] and images.get("b")[0]


def test_dump_details_can_pickle_rebuild_inputs_that_hold_thread_pools(monkeypatch):
    import pickle

    from miles.backends.training_utils.data import rollout as data_rollout

    monkeypatch.setattr(data_rollout, "image_decoder", lambda hf_checkpoint, mm_config: lambda path: path)
    monkeypatch.setattr(data_rollout.current_platform, "current_device", lambda: torch.device("cpu"))
    args = _lazy_args(multimodal_rebuild_decode_workers=2, multimodal_rebuild_prefetch=True)
    lazy = data_rollout.LazyMultimodalTrainInputs(args, [{"images": ["/s/a.png"]}], [[1, 9, 2]])

    restored = pickle.loads(pickle.dumps(lazy))

    assert restored._multimodal_inputs == [{"images": ["/s/a.png"]}]
    assert restored._token_ids == [[1, 9, 2]]


class _PerImageProcessor:
    """Two rows per image, valued by the image's path, as Qwen-VL processes each image on its own."""

    def __init__(self):
        self.calls = []

    async def process(self, **kwargs):
        self.calls.append(kwargs["image_data"])
        values = [float(sum(map(ord, path))) for path in kwargs["image_data"]]
        return {
            "input_ids": torch.tensor([kwargs["input_ids"]]),
            "pixel_values": torch.tensor([[v, v + 1] for v in values for _ in range(2)]),
            "image_grid_thw": torch.tensor([[1, 1, 2]] * len(values)),
        }


def test_feature_cache_processes_a_shared_screenshot_once_and_matches_the_processor(monkeypatch):
    from miles.backends.training_utils.data import rollout as data_rollout

    processor = _PerImageProcessor()
    monkeypatch.setattr(
        token_space_media_process,
        "_token_space_processor",
        lambda hf_checkpoint, mm_process_config_json, keep_on_device=False: processor,
    )
    monkeypatch.setattr(data_rollout.current_platform, "current_device", lambda: torch.device("cpu"))
    multimodal_inputs = [{"images": ["/s/a.png", "/s/b.png"]}, {"images": ["/s/b.png", "/s/c.png"]}, {"images": ["/s/b.png"]}]
    token_ids = [[1, 9, 9, 2], [1, 9, 9, 2], [1, 9, 2]]

    expected = data_rollout.LazyMultimodalTrainInputs(_lazy_args(), multimodal_inputs, token_ids).get_many([0, 1, 2])
    processor.calls.clear()
    lazy = data_rollout.LazyMultimodalTrainInputs(
        _lazy_args(multimodal_rebuild_feature_cache_mb=1), multimodal_inputs, token_ids
    )
    first_pass = lazy.get_many([0, 1, 2])
    second_pass = lazy.get_many([2, 1, 0])

    assert processor.calls == [["/s/a.png", "/s/b.png"], ["/s/b.png", "/s/c.png"]]
    for got, want in zip(first_pass + second_pass[::-1], expected + expected, strict=True):
        assert got.keys() == want.keys()
        assert all(torch.equal(got[key], want[key]) for key in got)


def test_feature_cache_evicts_past_its_byte_cap():
    from miles.backends.training_utils.data.decoded_media import ImageFeatureCache

    cache = ImageFeatureCache(cache_bytes=64)
    for path in ("a", "b", "c"):
        cache.put(path, torch.zeros(4, 2), torch.tensor([1, 2, 2]))

    assert cache.get(["a"]) is None
    assert cache.get(["b", "c"]) is not None
