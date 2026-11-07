import contextlib
import pickle
import threading

import pytest
import torch
from tests.fast.fixtures.sglang_config_fixtures import with_parser_defaults_and_sglang_config
from transformers import BaseImageProcessor

from miles.backends.training_utils.data import token_space_media_process
from miles.backends.training_utils.data.rollout import DataIterator, LazyMultimodalTrainInputs
from miles.backends.training_utils.data.token_space_media_process import RolloutEngineProcessorArgs
from miles.utils.timer import Timer


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
        "_trainer_processor",
        lambda rollout_engine_args, device: token_space_media_process._TrainerProcessor(
            core=processor, image_device="cuda:3"
        ),
    )
    multimodal_inputs = [{"images": ["/shared/a.png", "/shared/b.png"]}, None]
    tokens = [torch.tensor([1, 9, 9, 9, 9, 2, 5]), torch.tensor([1, 2])]

    train_inputs = token_space_media_process.compute_multimodal_train_inputs(
        object(),
        multimodal_inputs,
        tokens,
        device=torch.device("cpu"),
    )

    (call,) = processor.calls
    # The rollout tokens are already expanded, so expansion starts past their end.
    assert call["input_ids"] == [1, 9, 9, 9, 9, 2, 5]
    assert call["mm_token_expansion_start_len"] == 7
    assert call["image_data"] == ["/shared/a.png", "/shared/b.png"]
    # As the engine's wrapper: the derived image device, and no video device.
    assert call["image_device"] == "cuda:3"
    assert "video_device" not in call
    assert train_inputs[1] is None
    assert set(train_inputs[0]) == {"pixel_values", "image_grid_thw"}
    assert torch.equal(train_inputs[0]["image_grid_thw"], torch.tensor([[1, 2, 2], [1, 2, 2]]))


def test_rollout_engine_args_read_the_parsed_engine_config():
    """Every value comes from the engines' resolved server args; the tokenizer defaults to the model, as in SGLang."""
    values = with_parser_defaults_and_sglang_config(
        {
            "hf_checkpoint": "/ckpt",
            "rollout_num_gpus": 2,
            "rollout_num_gpus_per_engine": 1,
            "num_gpus_per_node": 8,
            "sglang_revision": "v1",
            "sglang_image_processor_backend": "pil",
            "sglang_mm_process_config": {"video": {"fps": 2}},
        }
    )

    rollout_engine_args = RolloutEngineProcessorArgs.from_sglang_config(values["sglang"])

    assert (rollout_engine_args.model_path, rollout_engine_args.tokenizer_path) == ("/ckpt", "/ckpt")
    assert rollout_engine_args.revision == "v1"
    assert rollout_engine_args.image_processor_backend == "pil"
    assert rollout_engine_args.mm_process_config_json == '{"video": {"fps": 2}}'


class _ImageProcessor(BaseImageProcessor):
    pass


@pytest.mark.parametrize(
    ("backend", "image_processor", "expected"),
    [("auto", _ImageProcessor(), "cuda:3"), ("pil", _ImageProcessor(), None), ("auto", object(), None)],
    ids=["image-processor", "pil-backend", "not-an-image-processor"],
)
def test_image_device_follows_the_engine_wrapper(monkeypatch, backend, image_processor, expected):
    """A trainer that preprocesses images on another device than the engines computes other pixels."""
    from sglang.srt.multimodal import serving_preprocessing

    calls = []

    def fake_fast_image_processor_device(processor, *, base_gpu_id, rl_on_policy_target, true_on_policy_enabled):
        calls.append(base_gpu_id)
        return f"cuda:{base_gpu_id}"

    monkeypatch.setattr(serving_preprocessing, "fast_image_processor_device", fake_fast_image_processor_device)
    hf_processor = type("Processor", (), {"image_processor": image_processor})()

    image_device = token_space_media_process._image_device(
        hf_processor, image_processor_backend=backend, device=torch.device("cuda", 3)
    )

    assert image_device == expected
    assert calls == ([3] if expected else [])


def _patch_trainer_processor(monkeypatch, processor):
    monkeypatch.setattr(
        token_space_media_process,
        "_trainer_processor",
        lambda rollout_engine_args, device: token_space_media_process._TrainerProcessor(
            core=processor, image_device=None
        ),
    )


@pytest.fixture
def fake_cuda_streams(monkeypatch):
    events = []

    class Stream:
        def __init__(self, device):
            self.device = device

        def record_event(self):
            event = object()
            events.append(("record", threading.get_ident(), event))
            return event

    class CurrentStream:
        def wait_event(self, event):
            events.append(("wait", threading.get_ident(), event))

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: CurrentStream())
    return events


def _lazy(multimodal_inputs, token_ids, *, prefetch):
    return LazyMultimodalTrainInputs(
        None,
        multimodal_inputs,
        token_ids,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        prefetch=prefetch,
    )


def test_a_micro_batch_rebuild_is_timed_once_as_the_media_wait(monkeypatch):
    processor = _RecordingProcessor()
    _patch_trainer_processor(monkeypatch, processor)
    phases = []
    monkeypatch.setattr(Timer(), "event_sinks", [lambda name, t0, t1: phases.append((name, t0, t1))])
    multimodal_inputs = [{"images": ["/s/a.png"]}, None, {"images": ["/s/b.png", "/s/c.png"]}]
    lazy = _lazy(multimodal_inputs, [[1, 9, 2], [1, 2], [1, 9, 9, 2]], prefetch=False)

    batch = DataIterator({"multimodal_train_inputs": lazy}, micro_batch_indices=[[2, 1]]).get_next(
        ["multimodal_train_inputs"]
    )

    rebuilt, text_only = batch["multimodal_train_inputs"]
    assert text_only is None
    assert rebuilt["pixel_values"].dtype == torch.bfloat16
    assert [call["image_data"] for call in processor.calls] == [["/s/b.png", "/s/c.png"]]
    assert [name for name, _, _ in phases] == ["multimodal_rebuild"]
    assert all(t1 >= t0 for _, t0, t1 in phases)


def test_prefetch_rebuilds_the_next_micro_batch_on_one_thread_and_the_forward_waits_for_it(
    monkeypatch, fake_cuda_streams
):
    processor = _RecordingProcessor()
    threads = []
    process = processor.process

    async def recording_process(**kwargs):
        threads.append(threading.get_ident())
        return await process(**kwargs)

    processor.process = recording_process
    _patch_trainer_processor(monkeypatch, processor)
    multimodal_inputs = [{"images": ["/s/a.png"]}, {"images": ["/s/b.png", "/s/c.png"]}, {"images": ["/s/c.png"]}]
    token_ids = [[1, 9, 2], [1, 9, 9, 2], [1, 9, 2]]
    schedule = [[0], [1], [2]]
    inline = _lazy(multimodal_inputs, token_ids, prefetch=False)
    expected = [inline.get_many(indices) for indices in schedule] + [inline.get_many([0])]
    processor.calls.clear()
    threads.clear()

    lazy = _lazy(multimodal_inputs, token_ids, prefetch=True)
    iterator = DataIterator({"multimodal_train_inputs": lazy}, micro_batch_indices=schedule)
    batches = [iterator.get_next(["multimodal_train_inputs"])["multimodal_train_inputs"]]
    lazy._pending[1].result()
    assert [call["input_ids"] for call in processor.calls] == [[1, 9, 2], [1, 9, 9, 2]]
    batches += [iterator.get_next(["multimodal_train_inputs"])["multimodal_train_inputs"] for _ in range(2)]
    batches.append(lazy.get_many([0]))

    assert len(processor.calls) == 4
    assert len(set(threads)) == 1 and threads[0] != threading.get_ident()
    recorded = [event for kind, thread, event in fake_cuda_streams if kind == "record" and thread == threads[0]]
    waited = [event for kind, thread, event in fake_cuda_streams if kind == "wait" and thread == threading.get_ident()]
    assert waited == recorded
    for got, want in zip(batches, expected, strict=True):
        for got_mm, want_mm in zip(got, want, strict=True):
            assert got_mm.keys() == want_mm.keys()
            assert all(torch.equal(got_mm[key], want_mm[key]) for key in got_mm)


def test_a_prefetch_still_running_when_the_iterator_resets_is_dropped(monkeypatch, fake_cuda_streams):
    processor = _RecordingProcessor()
    gate = threading.Event()
    process = processor.process

    async def gated_process(**kwargs):
        if kwargs["input_ids"] == [1, 9, 9, 2]:
            gate.wait(timeout=10)
        return await process(**kwargs)

    processor.process = gated_process
    _patch_trainer_processor(monkeypatch, processor)
    multimodal_inputs = [{"images": ["/s/a.png"]}, {"images": ["/s/b.png", "/s/c.png"]}]
    token_ids = [[1, 9, 2], [1, 9, 9, 2]]
    expected = _lazy(multimodal_inputs, token_ids, prefetch=False).get_many([0])
    processor.calls.clear()

    lazy = _lazy(multimodal_inputs, token_ids, prefetch=True)
    iterator = DataIterator({"multimodal_train_inputs": lazy}, micro_batch_indices=[[0], [1]])
    iterator.get_next(["multimodal_train_inputs"])
    dropped = lazy._pending[1]
    assert not dropped.done()

    iterator.reset()
    gate.set()
    (got,) = iterator.get_next(["multimodal_train_inputs"])["multimodal_train_inputs"]

    assert dropped.result()[1] not in [event for kind, _, event in fake_cuda_streams if kind == "wait"]
    assert lazy._pending[0] == (1,) and lazy._pending[1] is not dropped
    assert got.keys() == expected[0].keys()
    assert all(torch.equal(got[key], expected[0][key]) for key in got)


def test_debug_train_data_can_pickle_rebuild_inputs_that_hold_a_thread_pool(fake_cuda_streams):
    lazy = _lazy([{"images": ["/s/a.png"]}], [[1, 9, 2]], prefetch=True)

    restored = pickle.loads(pickle.dumps(lazy))

    assert restored._multimodal_inputs == [{"images": ["/s/a.png"]}]
    assert restored._token_ids == [[1, 9, 2]]
