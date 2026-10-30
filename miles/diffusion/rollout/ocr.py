"""OCR reward of Flow-GRPO: 1 - normalized edit distance between the quoted prompt text and PaddleOCR's reading."""

import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

logger = logging.getLogger(__name__)


def ocr_target_text(prompt: str) -> str:
    """The text an OCR prompt asks to render: its first double-quoted span."""
    return prompt.split('"')[1]


def ocr_reward(recognized_text: str, target_text: str) -> float:
    from Levenshtein import distance

    recognized = recognized_text.replace(" ", "").lower()
    target = target_text.replace(" ", "").lower()
    dist = 0 if target in recognized else distance(recognized, target)
    # Many unrelated recognized characters cost at most the whole target.
    return 1 - min(dist, len(target)) / len(target)


def image_to_rgb_uint8(cfhw: torch.Tensor) -> np.ndarray:
    """First frame of a ``[C, F, H, W]`` image as HWC uint8, rounded as flow_grpo feeds PaddleOCR."""
    image = cfhw[:, 0].detach().cpu().float()
    if float(image.max()) <= 1.0 + 1e-3:
        image = (image * 255.0).round()
    return np.ascontiguousarray(image.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy())


class OcrScorer:
    def __init__(self) -> None:
        _patch_numpy_for_imgaug()
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)

    def __call__(self, image: np.ndarray, prompt: str) -> float:
        target = ocr_target_text(prompt)
        try:
            result = self._ocr.ocr(image, cls=False)
            recognized = "".join(res[1][0] if res[1][1] > 0 else "" for res in result[0]) if result[0] else ""
        except Exception as e:
            logger.warning(f"OCR processing failed: {e}")
            return 0.0
        return ocr_reward(recognized, target)


def _patch_numpy_for_imgaug() -> None:
    # paddleocr 2.x imports imgaug, which still reads np.sctypes (removed in numpy 2).
    if not hasattr(np, "sctypes"):  # config-access-exempt: numpy 2 removed sctypes
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [bool, object, bytes, str, np.void],
        }


_worker_scorer: OcrScorer | None = None


def _init_worker() -> None:
    global _worker_scorer
    _worker_scorer = OcrScorer()


def _score_in_worker(image: np.ndarray, prompt: str) -> float:
    return _worker_scorer(image, prompt)


class OcrRewardPool:
    """CPU PaddleOCR worker processes, one image per task."""

    def __init__(self, num_workers: int) -> None:
        self._executor = ProcessPoolExecutor(
            max_workers=num_workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker
        )

    async def score(self, images: list[np.ndarray], prompts: list[str]) -> list[float]:
        loop = asyncio.get_running_loop()
        return list(
            await asyncio.gather(
                *(
                    loop.run_in_executor(self._executor, _score_in_worker, image, prompt)
                    for image, prompt in zip(images, prompts, strict=True)
                )
            )
        )

    def shutdown(self) -> None:
        self._executor.shutdown(cancel_futures=True)
