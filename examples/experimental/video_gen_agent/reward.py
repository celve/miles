"""The video-generation env's reward step: ``--custom-rm-path examples.experimental.video_gen_agent.reward.reward_func``.

The env records the submitted clip under ``metadata["video_gen"]``; this scores it in the rollout process, where
the reward pool lives. Every kept sample of one episode carries the same clip, so each distinct clip is scored once.
"""

import hashlib
import logging
from pathlib import Path

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class StubClipScorer:
    """Stands in for a reward pool scorer, with its ``score_batch(outputs, prompts)`` shape.

    The score is a fixed function of the clip's bytes in [0, 1), so equal clips score equally and a test can
    predict it; it says nothing about quality.
    """

    def score_batch(self, outputs: list[str], prompts: list[str]) -> list[float]:
        return [stub_score(output) for output in outputs]


def stub_score(path: str) -> float:
    return int.from_bytes(hashlib.sha256(Path(path).read_bytes()).digest()[:8], "big") / 2**64


_SCORER = StubClipScorer()


async def reward_func(args, samples: list[Sample], **kwargs) -> list[float | None]:
    """Score each sample's submitted clip against the user's request.

    A sample without a submitted clip scores 0: the policy ran out of turns or never submitted. A scorer failure
    scores None, so the sample is dropped rather than trained on as a failure.
    """
    submitted = [(sample.metadata.get("video_gen") or {}).get("submitted") for sample in samples]
    requests = {
        s["path"]: sample.metadata["video_gen"]["request"] for s, sample in zip(submitted, samples, strict=True) if s
    }
    try:
        scores = dict(zip(requests, _SCORER.score_batch(list(requests), list(requests.values())), strict=True))
    except Exception:
        logger.exception("scoring %d submitted clips failed", len(requests))
        scores = dict.fromkeys(requests)
    return [scores[s["path"]] if s else 0.0 for s in submitted]
