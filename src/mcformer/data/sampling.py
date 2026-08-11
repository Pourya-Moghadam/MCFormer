"""Deterministic temporal clip sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass


class SamplingError(ValueError):
    """Raised for invalid video or sampling parameters."""


@dataclass(frozen=True, slots=True)
class TemporalSample:
    """Source indices and padding indicators for one sampled clip."""

    indices: tuple[int, ...]
    padding_mask: tuple[bool, ...]
    start: int
    span: int


def sample_frame_indices(
    num_frames: int,
    *,
    clip_length: int = 32,
    stride: int = 2,
    training: bool,
    rng: random.Random | None = None,
) -> TemporalSample:
    """Sample a random training clip or centered evaluation clip.

    Out-of-range positions are clamped to the final decoded frame and marked as
    padding, as fixed by the release protocol.
    """

    if num_frames <= 0 or clip_length <= 0 or stride <= 0:
        raise SamplingError("num_frames, clip_length, and stride must be positive")
    span = (clip_length - 1) * stride + 1
    maximum_start = max(num_frames - span, 0)
    if training:
        if rng is None:
            raise SamplingError("Training sampling requires an explicit reproducible RNG")
        start = rng.randint(0, maximum_start)
    else:
        start = maximum_start // 2
    raw = tuple(start + offset * stride for offset in range(clip_length))
    return TemporalSample(
        indices=tuple(min(index, num_frames - 1) for index in raw),
        padding_mask=tuple(index >= num_frames for index in raw),
        start=start,
        span=span,
    )
