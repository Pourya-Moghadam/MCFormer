"""Framework-light RGB clip dataset with deterministic epoch-aware sampling."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from torch.utils.data import Dataset

from mcformer.data.manifest import Manifest, SampleRecord
from mcformer.data.sampling import TemporalSample, sample_frame_indices
from mcformer.data.transforms import SpatialTransform, apply_rgb_transform, make_spatial_transform
from mcformer.data.video import decode_video_frames


class DatasetError(RuntimeError):
    """Raised when a manifest record cannot produce a valid clip."""


@dataclass(frozen=True, slots=True)
class ClipSample:
    """One normalized RGB clip and its auditable sampling metadata."""

    video: Any
    label: int
    sample_id: str
    frame_indices: tuple[int, ...]
    padding_mask: tuple[bool, ...]
    spatial_transform: SpatialTransform
    auxiliary: Mapping[str, Any] | None


TargetBuilder = Callable[
    [SampleRecord, tuple[int, ...], SpatialTransform], Mapping[str, Any] | None
]


def _stable_seed(base_seed: int, epoch: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{epoch}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class VideoClipDataset(Dataset[ClipSample]):
    """Load RGB clips from a split while preserving deterministic randomness."""

    def __init__(
        self,
        *,
        manifest: Manifest,
        sample_ids: Sequence[str],
        root: str | Path,
        training: bool,
        seed: int,
        clip_length: int = 32,
        stride: int = 2,
        input_size: int = 224,
        resize_short_side: int = 256,
        target_builder: TargetBuilder | None = None,
    ) -> None:
        self.records = tuple(manifest.by_id(sample_id) for sample_id in sample_ids)
        self.root = Path(root).expanduser().resolve()
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.clip_length = clip_length
        self.stride = stride
        self.input_size = input_size
        self.resize_short_side = resize_short_side
        self.target_builder = target_builder

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch incorporated into per-sample deterministic RNG state."""

        if epoch < 0:
            raise DatasetError("epoch must be non-negative")
        self.epoch = epoch

    def _video_path(self, record: SampleRecord) -> Path:
        path = Path(record.rgb_path).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @overload
    def __getitem__(self, index: int) -> ClipSample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ClipSample, ...]: ...

    def __getitem__(self, index: int | slice) -> ClipSample | tuple[ClipSample, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        record = self.records[index]
        if record.num_frames is None or record.width is None or record.height is None:
            raise DatasetError(f"Missing video metadata for {record.sample_id}")
        rng = random.Random(_stable_seed(self.seed, self.epoch, record.sample_id))
        temporal: TemporalSample = sample_frame_indices(
            record.num_frames,
            clip_length=self.clip_length,
            stride=self.stride,
            training=self.training,
            rng=rng,
        )
        spatial = make_spatial_transform(
            width=record.width,
            height=record.height,
            training=self.training,
            rng=rng,
            resize_short_side=self.resize_short_side,
            output_size=self.input_size,
        )
        frames = decode_video_frames(self._video_path(record), temporal.indices)
        video = apply_rgb_transform(frames, spatial)
        auxiliary = (
            self.target_builder(record, temporal.indices, spatial)
            if self.target_builder is not None
            else None
        )
        return ClipSample(
            video=video,
            label=record.label_id,
            sample_id=record.sample_id,
            frame_indices=temporal.indices,
            padding_mask=temporal.padding_mask,
            spatial_transform=spatial,
            auxiliary=auxiliary,
        )
