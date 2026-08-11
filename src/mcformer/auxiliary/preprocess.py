"""Video-level auxiliary observation extraction orchestration."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from mcformer.auxiliary.cache import ObservationBundle
from mcformer.auxiliary.pose import read_ntu_skeleton, read_projected_3d_pose
from mcformer.auxiliary.types import ObjectFrame, PoseFrame
from mcformer.data.manifest import SampleRecord
from mcformer.data.sampling import sample_frame_indices
from mcformer.data.video import decode_all_video_frames, decode_video_frames


class PoseBackend(Protocol):
    """Interface implemented by training-only wrist estimators."""

    def infer(
        self, frames: Sequence[Any], frame_indices: Sequence[int]
    ) -> tuple[PoseFrame, ...]: ...


class ObjectBackend(Protocol):
    """Interface implemented by training-only detector/trackers."""

    def track(
        self, frames: Sequence[Any], frame_indices: Sequence[int]
    ) -> tuple[ObjectFrame, ...]: ...


class PreprocessingError(RuntimeError):
    """Raised when video and auxiliary inputs cannot be aligned."""


def _path(root: Path, value: str | None, kind: str) -> Path:
    if value is None:
        raise PreprocessingError(f"Sample does not provide a {kind} path")
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _select_pose_frames(
    all_frames: Sequence[PoseFrame], frame_indices: Sequence[int]
) -> tuple[PoseFrame, ...]:
    by_index = {frame.frame_index: frame for frame in all_frames}
    selected: list[PoseFrame] = []
    for index in frame_indices:
        frame = by_index.get(index)
        if frame is None:
            selected.append(PoseFrame(frame_index=index, people=()))
        else:
            selected.append(PoseFrame(frame_index=index, people=frame.people))
    return tuple(selected)


def extract_observation_bundle(
    record: SampleRecord,
    *,
    root: str | Path,
    object_backend: ObjectBackend,
    pose_backend: PoseBackend | None = None,
    pose_source: str = "hrnet",
    mode: str = "native_frames",
    clip_length: int = 32,
    stride: int = 2,
    stage_timings: dict[str, float] | None = None,
) -> ObservationBundle:
    """Decode a video and extract aligned pose plus tracked-object observations."""

    if record.num_frames is None or record.width is None or record.height is None:
        raise PreprocessingError(f"Missing video metadata for {record.sample_id}")
    dataset_root = Path(root).expanduser().resolve()
    video_path = _path(dataset_root, record.rgb_path, "RGB")
    decode_start = time.perf_counter()
    if mode == "native_frames":
        frames = decode_all_video_frames(video_path)
        if len(frames) != record.num_frames:
            raise PreprocessingError(
                f"Decoded {len(frames)} frames but manifest lists {record.num_frames} "
                f"for {record.sample_id}"
            )
        frame_indices = tuple(range(len(frames)))
    elif mode == "paper_cost_mode":
        temporal = sample_frame_indices(
            record.num_frames,
            clip_length=clip_length,
            stride=stride,
            training=False,
        )
        frame_indices = temporal.indices
        frames = decode_video_frames(video_path, frame_indices)
    else:
        raise PreprocessingError(f"Unknown preprocessing mode: {mode}")
    if stage_timings is not None:
        stage_timings["decode_seconds"] = time.perf_counter() - decode_start

    if frames.shape[2] != record.width or frames.shape[1] != record.height:
        raise PreprocessingError(
            f"Decoded dimensions disagree with manifest for {record.sample_id}"
        )
    if pose_source == "hrnet":
        if pose_backend is None:
            raise PreprocessingError("HRNet pose source requires a pose backend")
        pose_start = time.perf_counter()
        poses = pose_backend.infer(frames, frame_indices)
        if stage_timings is not None:
            stage_timings["hrnet_seconds"] = time.perf_counter() - pose_start
        pose_backend_name = type(pose_backend).__name__
    elif pose_source == "ntu_projected_3d":
        skeleton_path = _path(dataset_root, record.skeleton_path, "skeleton")
        poses = _select_pose_frames(read_ntu_skeleton(skeleton_path), frame_indices)
        pose_backend_name = "ntu_projected_color_coordinates"
    elif pose_source == "projected_3d_json":
        skeleton_path = _path(dataset_root, record.skeleton_path, "3D pose")
        calibration_path = _path(dataset_root, record.calibration_path, "calibration")
        poses = _select_pose_frames(
            read_projected_3d_pose(skeleton_path, calibration_path), frame_indices
        )
        pose_backend_name = "portable_3d_projection"
    else:
        raise PreprocessingError(f"Unknown pose source: {pose_source}")
    object_start = time.perf_counter()
    objects = object_backend.track(frames, frame_indices)
    if stage_timings is not None:
        stage_timings["object_pipeline_seconds"] = time.perf_counter() - object_start
        backend_timing = getattr(object_backend, "last_timing", None)
        if isinstance(backend_timing, dict):
            stage_timings.update(
                {str(name): float(value) for name, value in backend_timing.items()}
            )
    bundle = ObservationBundle(
        sample_id=record.sample_id,
        width=record.width,
        height=record.height,
        frame_indices=frame_indices,
        poses=poses,
        objects=objects,
        pose_backend=pose_backend_name,
        object_backend=type(object_backend).__name__,
        mode=mode,
    )
    bundle.validate()
    return bundle
