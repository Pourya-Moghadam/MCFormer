"""End-to-end observation alignment and coupling-target construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from mcformer.auxiliary.cache import ObservationBundle
from mcformer.auxiliary.coupling import CouplingTarget, compute_coupling_target
from mcformer.auxiliary.trajectories import (
    ObjectTrajectory,
    PositionTrajectory,
    dominant_hand_trajectory,
    primary_object_trajectory,
)
from mcformer.auxiliary.types import ObjectFrame, PersonPose, PoseFrame, TrackedObject, Wrist
from mcformer.data.transforms import SpatialTransform


class PipelineError(RuntimeError):
    """Raised when cached observations cannot be aligned to a sampled clip."""


@dataclass(frozen=True, slots=True)
class TargetSettings:
    pose_confidence: float = 0.30
    pose_max_gap: int = 5
    gaussian_sigma_frames: float = 1.0
    object_max_gap: int = 3
    minimum_track_coverage: float = 0.50
    minimum_track_mean_confidence: float = 0.25
    distance_threshold: float = 0.15
    epsilon: float = 1e-6

    def validate(self) -> None:
        if not 0 <= self.pose_confidence <= 1:
            raise PipelineError("pose_confidence must lie in [0,1]")
        if self.pose_max_gap < 0 or self.object_max_gap < 0:
            raise PipelineError("Interpolation gaps must be non-negative")
        if self.gaussian_sigma_frames < 0:
            raise PipelineError("Gaussian sigma must be non-negative")
        if not 0 <= self.minimum_track_coverage <= 1:
            raise PipelineError("minimum_track_coverage must lie in [0,1]")
        if not 0 <= self.minimum_track_mean_confidence <= 1:
            raise PipelineError("minimum_track_mean_confidence must lie in [0,1]")
        if self.distance_threshold < 0 or self.epsilon <= 0:
            raise PipelineError("Distance threshold and epsilon are invalid")


@dataclass(frozen=True, slots=True)
class SampleTarget:
    """Coupling target plus selected-trajectory diagnostic metadata."""

    coupling: CouplingTarget
    hand: PositionTrajectory
    object_trajectory: ObjectTrajectory | None
    settings: TargetSettings

    def as_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["targets"] = _auxiliary_targets(self)
        return value


def _position_channels(
    points: tuple[tuple[float, float] | None, ...], valid: tuple[bool, ...]
) -> tuple[list[float], list[bool]]:
    """Flatten x/y/dx/dy channels in channel-major order for E10."""

    x = [
        point[0] if point is not None and is_valid else 0.0
        for point, is_valid in zip(points, valid, strict=True)
    ]
    y = [
        point[1] if point is not None and is_valid else 0.0
        for point, is_valid in zip(points, valid, strict=True)
    ]
    position_mask = list(valid)
    dx = [0.0] * len(points)
    dy = [0.0] * len(points)
    difference_mask = [False] * len(points)
    for index in range(1, len(points)):
        previous, current = points[index - 1], points[index]
        if valid[index - 1] and valid[index] and previous is not None and current is not None:
            dx[index] = current[0] - previous[0]
            dy[index] = current[1] - previous[1]
            difference_mask[index] = True
    return x + y + dx + dy, position_mask + position_mask + difference_mask + difference_mask


def _auxiliary_targets(sample: SampleTarget) -> dict[str, dict[str, list[float] | list[bool]]]:
    coupling = sample.coupling
    temporal_mask = [False] * len(coupling.raw)
    for index in range(1, len(temporal_mask)):
        temporal_mask[index] = (
            coupling.hand_valid[index - 1]
            and coupling.hand_valid[index]
            and coupling.object_valid[index - 1]
            and coupling.object_valid[index]
        )
    spatial_target = [0.0] * len(sample.hand.points)
    spatial_mask = [False] * len(sample.hand.points)
    object_points = (
        sample.object_trajectory.points
        if sample.object_trajectory is not None
        else tuple(None for _ in sample.hand.points)
    )
    object_valid = (
        sample.object_trajectory.valid
        if sample.object_trajectory is not None
        else tuple(False for _ in sample.hand.points)
    )
    for index, (hand_point, object_point) in enumerate(
        zip(sample.hand.points, object_points, strict=True)
    ):
        if (
            sample.hand.valid[index]
            and object_valid[index]
            and hand_point is not None
            and object_point is not None
        ):
            spatial_target[index] = (
                (hand_point[0] - object_point[0]) ** 2 + (hand_point[1] - object_point[1]) ** 2
            ) ** 0.5
            spatial_mask[index] = True
    hand_values, hand_mask = _position_channels(sample.hand.points, sample.hand.valid)
    object_values, object_mask = _position_channels(object_points, object_valid)
    return {
        "temporal_gated": {
            "target": list(coupling.target),
            "mask": list(coupling.gate),
        },
        "temporal_ungated": {
            "target": list(coupling.raw),
            "mask": temporal_mask,
        },
        "spatial": {"target": spatial_target, "mask": spatial_mask},
        "hallucination": {
            "target": hand_values + object_values,
            "mask": hand_mask + object_mask,
        },
    }


def _sample_positions(bundle: ObservationBundle, frame_indices: Sequence[int]) -> tuple[int, ...]:
    positions_by_frame: dict[int, list[int]] = {}
    for position, frame_index in enumerate(bundle.frame_indices):
        positions_by_frame.setdefault(frame_index, []).append(position)
    positions: list[int] = []
    for frame_index in frame_indices:
        candidates = positions_by_frame.get(frame_index)
        if not candidates:
            raise PipelineError(
                f"Frame {frame_index} is absent from {bundle.mode!r} cache for {bundle.sample_id}"
            )
        positions.append(candidates[0])
    return tuple(positions)


def _transform_poses(
    frames: Sequence[PoseFrame], spatial: SpatialTransform
) -> tuple[PoseFrame, ...]:
    def transform(wrist: Wrist | None) -> Wrist | None:
        if wrist is None:
            return None
        point = spatial.transform_point(wrist.point)
        if not (0 <= point[0] < spatial.output_size and 0 <= point[1] < spatial.output_size):
            return None
        return Wrist(point=point, confidence=wrist.confidence)

    return tuple(
        PoseFrame(
            frame_index=frame.frame_index,
            people=tuple(
                PersonPose(
                    actor_id=person.actor_id,
                    left_wrist=transform(person.left_wrist),
                    right_wrist=transform(person.right_wrist),
                )
                for person in frame.people
            ),
        )
        for frame in frames
    )


def _transform_objects(
    frames: Sequence[ObjectFrame], spatial: SpatialTransform
) -> tuple[ObjectFrame, ...]:
    def transform(item: TrackedObject) -> TrackedObject | None:
        box = spatial.transform_box(item.box)
        clipped = (
            min(max(box[0], 0.0), spatial.output_size),
            min(max(box[1], 0.0), spatial.output_size),
            min(max(box[2], 0.0), spatial.output_size),
            min(max(box[3], 0.0), spatial.output_size),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return TrackedObject(
            track_id=item.track_id,
            class_id=item.class_id,
            class_name=item.class_name,
            confidence=item.confidence,
            box=clipped,
        )

    return tuple(
        ObjectFrame(
            frame_index=frame.frame_index,
            objects=tuple(
                transformed
                for item in frame.objects
                if (transformed := transform(item)) is not None
            ),
        )
        for frame in frames
    )


def build_sample_target(
    bundle: ObservationBundle,
    *,
    frame_indices: Sequence[int],
    spatial_transform: SpatialTransform,
    settings: TargetSettings | None = None,
    forced_track_id: int | None = None,
) -> SampleTarget:
    """Align raw observations to a sampled/augmented clip and construct its target."""

    bundle.validate()
    effective_settings = settings if settings is not None else TargetSettings()
    effective_settings.validate()
    if (
        spatial_transform.source_width != bundle.width
        or spatial_transform.source_height != bundle.height
    ):
        raise PipelineError("Spatial transform source dimensions do not match the cache")
    positions = _sample_positions(bundle, frame_indices)
    poses = _transform_poses(
        tuple(bundle.poses[position] for position in positions), spatial_transform
    )
    objects = _transform_objects(
        tuple(bundle.objects[position] for position in positions), spatial_transform
    )
    hand = dominant_hand_trajectory(
        poses,
        width=spatial_transform.output_size,
        height=spatial_transform.output_size,
        confidence_threshold=effective_settings.pose_confidence,
        max_gap=effective_settings.pose_max_gap,
        gaussian_sigma=effective_settings.gaussian_sigma_frames,
    )
    object_trajectory = primary_object_trajectory(
        objects,
        hand,
        width=spatial_transform.output_size,
        height=spatial_transform.output_size,
        minimum_coverage=effective_settings.minimum_track_coverage,
        minimum_mean_confidence=effective_settings.minimum_track_mean_confidence,
        max_gap=effective_settings.object_max_gap,
        forced_track_id=forced_track_id,
    )
    coupling = compute_coupling_target(
        hand,
        object_trajectory,
        distance_threshold=effective_settings.distance_threshold,
        epsilon=effective_settings.epsilon,
    )
    return SampleTarget(
        coupling=coupling,
        hand=hand,
        object_trajectory=object_trajectory,
        settings=effective_settings,
    )


def target_builder_from_cache(cache: Any, settings: TargetSettings) -> Any:
    """Create a :class:`VideoClipDataset` target callback from an observation cache."""

    def build(record: Any, indices: tuple[int, ...], spatial: SpatialTransform) -> dict[str, Any]:
        bundle = cache.read(record.sample_id)
        return build_sample_target(
            bundle,
            frame_indices=indices,
            spatial_transform=spatial,
            settings=settings,
        ).as_mapping()

    return build
