"""Deterministic auxiliary-signal corruptions used by experiment E12."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from mcformer.auxiliary.trajectories import PositionTrajectory, primary_object_trajectory
from mcformer.auxiliary.types import ObjectFrame, PersonPose, PoseFrame, TrackedObject, Wrist


def add_wrist_noise(
    frames: Sequence[PoseFrame],
    *,
    sigma_diagonal: float,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[PoseFrame, ...]:
    """Add isotropic Gaussian wrist noise in pixels relative to frame diagonal."""

    if sigma_diagonal < 0:
        raise ValueError("sigma_diagonal must be non-negative")
    sigma_pixels = sigma_diagonal * math.hypot(width, height)

    def perturb(wrist: Wrist | None) -> Wrist | None:
        if wrist is None:
            return None
        return Wrist(
            point=(
                wrist.point[0] + rng.gauss(0, sigma_pixels),
                wrist.point[1] + rng.gauss(0, sigma_pixels),
            ),
            confidence=wrist.confidence,
        )

    return tuple(
        PoseFrame(
            frame_index=frame.frame_index,
            people=tuple(
                PersonPose(
                    actor_id=person.actor_id,
                    left_wrist=perturb(person.left_wrist),
                    right_wrist=perturb(person.right_wrist),
                )
                for person in frame.people
            ),
        )
        for frame in frames
    )


def drop_object_detections(
    frames: Sequence[ObjectFrame], *, probability: float, rng: random.Random
) -> tuple[ObjectFrame, ...]:
    """Independently drop tracked object observations with a fixed probability."""

    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0,1]")
    return tuple(
        ObjectFrame(
            frame_index=frame.frame_index,
            objects=tuple(item for item in frame.objects if rng.random() >= probability),
        )
        for frame in frames
    )


def occlude_track(
    frames: Sequence[ObjectFrame],
    *,
    track_id: int,
    length: int,
    rng: random.Random,
) -> tuple[ObjectFrame, ...]:
    """Remove one contiguous segment from a selected object track."""

    if length <= 0 or length > len(frames):
        raise ValueError("Occlusion length must fit within the sequence")
    start = rng.randint(0, len(frames) - length)
    end = start + length
    return tuple(
        ObjectFrame(
            frame_index=frame.frame_index,
            objects=tuple(
                item
                for item in frame.objects
                if not (start <= position < end and item.track_id == track_id)
            ),
        )
        for position, frame in enumerate(frames)
    )


def relabel_track(
    frames: Sequence[ObjectFrame], *, source_track_id: int, replacement_track_id: int
) -> tuple[ObjectFrame, ...]:
    """Relabel a source track as another ID for controlled association tests."""

    return tuple(
        ObjectFrame(
            frame_index=frame.frame_index,
            objects=tuple(
                TrackedObject(
                    track_id=(
                        replacement_track_id if item.track_id == source_track_id else item.track_id
                    ),
                    class_id=item.class_id,
                    class_name=item.class_name,
                    confidence=item.confidence,
                    box=item.box,
                )
                for item in frame.objects
            ),
        )
        for frame in frames
    )


def nearest_alternative_track(
    frames: Sequence[ObjectFrame],
    hand: PositionTrajectory,
    *,
    selected_track_id: int,
    width: int,
    height: int,
    minimum_coverage: float = 0.50,
    minimum_mean_confidence: float = 0.25,
    max_gap: int = 3,
) -> int | None:
    """Return the proximity-ranked eligible track other than the selected track."""

    track_ids = sorted(
        {
            item.track_id
            for frame in frames
            for item in frame.objects
            if item.track_id != selected_track_id
        }
    )
    candidates = [
        trajectory
        for track_id in track_ids
        if (
            trajectory := primary_object_trajectory(
                frames,
                hand,
                width=width,
                height=height,
                minimum_coverage=minimum_coverage,
                minimum_mean_confidence=minimum_mean_confidence,
                max_gap=max_gap,
                forced_track_id=track_id,
            )
        )
        is not None
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item.median_hand_box_distance,
            -item.mean_confidence,
            item.track_id,
        ),
    ).track_id
