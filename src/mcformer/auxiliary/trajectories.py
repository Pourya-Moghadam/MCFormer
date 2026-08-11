"""Dominant-wrist and primary-object trajectory construction."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from mcformer.auxiliary.types import Box, ObjectFrame, Point, PoseFrame, TrackedObject, Wrist


class TrajectoryError(ValueError):
    """Raised when trajectory inputs or hyperparameters are invalid."""


@dataclass(frozen=True, slots=True)
class PositionTrajectory:
    """A normalized position series with observed/interpolated validity."""

    points: tuple[Point | None, ...]
    valid: tuple[bool, ...]
    observed: tuple[bool, ...]
    selected_actor_id: str | None = None
    selected_wrist: str | None = None


@dataclass(frozen=True, slots=True)
class ObjectTrajectory:
    """Selected normalized object-center/box trajectory and track metadata."""

    track_id: int
    class_id: int
    class_name: str
    points: tuple[Point | None, ...]
    boxes: tuple[Box | None, ...]
    valid: tuple[bool, ...]
    observed: tuple[bool, ...]
    confidences: tuple[float | None, ...]
    coverage: float
    mean_confidence: float
    median_hand_box_distance: float


def _squared_distance(left: Point, right: Point) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _motion_energy(values: Sequence[Wrist | None], threshold: float) -> float:
    energy = 0.0
    for previous, current in pairwise(values):
        if (
            previous is not None
            and current is not None
            and previous.confidence >= threshold
            and current.confidence >= threshold
        ):
            energy += _squared_distance(previous.point, current.point)
    return energy


def _interpolate_points(
    values: Sequence[Point | None], max_gap: int
) -> tuple[list[Point | None], list[bool], list[bool]]:
    result = list(values)
    observed = [value is not None for value in values]
    valid = observed.copy()
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        start = index
        while index < len(result) and result[index] is None:
            index += 1
        gap = index - start
        if start == 0 or index == len(result) or gap > max_gap:
            continue
        left = result[start - 1]
        right = result[index]
        if left is None or right is None:
            continue
        for offset in range(gap):
            fraction = (offset + 1) / (gap + 1)
            result[start + offset] = (
                left[0] + fraction * (right[0] - left[0]),
                left[1] + fraction * (right[1] - left[1]),
            )
            valid[start + offset] = True
    return result, valid, observed


def _interpolate_boxes(
    values: Sequence[Box | None], max_gap: int
) -> tuple[list[Box | None], list[bool], list[bool]]:
    result = list(values)
    observed = [value is not None for value in values]
    valid = observed.copy()
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        start = index
        while index < len(result) and result[index] is None:
            index += 1
        gap = index - start
        if start == 0 or index == len(result) or gap > max_gap:
            continue
        left = result[start - 1]
        right = result[index]
        if left is None or right is None:
            continue
        for offset in range(gap):
            fraction = (offset + 1) / (gap + 1)
            result[start + offset] = (
                left[0] + fraction * (right[0] - left[0]),
                left[1] + fraction * (right[1] - left[1]),
                left[2] + fraction * (right[2] - left[2]),
                left[3] + fraction * (right[3] - left[3]),
            )
            valid[start + offset] = True
    return result, valid, observed


def _gaussian_smooth(
    values: Sequence[Point | None], valid: Sequence[bool], sigma: float
) -> list[Point | None]:
    if sigma <= 0:
        return list(values)
    radius = max(1, math.ceil(3 * sigma))
    output: list[Point | None] = [None] * len(values)
    for index, point in enumerate(values):
        if point is None or not valid[index]:
            continue
        segment_start = index
        while segment_start > 0 and valid[segment_start - 1]:
            segment_start -= 1
        segment_end = index
        while segment_end + 1 < len(valid) and valid[segment_end + 1]:
            segment_end += 1
        weighted_x = weighted_y = weight_sum = 0.0
        for neighbor in range(
            max(segment_start, index - radius), min(segment_end, index + radius) + 1
        ):
            neighbor_point = values[neighbor]
            if neighbor_point is None:
                continue
            weight = math.exp(-0.5 * ((neighbor - index) / sigma) ** 2)
            weighted_x += weight * neighbor_point[0]
            weighted_y += weight * neighbor_point[1]
            weight_sum += weight
        output[index] = (weighted_x / weight_sum, weighted_y / weight_sum)
    return output


def dominant_hand_trajectory(
    frames: Sequence[PoseFrame],
    *,
    width: int,
    height: int,
    confidence_threshold: float = 0.30,
    max_gap: int = 5,
    gaussian_sigma: float = 1.0,
) -> PositionTrajectory:
    """Select one actor and dominant wrist, then normalize/interpolate/smooth it."""

    if width <= 0 or height <= 0 or max_gap < 0 or gaussian_sigma < 0:
        raise TrajectoryError("Invalid hand trajectory dimensions or hyperparameters")
    actor_wrist: dict[str, dict[str, list[Wrist | None]]] = defaultdict(
        lambda: {"left": [None] * len(frames), "right": [None] * len(frames)}
    )
    for position, frame in enumerate(frames):
        for person in frame.people:
            actor_wrist[person.actor_id]["left"][position] = person.left_wrist
            actor_wrist[person.actor_id]["right"][position] = person.right_wrist
    if not actor_wrist:
        return PositionTrajectory(
            points=tuple([None] * len(frames)),
            valid=tuple([False] * len(frames)),
            observed=tuple([False] * len(frames)),
        )

    energies: dict[str, dict[str, float]] = {
        actor_id: {
            side: _motion_energy(wrists, confidence_threshold) for side, wrists in sides.items()
        }
        for actor_id, sides in actor_wrist.items()
    }
    actor_id = sorted(
        energies,
        key=lambda candidate: (-sum(energies[candidate].values()), candidate),
    )[0]
    side = sorted(("left", "right"), key=lambda name: (-energies[actor_id][name], name))[0]
    selected = actor_wrist[actor_id][side]
    diagonal = math.hypot(width, height)
    normalized = [
        (wrist.point[0] / diagonal, wrist.point[1] / diagonal)
        if wrist is not None and wrist.confidence >= confidence_threshold
        else None
        for wrist in selected
    ]
    interpolated, valid, observed = _interpolate_points(normalized, max_gap)
    smoothed = _gaussian_smooth(interpolated, valid, gaussian_sigma)
    return PositionTrajectory(
        points=tuple(smoothed),
        valid=tuple(valid),
        observed=tuple(observed),
        selected_actor_id=actor_id,
        selected_wrist=side,
    )


def _point_to_box_distance(point: Point, box: Box) -> float:
    x = min(max(point[0], box[0]), box[2])
    y = min(max(point[1], box[1]), box[3])
    return math.sqrt(_squared_distance(point, (x, y)))


def primary_object_trajectory(
    frames: Sequence[ObjectFrame],
    hand: PositionTrajectory,
    *,
    width: int,
    height: int,
    minimum_coverage: float = 0.50,
    minimum_mean_confidence: float = 0.25,
    max_gap: int = 3,
    forced_track_id: int | None = None,
) -> ObjectTrajectory | None:
    """Select the eligible proximity-ranked track independently of motion direction."""

    if len(frames) != len(hand.points):
        raise TrajectoryError("Object frames and hand trajectory must have equal length")
    if not 0 <= minimum_coverage <= 1 or not 0 <= minimum_mean_confidence <= 1:
        raise TrajectoryError("Coverage and confidence thresholds must lie in [0,1]")
    diagonal = math.hypot(width, height)
    tracks: dict[int, list[tuple[int, TrackedObject]]] = defaultdict(list)
    for index, frame in enumerate(frames):
        for observation in frame.objects:
            tracks[observation.track_id].append((index, observation))

    candidates: list[ObjectTrajectory] = []
    for track_id, observations in tracks.items():
        if forced_track_id is not None and track_id != forced_track_id:
            continue
        coverage = len({index for index, _ in observations}) / len(frames) if frames else 0.0
        mean_confidence = statistics.fmean(
            observation.confidence for _, observation in observations
        )
        if coverage < minimum_coverage or mean_confidence < minimum_mean_confidence:
            continue
        raw_boxes: list[Box | None] = [None] * len(frames)
        confidences: list[float | None] = [None] * len(frames)
        representative = max(
            (observation for _, observation in observations),
            key=lambda observation: observation.confidence,
        )
        for index, observation in observations:
            raw_boxes[index] = (
                observation.box[0] / diagonal,
                observation.box[1] / diagonal,
                observation.box[2] / diagonal,
                observation.box[3] / diagonal,
            )
            confidences[index] = observation.confidence
        boxes, valid, observed = _interpolate_boxes(raw_boxes, max_gap)
        points: list[Point | None] = [
            ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) if box is not None else None
            for box in boxes
        ]
        distances = [
            _point_to_box_distance(hand_point, box)
            for hand_point, box, hand_valid, object_valid in zip(
                hand.points, boxes, hand.valid, valid, strict=True
            )
            if hand_point is not None and box is not None and hand_valid and object_valid
        ]
        if not distances:
            continue
        candidates.append(
            ObjectTrajectory(
                track_id=track_id,
                class_id=representative.class_id,
                class_name=representative.class_name,
                points=tuple(points),
                boxes=tuple(boxes),
                valid=tuple(valid),
                observed=tuple(observed),
                confidences=tuple(confidences),
                coverage=coverage,
                mean_confidence=mean_confidence,
                median_hand_box_distance=statistics.median(distances),
            )
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            candidate.median_hand_box_distance,
            -candidate.mean_confidence,
            candidate.track_id,
        ),
    )
