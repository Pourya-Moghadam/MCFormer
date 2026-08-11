"""Framewise proximity-gated hand-object directional coupling target."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mcformer.auxiliary.trajectories import ObjectTrajectory, PositionTrajectory


@dataclass(frozen=True, slots=True)
class CouplingTarget:
    """Raw cosine coupling, loss gate, target, and target-coverage statistics."""

    raw: tuple[float, ...]
    gate: tuple[bool, ...]
    target: tuple[float, ...]
    hand_valid: tuple[bool, ...]
    object_valid: tuple[bool, ...]
    coverage: float
    selected_track_id: int | None


def _subtract(current: tuple[float, float], previous: tuple[float, float]) -> tuple[float, float]:
    return current[0] - previous[0], current[1] - previous[1]


def compute_coupling_target(
    hand: PositionTrajectory,
    object_trajectory: ObjectTrajectory | None,
    *,
    distance_threshold: float = 0.15,
    epsilon: float = 1e-6,
) -> CouplingTarget:
    """Compute cosine co-motion only at valid, proximal displacement pairs."""

    length = len(hand.points)
    if distance_threshold < 0 or epsilon <= 0:
        raise ValueError("distance_threshold must be non-negative and epsilon positive")
    if object_trajectory is None:
        empty_float = tuple(0.0 for _ in range(length))
        empty_bool = tuple(False for _ in range(length))
        return CouplingTarget(
            raw=empty_float,
            gate=empty_bool,
            target=empty_float,
            hand_valid=hand.valid,
            object_valid=empty_bool,
            coverage=0.0,
            selected_track_id=None,
        )
    if len(object_trajectory.points) != length:
        raise ValueError("Hand and object trajectories must have equal length")

    raw = [0.0] * length
    gate = [False] * length
    for index in range(1, length):
        hand_current = hand.points[index]
        hand_previous = hand.points[index - 1]
        object_current = object_trajectory.points[index]
        object_previous = object_trajectory.points[index - 1]
        if not (
            hand.valid[index]
            and hand.valid[index - 1]
            and object_trajectory.valid[index]
            and object_trajectory.valid[index - 1]
        ):
            continue
        if (
            hand_current is None
            or hand_previous is None
            or object_current is None
            or object_previous is None
        ):
            continue
        hand_delta = _subtract(hand_current, hand_previous)
        object_delta = _subtract(object_current, object_previous)
        hand_norm = math.hypot(*hand_delta)
        object_norm = math.hypot(*object_delta)
        if hand_norm >= epsilon and object_norm >= epsilon:
            raw[index] = (hand_delta[0] * object_delta[0] + hand_delta[1] * object_delta[1]) / (
                hand_norm * object_norm + epsilon
            )
        distance = math.sqrt(
            (hand_current[0] - object_current[0]) ** 2 + (hand_current[1] - object_current[1]) ** 2
        )
        gate[index] = distance < distance_threshold
    target = tuple(value if valid else 0.0 for value, valid in zip(raw, gate, strict=True))
    return CouplingTarget(
        raw=tuple(raw),
        gate=tuple(gate),
        target=target,
        hand_valid=hand.valid,
        object_valid=object_trajectory.valid,
        coverage=sum(gate) / length if length else 0.0,
        selected_track_id=object_trajectory.track_id,
    )
