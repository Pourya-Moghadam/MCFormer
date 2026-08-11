"""Serializable observation types shared by preprocessing backends."""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Wrist:
    """One image-plane wrist observation."""

    point: Point
    confidence: float


@dataclass(frozen=True, slots=True)
class PersonPose:
    """Left/right wrists for one consistently identified person."""

    actor_id: str
    left_wrist: Wrist | None
    right_wrist: Wrist | None


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """All person poses observed at a source frame."""

    frame_index: int
    people: tuple[PersonPose, ...]


@dataclass(frozen=True, slots=True)
class TrackedObject:
    """One non-person object observation associated with a track ID."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: Box


@dataclass(frozen=True, slots=True)
class ObjectFrame:
    """Tracked objects observed at a source frame."""

    frame_index: int
    objects: tuple[TrackedObject, ...]
