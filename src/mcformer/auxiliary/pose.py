"""NTU skeleton parsing and optional MMPose HRNet-W48 inference."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mcformer.auxiliary.types import PersonPose, PoseFrame, Wrist


class PoseError(RuntimeError):
    """Raised when pose data cannot be parsed or inferred."""


NTU_LEFT_WRIST_INDEX = 6
NTU_RIGHT_WRIST_INDEX = 10
COCO_WHOLEBODY_LEFT_WRIST_INDEX = 9
COCO_WHOLEBODY_RIGHT_WRIST_INDEX = 10


def _read_int(handle: Any, context: str) -> int:
    line = handle.readline()
    if not line:
        raise PoseError(f"Unexpected end of NTU skeleton while reading {context}")
    try:
        return int(line.strip())
    except ValueError as error:
        raise PoseError(f"Invalid integer for {context}: {line!r}") from error


def _ntu_wrist(tokens: list[str]) -> Wrist | None:
    if len(tokens) < 7:
        raise PoseError("NTU joint row has fewer than seven coordinates")
    color_x, color_y = float(tokens[5]), float(tokens[6])
    if not math.isfinite(color_x) or not math.isfinite(color_y):
        return None
    if color_x <= 0 or color_y <= 0:
        return None
    return Wrist(point=(color_x, color_y), confidence=1.0)


def read_ntu_skeleton(path: str | Path) -> tuple[PoseFrame, ...]:
    """Read official NTU `.skeleton` files using supplied color-plane projections.

    The files include projected color coordinates for each 3D joint. Using these
    fields is the canonical NTU projection used by this release.
    """

    frames: list[PoseFrame] = []
    with Path(path).open(encoding="utf-8") as handle:
        frame_count = _read_int(handle, "frame count")
        for frame_index in range(frame_count):
            body_count = _read_int(handle, f"body count at frame {frame_index}")
            people: list[PersonPose] = []
            for body_index in range(body_count):
                body_info = handle.readline().split()
                if not body_info:
                    raise PoseError(f"Missing body metadata at frame {frame_index}")
                actor_id = body_info[0] if body_info else str(body_index)
                joint_count = _read_int(handle, "joint count")
                joints = [handle.readline().split() for _ in range(joint_count)]
                if joint_count <= NTU_RIGHT_WRIST_INDEX:
                    raise PoseError(f"Too few joints at frame {frame_index}: {joint_count}")
                people.append(
                    PersonPose(
                        actor_id=actor_id,
                        left_wrist=_ntu_wrist(joints[NTU_LEFT_WRIST_INDEX]),
                        right_wrist=_ntu_wrist(joints[NTU_RIGHT_WRIST_INDEX]),
                    )
                )
            frames.append(PoseFrame(frame_index=frame_index, people=tuple(people)))
    return tuple(frames)


def _projection_matrix(path: str | Path) -> tuple[tuple[float, ...], ...]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        matrix = value["projection_matrix"]
        rows = tuple(tuple(float(item) for item in row) for row in matrix)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise PoseError(f"Invalid camera calibration file {path}: {error}") from error
    if len(rows) != 3 or any(len(row) != 4 for row in rows):
        raise PoseError("projection_matrix must have shape 3x4")
    if not all(math.isfinite(item) for row in rows for item in row):
        raise PoseError("projection_matrix contains a non-finite value")
    return rows


def _project_wrist(value: Any, matrix: tuple[tuple[float, ...], ...]) -> Wrist | None:
    if value is None:
        return None
    try:
        xyz = tuple(float(item) for item in value["xyz"])
        confidence = float(value.get("confidence", 1.0))
    except (KeyError, TypeError, ValueError) as error:
        raise PoseError(f"Invalid 3D wrist entry: {value!r}") from error
    if len(xyz) != 3 or not all(math.isfinite(item) for item in xyz):
        raise PoseError("A 3D wrist must contain three finite xyz coordinates")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise PoseError("3D wrist confidence must lie in [0,1]")
    homogeneous = (*xyz, 1.0)
    projected = tuple(sum(row[index] * homogeneous[index] for index in range(4)) for row in matrix)
    if abs(projected[2]) <= 1e-9:
        return None
    x, y = projected[0] / projected[2], projected[1] / projected[2]
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return Wrist(point=(x, y), confidence=confidence)


def read_projected_3d_pose(
    pose_path: str | Path, calibration_path: str | Path
) -> tuple[PoseFrame, ...]:
    """Project the release's portable 3D-wrist JSON contract into image pixels."""

    matrix = _projection_matrix(calibration_path)
    try:
        value = json.loads(Path(pose_path).read_text(encoding="utf-8"))
        raw_frames = value["frames"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise PoseError(f"Invalid 3D pose file {pose_path}: {error}") from error
    if not isinstance(raw_frames, list):
        raise PoseError("3D pose 'frames' must be a list")
    frames: list[PoseFrame] = []
    seen: set[int] = set()
    for raw_frame in raw_frames:
        try:
            frame_index = int(raw_frame["frame_index"])
            raw_people = raw_frame["people"]
        except (KeyError, TypeError, ValueError) as error:
            raise PoseError(f"Invalid 3D pose frame: {raw_frame!r}") from error
        if frame_index < 0 or frame_index in seen or not isinstance(raw_people, list):
            raise PoseError("Frame indices must be unique non-negative integers")
        seen.add(frame_index)
        people: list[PersonPose] = []
        actor_ids: set[str] = set()
        for raw_person in raw_people:
            try:
                actor_id = str(raw_person["actor_id"])
            except (KeyError, TypeError) as error:
                raise PoseError(f"Invalid 3D person: {raw_person!r}") from error
            if not actor_id or actor_id in actor_ids:
                raise PoseError("Actor IDs must be non-empty and unique within a frame")
            actor_ids.add(actor_id)
            people.append(
                PersonPose(
                    actor_id=actor_id,
                    left_wrist=_project_wrist(raw_person.get("left_wrist"), matrix),
                    right_wrist=_project_wrist(raw_person.get("right_wrist"), matrix),
                )
            )
        frames.append(PoseFrame(frame_index=frame_index, people=tuple(people)))
    return tuple(sorted(frames, key=lambda frame: frame.frame_index))


class HRNetWholeBodyEstimator:
    """MMPose adapter for HRNet-W48 COCO WholeBody at 384x288.

    MMPose is intentionally imported lazily because its wheel selection depends on
    the installed PyTorch/CUDA stack. A local config and checkpoint are required;
    this class never downloads weights implicitly.
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str,
        confidence_threshold: float = 0.30,
    ) -> None:
        config = Path(config_path).expanduser().resolve()
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not config.is_file() or not checkpoint.is_file():
            raise PoseError("HRNet config and checkpoint must be existing local files")
        if not 0 <= confidence_threshold <= 1:
            raise PoseError("confidence_threshold must lie in [0,1]")
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError as error:
            raise PoseError("HRNet inference requires a compatible mmpose installation") from error
        self._inferencer = MMPoseInferencer(
            pose2d=str(config),
            pose2d_weights=str(checkpoint),
            device=device,
        )
        self.confidence_threshold = confidence_threshold

    def infer(self, frames: Iterable[Any], frame_indices: Iterable[int]) -> tuple[PoseFrame, ...]:
        """Infer wrists for frames, preserving per-frame person ordering as actor IDs."""

        materialized_frames = list(frames)
        materialized_indices = list(frame_indices)
        output: list[PoseFrame] = []
        for frame, frame_index in zip(materialized_frames, materialized_indices, strict=True):
            result = next(self._inferencer(frame, return_vis=False, show=False))
            predictions = result.get("predictions", [])
            if len(predictions) == 1 and isinstance(predictions[0], list):
                predictions = predictions[0]
            people: list[PersonPose] = []
            for actor_index, prediction in enumerate(predictions):
                keypoints = prediction.get("keypoints", [])
                scores = prediction.get("keypoint_scores", [])
                people.append(
                    PersonPose(
                        actor_id=str(actor_index),
                        left_wrist=self._wrist(
                            keypoints,
                            scores,
                            COCO_WHOLEBODY_LEFT_WRIST_INDEX,
                            self.confidence_threshold,
                        ),
                        right_wrist=self._wrist(
                            keypoints,
                            scores,
                            COCO_WHOLEBODY_RIGHT_WRIST_INDEX,
                            self.confidence_threshold,
                        ),
                    )
                )
            output.append(PoseFrame(frame_index=frame_index, people=tuple(people)))
        if not materialized_frames:
            return ()
        height, width = materialized_frames[0].shape[:2]
        return associate_people(output, maximum_distance=0.25 * math.hypot(width, height))

    @staticmethod
    def _wrist(
        keypoints: Any, scores: Any, index: int, confidence_threshold: float
    ) -> Wrist | None:
        if len(keypoints) <= index or len(scores) <= index:
            return None
        x, y = float(keypoints[index][0]), float(keypoints[index][1])
        confidence = float(scores[index])
        if not all(math.isfinite(value) for value in (x, y, confidence)):
            return None
        if confidence < confidence_threshold:
            return None
        return Wrist(point=(x, y), confidence=confidence)


def _person_center(person: PersonPose) -> tuple[float, float] | None:
    wrists = [wrist for wrist in (person.left_wrist, person.right_wrist) if wrist is not None]
    if not wrists:
        return None
    return (
        sum(wrist.point[0] for wrist in wrists) / len(wrists),
        sum(wrist.point[1] for wrist in wrists) / len(wrists),
    )


def associate_people(
    frames: Iterable[PoseFrame], *, maximum_distance: float
) -> tuple[PoseFrame, ...]:
    """Greedily associate framewise poses without mixing actor trajectories."""

    if maximum_distance <= 0:
        raise PoseError("maximum_distance must be positive")
    next_track = 0
    previous: dict[str, tuple[float, float]] = {}
    output: list[PoseFrame] = []
    for frame in frames:
        candidates = [(person, _person_center(person)) for person in frame.people]
        pair_distances: list[tuple[float, str, int]] = []
        for track_id, old_center in previous.items():
            for index, (_, center) in enumerate(candidates):
                if center is not None:
                    distance = math.dist(old_center, center)
                    if distance <= maximum_distance:
                        pair_distances.append((distance, track_id, index))
        assignments: dict[int, str] = {}
        used_tracks: set[str] = set()
        for _, track_id, index in sorted(pair_distances):
            if index not in assignments and track_id not in used_tracks:
                assignments[index] = track_id
                used_tracks.add(track_id)
        people: list[PersonPose] = []
        updated: dict[str, tuple[float, float]] = {}
        for index, (person, center) in enumerate(candidates):
            assigned_track_id = assignments.get(index)
            if assigned_track_id is None:
                assigned_track_id = f"actor-{next_track:03d}"
                next_track += 1
            people.append(
                PersonPose(
                    actor_id=assigned_track_id,
                    left_wrist=person.left_wrist,
                    right_wrist=person.right_wrist,
                )
            )
            if center is not None:
                updated[assigned_track_id] = center
        previous = updated
        output.append(PoseFrame(frame_index=frame.frame_index, people=tuple(people)))
    return tuple(output)
