"""Presentation-order OpenCV video decoding for selected frame indices."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


class VideoDecodeError(RuntimeError):
    """Raised when requested video frames cannot be decoded."""


def decode_video_frames(path: str | Path, indices: Sequence[int]) -> Any:
    """Decode selected zero-based frames sequentially and return RGB uint8 data.

    The return type is a NumPy array shaped ``(T, H, W, 3)``. Imports are lazy so
    manifest and protocol tooling remains usable without video dependencies.
    """

    if not indices:
        raise VideoDecodeError("At least one frame index is required")
    if any(index < 0 for index in indices):
        raise VideoDecodeError("Frame indices must be non-negative")
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise VideoDecodeError("Video decoding requires NumPy and OpenCV") from error

    requested = set(indices)
    maximum = max(requested)
    decoded: dict[int, Any] = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoDecodeError(f"Could not open video: {path}")
    try:
        frame_index = 0
        while frame_index <= maximum:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in requested:
                decoded[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_index += 1
    finally:
        capture.release()
    missing = sorted(requested - decoded.keys())
    if missing:
        raise VideoDecodeError(f"Failed to decode frames {missing[:10]} from {path}")
    try:
        return np.stack([decoded[index] for index in indices], axis=0)
    except ValueError as error:
        raise VideoDecodeError(f"Video changes dimensions while decoding: {path}") from error


def decode_all_video_frames(path: str | Path) -> Any:
    """Decode all frames in presentation order as RGB uint8 data."""

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise VideoDecodeError("Video decoding requires NumPy and OpenCV") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoDecodeError(f"Could not open video: {path}")
    frames: list[Any] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise VideoDecodeError(f"Video contains no decodable frames: {path}")
    try:
        return np.stack(frames, axis=0)
    except ValueError as error:
        raise VideoDecodeError(f"Video changes dimensions while decoding: {path}") from error
