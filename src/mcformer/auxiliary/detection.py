"""Optional Ultralytics YOLOv8x + ByteTrack inference adapter."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcformer.auxiliary.types import ObjectFrame, TrackedObject


class DetectionError(RuntimeError):
    """Raised when detection or tracking cannot be executed safely."""


@dataclass(frozen=True, slots=True)
class ByteTrackSettings:
    high_threshold: float = 0.50
    low_threshold: float = 0.10
    match_threshold: float = 0.80
    buffer_frames: int = 30

    def validate(self) -> None:
        if not 0 <= self.low_threshold <= self.high_threshold <= 1:
            raise DetectionError("ByteTrack thresholds must satisfy 0 <= low <= high <= 1")
        if not 0 <= self.match_threshold <= 1 or self.buffer_frames <= 0:
            raise DetectionError("Invalid ByteTrack match threshold or buffer")


class YOLOv8ByteTrack:
    """Run a local YOLOv8x checkpoint with a manuscript-matched ByteTrack config."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str,
        detector_confidence: float = 0.25,
        nms_iou: float = 0.70,
        settings: ByteTrackSettings | None = None,
    ) -> None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise DetectionError("YOLO checkpoint must be an existing local file")
        if not 0 <= detector_confidence <= 1 or not 0 <= nms_iou <= 1:
            raise DetectionError("Detector confidence and NMS IoU must lie in [0,1]")
        effective_settings = settings if settings is not None else ByteTrackSettings()
        effective_settings.validate()
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise DetectionError("Detection requires the pinned ultralytics package") from error
        self.model = YOLO(str(checkpoint))
        self.device = device
        self.detector_confidence = detector_confidence
        self.nms_iou = nms_iou
        self.settings = effective_settings
        self.last_timing: dict[str, float] | None = None

    def track(self, frames: Iterable[Any], frame_indices: Iterable[int]) -> tuple[ObjectFrame, ...]:
        """Detect and associate non-person objects over one video sequence."""

        materialized_frames = list(frames)
        materialized_indices = list(frame_indices)
        if len(materialized_frames) != len(materialized_indices):
            raise DetectionError("frames and frame_indices must have equal length")
        names = self.model.names
        allowed_classes = [
            int(class_id) for class_id, name in names.items() if str(name).lower() != "person"
        ]
        tracker_yaml = (
            "tracker_type: bytetrack\n"
            f"track_high_thresh: {self.settings.high_threshold}\n"
            f"track_low_thresh: {self.settings.low_threshold}\n"
            f"new_track_thresh: {self.settings.high_threshold}\n"
            f"track_buffer: {self.settings.buffer_frames}\n"
            f"match_thresh: {self.settings.match_threshold}\n"
            "fuse_score: true\n"
        )
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as tracker_file:
            tracker_file.write(tracker_yaml)
            tracker_file.flush()
            results = list(
                self.model.track(
                    source=materialized_frames,
                    stream=True,
                    tracker=tracker_file.name,
                    persist=True,
                    conf=self.settings.low_threshold,
                    iou=self.nms_iou,
                    classes=allowed_classes,
                    device=self.device,
                    verbose=False,
                )
            )
        total_seconds = time.perf_counter() - start
        if len(results) != len(materialized_indices):
            raise DetectionError("Ultralytics returned a different number of frames")
        detector_seconds = sum(
            float(getattr(result, "speed", {}).get("inference", 0.0)) / 1000.0 for result in results
        )
        if detector_seconds <= 0 or detector_seconds > total_seconds:
            detector_seconds = total_seconds
        self.last_timing = {
            "yolov8_inference_seconds": detector_seconds,
            "tracking_and_adapter_seconds": max(0.0, total_seconds - detector_seconds),
            "object_pipeline_seconds": total_seconds,
        }
        output: list[ObjectFrame] = []
        for frame_index, result in zip(materialized_indices, results, strict=True):
            tracked: list[TrackedObject] = []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.detach().cpu().tolist()
                confidence = boxes.conf.detach().cpu().tolist()
                classes = boxes.cls.detach().cpu().tolist()
                track_ids = boxes.id.detach().cpu().tolist()
                for box, score, class_id, track_id in zip(
                    xyxy, confidence, classes, track_ids, strict=True
                ):
                    if float(score) < self.detector_confidence:
                        continue
                    integer_class = int(class_id)
                    tracked.append(
                        TrackedObject(
                            track_id=int(track_id),
                            class_id=integer_class,
                            class_name=str(names[integer_class]),
                            confidence=float(score),
                            box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        )
                    )
            output.append(ObjectFrame(frame_index=frame_index, objects=tuple(tracked)))
        return tuple(output)
