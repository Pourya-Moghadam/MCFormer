"""Immutable JSONL manifests and dataset-specific manifest builders."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypedDict, overload

from mcformer.reproducibility import sha256_file

NTU_SAMPLE_PATTERN = re.compile(
    r"^(?P<sample_id>S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})"
    r"R(?P<repetition>\d{3})A(?P<action>\d{3}))"
)


class ManifestError(ValueError):
    """Raised when dataset metadata violates the manifest contract."""


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One video clip and all metadata required by protocol generation."""

    sample_id: str
    dataset: str
    rgb_path: str
    label_id: int
    label_name: str
    subject_id: str | None = None
    camera_or_view_id: str | None = None
    setup_id: str | None = None
    repetition_id: str | None = None
    num_frames: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    skeleton_path: str | None = None
    calibration_path: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SampleRecord:
        """Construct a record while rejecting unknown or missing fields."""

        known = {field.name for field in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ManifestError(f"Unknown sample fields: {sorted(unknown)}")
        try:
            return cls(**value)
        except TypeError as error:
            raise ManifestError(f"Invalid sample record: {error}") from error


class Manifest(Sequence[SampleRecord]):
    """Validated, deterministically ordered collection of sample records."""

    def __init__(self, records: Iterable[SampleRecord], *, expected_classes: int | None = None):
        self._records = tuple(sorted(records, key=lambda record: record.sample_id))
        self.validate(expected_classes=expected_classes)
        self._by_id = {record.sample_id: record for record in self._records}

    def __len__(self) -> int:
        return len(self._records)

    @overload
    def __getitem__(self, index: int) -> SampleRecord: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[SampleRecord, ...]: ...

    def __getitem__(self, index: int | slice) -> SampleRecord | tuple[SampleRecord, ...]:
        return self._records[index]

    def __iter__(self) -> Iterator[SampleRecord]:
        return iter(self._records)

    def by_id(self, sample_id: str) -> SampleRecord:
        """Return one record by stable sample ID."""

        try:
            return self._by_id[sample_id]
        except KeyError as error:
            raise ManifestError(f"Unknown sample ID: {sample_id}") from error

    def validate(self, *, expected_classes: int | None = None) -> None:
        """Check identity, dimensions, labels, and dataset consistency."""

        if not self._records:
            raise ManifestError("Manifest is empty")
        duplicate_ids = [
            sample_id
            for sample_id, count in Counter(record.sample_id for record in self._records).items()
            if count > 1
        ]
        if duplicate_ids:
            raise ManifestError(f"Duplicate sample IDs: {duplicate_ids[:10]}")
        duplicate_paths = [
            path
            for path, count in Counter(record.rgb_path for record in self._records).items()
            if count > 1
        ]
        if duplicate_paths:
            raise ManifestError(f"Duplicate RGB paths: {duplicate_paths[:10]}")
        datasets = {record.dataset for record in self._records}
        if len(datasets) != 1:
            raise ManifestError(f"Manifest mixes datasets: {sorted(datasets)}")
        for record in self._records:
            if not all(
                isinstance(value, str) and value
                for value in (
                    record.sample_id,
                    record.dataset,
                    record.rgb_path,
                    record.label_name,
                )
            ):
                raise ManifestError(f"Required text field is empty for {record.sample_id!r}")
            if (
                not isinstance(record.label_id, int)
                or isinstance(record.label_id, bool)
                or record.label_id < 0
            ):
                raise ManifestError(f"Invalid label for {record.sample_id}")
            for name in ("num_frames", "width", "height"):
                value = getattr(record, name)
                if value is not None and (
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                ):
                    raise ManifestError(f"{name} must be positive for {record.sample_id}")
            if record.fps is not None and (
                not isinstance(record.fps, int | float)
                or isinstance(record.fps, bool)
                or record.fps <= 0
            ):
                raise ManifestError(f"fps must be positive for {record.sample_id}")
            for name in (
                "subject_id",
                "camera_or_view_id",
                "setup_id",
                "repetition_id",
                "skeleton_path",
                "calibration_path",
            ):
                value = getattr(record, name)
                if value is not None and not isinstance(value, str):
                    raise ManifestError(f"{name} must be text or null for {record.sample_id}")
        labels = sorted({record.label_id for record in self._records})
        if labels != list(range(max(labels) + 1)):
            raise ManifestError(f"Labels must be contiguous from zero, got {labels}")
        names_by_label: dict[int, set[str]] = {}
        for record in self._records:
            names_by_label.setdefault(record.label_id, set()).add(record.label_name)
        inconsistent = {
            label: sorted(names) for label, names in names_by_label.items() if len(names) != 1
        }
        if inconsistent:
            raise ManifestError(f"Labels have inconsistent names: {inconsistent}")
        if expected_classes is not None:
            expected = list(range(expected_classes))
            if labels != expected:
                raise ManifestError(
                    f"Expected contiguous labels {expected[:3]}...{expected[-3:]}, got {labels}"
                )

    def write_jsonl(self, path: str | Path) -> str:
        """Write canonical JSONL and return its SHA-256 digest."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record in self._records:
                handle.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        return sha256_file(destination)

    @classmethod
    def read_jsonl(
        cls,
        path: str | Path,
        *,
        expected_classes: int | None = None,
    ) -> Manifest:
        """Read and validate a JSONL manifest."""

        records: list[SampleRecord] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    records.append(SampleRecord.from_mapping(value))
                except (json.JSONDecodeError, ManifestError) as error:
                    raise ManifestError(f"Invalid manifest line {line_number}: {error}") from error
        return cls(records, expected_classes=expected_classes)

    def validate_files(self, root: str | Path, *, require_skeleton: bool = False) -> None:
        """Validate referenced local files without decoding videos."""

        dataset_root = Path(root).expanduser().resolve()
        missing: list[str] = []
        for record in self._records:
            if not _resolve_under_root(dataset_root, record.rgb_path).is_file():
                missing.append(f"{record.sample_id}:rgb")
            if require_skeleton and (
                not record.skeleton_path
                or not _resolve_under_root(dataset_root, record.skeleton_path).is_file()
            ):
                missing.append(f"{record.sample_id}:skeleton")
        if missing:
            raise ManifestError(f"Missing referenced files ({len(missing)}): {missing[:10]}")


def _resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_label_map(path: str | Path) -> dict[int, str]:
    """Read a JSON object mapping integer-like IDs to non-empty label names."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ManifestError("Label map must be a JSON object")
    labels = {int(key): str(value).strip() for key, value in raw.items()}
    if any(not value for value in labels.values()) or sorted(labels) != list(range(len(labels))):
        raise ManifestError("Label map IDs must be contiguous from zero with non-empty names")
    return labels


class VideoMetadata(TypedDict):
    num_frames: int
    fps: float
    width: int
    height: int


def build_ntu_manifest(
    *,
    rgb_root: str | Path,
    dataset: str,
    label_map: Mapping[int, str],
    skeleton_root: str | Path | None = None,
    missing_sample_ids: Iterable[str] = (),
    inspect_video: bool = True,
) -> Manifest:
    """Build an NTU60/120 manifest from official filename-encoded metadata."""

    expected_classes = {"ntu_rgbd_60": 60, "ntu_rgbd_120": 120}
    if dataset not in expected_classes:
        raise ManifestError(f"Unsupported NTU dataset: {dataset}")
    if len(label_map) != expected_classes[dataset]:
        raise ManifestError(f"{dataset} requires {expected_classes[dataset]} labels")

    root = Path(rgb_root).expanduser().resolve()
    skeleton_base = Path(skeleton_root).expanduser().resolve() if skeleton_root else None
    missing = set(missing_sample_ids)
    records: list[SampleRecord] = []
    video_suffixes = {".avi", ".mp4", ".mkv", ".mov"}
    rgb_paths = (path for path in root.rglob("*") if path.suffix.lower() in video_suffixes)
    for rgb_path in sorted(rgb_paths):
        match = NTU_SAMPLE_PATTERN.match(rgb_path.stem)
        if match is None:
            continue
        sample_id = match.group("sample_id")
        if sample_id in missing:
            continue
        label_id = int(match.group("action")) - 1
        if label_id not in label_map:
            continue
        metadata: VideoMetadata | None = inspect_video_metadata(rgb_path) if inspect_video else None
        skeleton_path: str | None = None
        if skeleton_base is not None:
            candidate = skeleton_base / f"{sample_id}.skeleton"
            if candidate.is_file():
                skeleton_path = (
                    candidate.relative_to(root).as_posix()
                    if candidate.is_relative_to(root)
                    else str(candidate)
                )
        records.append(
            SampleRecord(
                sample_id=sample_id,
                dataset=dataset,
                rgb_path=rgb_path.relative_to(root).as_posix(),
                label_id=label_id,
                label_name=label_map[label_id],
                subject_id=match.group("subject"),
                camera_or_view_id=match.group("camera"),
                setup_id=match.group("setup"),
                repetition_id=match.group("repetition"),
                skeleton_path=skeleton_path,
                num_frames=metadata["num_frames"] if metadata else None,
                fps=metadata["fps"] if metadata else None,
                width=metadata["width"] if metadata else None,
                height=metadata["height"] if metadata else None,
            )
        )
    return Manifest(records, expected_classes=expected_classes[dataset])


def _optional_int(row: Mapping[str, str | None], name: str) -> int | None:
    value = row.get(name)
    return int(value) if value else None


def _optional_float(row: Mapping[str, str | None], name: str) -> float | None:
    value = row.get(name)
    return float(value) if value else None


def build_tabular_manifest(
    *,
    annotation_path: str | Path,
    data_root: str | Path,
    dataset: str,
    label_map: Mapping[int, str],
    delimiter: str = ",",
    inspect_video: bool = True,
) -> Manifest:
    """Build a manifest from a portable CSV/TSV annotation contract.

    Required columns are ``sample_id``, ``rgb_path``, and ``label_id``. Optional
    columns match :class:`SampleRecord`. This is the canonical adapter for restricted
    Toyota annotations.
    """

    records: list[SampleRecord] = []
    annotations = Path(annotation_path).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    with annotations.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"sample_id", "rgb_path", "label_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ManifestError(f"Annotations require columns: {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                label_id = int(row["label_id"])
                rgb_path = row["rgb_path"]
                source_path = Path(rgb_path).expanduser()
                source_path = source_path if source_path.is_absolute() else root / source_path
                inspected = inspect_video_metadata(source_path) if inspect_video else None
                label_name = row.get("label_name") or label_map[label_id]
                if label_name != label_map[label_id]:
                    raise ManifestError(f"CSV label name {label_name!r} disagrees with label map")
                records.append(
                    SampleRecord(
                        sample_id=row["sample_id"],
                        dataset=dataset,
                        rgb_path=rgb_path,
                        label_id=label_id,
                        label_name=label_name,
                        subject_id=row.get("subject_id") or None,
                        camera_or_view_id=row.get("camera_or_view_id") or None,
                        setup_id=row.get("setup_id") or None,
                        repetition_id=row.get("repetition_id") or None,
                        skeleton_path=row.get("skeleton_path") or None,
                        calibration_path=row.get("calibration_path") or None,
                        num_frames=(
                            inspected["num_frames"]
                            if inspected
                            else _optional_int(row, "num_frames")
                        ),
                        fps=(inspected["fps"] if inspected else _optional_float(row, "fps")),
                        width=(inspected["width"] if inspected else _optional_int(row, "width")),
                        height=(inspected["height"] if inspected else _optional_int(row, "height")),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ManifestError(f"Invalid annotation line {line_number}: {error}") from error
    return Manifest(records, expected_classes=len(label_map))


def inspect_video_metadata(path: str | Path) -> VideoMetadata:
    """Read frame count, rate, and dimensions using OpenCV."""

    try:
        import cv2
    except ImportError as error:
        raise ManifestError("Video inspection requires opencv-python-headless") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ManifestError(f"Unreadable video: {path}")
    try:
        metadata: VideoMetadata = {
            "num_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()
    if (
        metadata["num_frames"] <= 0
        or metadata["fps"] <= 0
        or metadata["width"] <= 0
        or metadata["height"] <= 0
    ):
        raise ManifestError(f"Invalid video metadata for {path}: {metadata}")
    return metadata
