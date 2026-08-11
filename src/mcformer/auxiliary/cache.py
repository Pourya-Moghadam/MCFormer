"""Immutable, checksummed storage for raw auxiliary observations."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from mcformer.auxiliary.types import (
    ObjectFrame,
    PersonPose,
    PoseFrame,
    TrackedObject,
    Wrist,
)


class CacheError(RuntimeError):
    """Raised when an observation cache is incompatible, corrupt, or unsafe."""


@dataclass(frozen=True, slots=True)
class ObservationBundle:
    """Raw pose and tracked-object observations for one video or sampled view."""

    sample_id: str
    width: int
    height: int
    frame_indices: tuple[int, ...]
    poses: tuple[PoseFrame, ...]
    objects: tuple[ObjectFrame, ...]
    pose_backend: str
    object_backend: str
    mode: str

    def validate(self) -> None:
        length = len(self.frame_indices)
        if (
            not self.sample_id
            or self.width <= 0
            or self.height <= 0
            or length == 0
            or not self.pose_backend
            or not self.object_backend
        ):
            raise CacheError("Observation bundle dimensions and frame list must be positive")
        if self.mode not in {"native_frames", "paper_cost_mode"}:
            raise CacheError(f"Unknown observation cache mode: {self.mode}")
        if any(index < 0 for index in self.frame_indices) or any(
            right < left for left, right in pairwise(self.frame_indices)
        ):
            raise CacheError("Frame indices must be non-negative and nondecreasing")
        if len(self.poses) != length or len(self.objects) != length:
            raise CacheError("Pose/object observations must align with frame_indices")
        if tuple(frame.frame_index for frame in self.poses) != self.frame_indices:
            raise CacheError("Pose frame indices do not match the bundle")
        if tuple(frame.frame_index for frame in self.objects) != self.frame_indices:
            raise CacheError("Object frame indices do not match the bundle")
        for pose_frame in self.poses:
            actor_ids = [person.actor_id for person in pose_frame.people]
            if len(actor_ids) != len(set(actor_ids)) or any(not value for value in actor_ids):
                raise CacheError(f"Invalid or duplicate actor ID at frame {pose_frame.frame_index}")
            for person in pose_frame.people:
                for wrist in (person.left_wrist, person.right_wrist):
                    if wrist is None:
                        continue
                    values = (*wrist.point, wrist.confidence)
                    if not all(math.isfinite(value) for value in values):
                        raise CacheError(f"Non-finite wrist at frame {pose_frame.frame_index}")
                    if not 0 <= wrist.confidence <= 1:
                        raise CacheError(
                            f"Invalid wrist confidence at frame {pose_frame.frame_index}"
                        )
        for object_frame in self.objects:
            track_ids = [item.track_id for item in object_frame.objects]
            if len(track_ids) != len(set(track_ids)):
                raise CacheError(f"Duplicate track ID at frame {object_frame.frame_index}")
            for item in object_frame.objects:
                if (
                    item.track_id < 0
                    or item.class_id < 0
                    or not item.class_name
                    or not 0 <= item.confidence <= 1
                    or not all(math.isfinite(value) for value in item.box)
                    or item.box[2] <= item.box[0]
                    or item.box[3] <= item.box[1]
                ):
                    raise CacheError(
                        f"Invalid object observation at frame {object_frame.frame_index}"
                    )


def configuration_digest(value: Any) -> str:
    """Hash JSON-compatible settings using a canonical representation."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _bundle_to_mapping(bundle: ObservationBundle, cache_key: str) -> dict[str, Any]:
    return {"schema_version": 1, "cache_key": cache_key, "bundle": asdict(bundle)}


def _wrist(value: dict[str, Any] | None) -> Wrist | None:
    if value is None:
        return None
    point = value["point"]
    return Wrist(point=(float(point[0]), float(point[1])), confidence=float(value["confidence"]))


def _bundle_from_mapping(raw: Any, expected_key: str) -> ObservationBundle:
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 1
        or raw.get("cache_key") != expected_key
    ):
        raise CacheError("Observation cache schema or key mismatch")
    value = raw["bundle"]
    poses = tuple(
        PoseFrame(
            frame_index=int(frame["frame_index"]),
            people=tuple(
                PersonPose(
                    actor_id=str(person["actor_id"]),
                    left_wrist=_wrist(person["left_wrist"]),
                    right_wrist=_wrist(person["right_wrist"]),
                )
                for person in frame["people"]
            ),
        )
        for frame in value["poses"]
    )
    objects = tuple(
        ObjectFrame(
            frame_index=int(frame["frame_index"]),
            objects=tuple(
                TrackedObject(
                    track_id=int(item["track_id"]),
                    class_id=int(item["class_id"]),
                    class_name=str(item["class_name"]),
                    confidence=float(item["confidence"]),
                    box=(
                        float(item["box"][0]),
                        float(item["box"][1]),
                        float(item["box"][2]),
                        float(item["box"][3]),
                    ),
                )
                for item in frame["objects"]
            ),
        )
        for frame in value["objects"]
    )
    bundle = ObservationBundle(
        sample_id=str(value["sample_id"]),
        width=int(value["width"]),
        height=int(value["height"]),
        frame_indices=tuple(int(index) for index in value["frame_indices"]),
        poses=poses,
        objects=objects,
        pose_backend=str(value["pose_backend"]),
        object_backend=str(value["object_backend"]),
        mode=str(value["mode"]),
    )
    bundle.validate()
    return bundle


class ObservationCache:
    """Directory of deterministic gzip JSON bundles tied to one config digest."""

    def __init__(self, root: str | Path, *, cache_key: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.cache_key = cache_key
        self.samples = self.root / "samples"
        self.targets = self.root / "targets"

    @classmethod
    def open(cls, root: str | Path) -> ObservationCache:
        """Open an initialized cache and recover its immutable identity."""

        cache_root = Path(root).expanduser().resolve()
        try:
            value = json.loads((cache_root / "cache.json").read_text(encoding="utf-8"))
            cache_key = value["cache_key"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise CacheError(f"Cannot open observation cache {cache_root}: {error}") from error
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or not isinstance(cache_key, str)
            or len(cache_key) != 64
        ):
            raise CacheError("Observation cache metadata is invalid")
        cache = cls(cache_root, cache_key=cache_key)
        if not cache.samples.is_dir() or not cache.targets.is_dir():
            raise CacheError("Observation cache sample/target directories are missing")
        return cache

    def initialize(self, metadata: dict[str, Any]) -> None:
        """Create cache metadata or validate an already initialized cache."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.samples.mkdir(parents=True, exist_ok=True)
        self.targets.mkdir(parents=True, exist_ok=True)
        path = self.root / "cache.json"
        payload = {"schema_version": 1, "cache_key": self.cache_key, "metadata": metadata}
        canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise CacheError(f"Unreadable cache metadata: {error}") from error
            if existing != payload:
                raise CacheError("Existing cache metadata is incompatible")
        else:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.root, delete=False
            ) as handle:
                handle.write(canonical)
                temporary = Path(handle.name)
            temporary.replace(path)

    def _path(self, sample_id: str) -> Path:
        safe_name = hashlib.sha256(sample_id.encode()).hexdigest()
        return self.samples / f"{safe_name}.json.gz"

    def contains(self, sample_id: str) -> bool:
        return self._path(sample_id).is_file()

    def contains_target(self, sample_id: str) -> bool:
        return self._target_path(sample_id).is_file()

    def content_digest(self, sample_ids: Iterable[str]) -> str:
        """Hash cache identity and exact compressed sample contents in stable ID order."""

        digest = hashlib.sha256(self.cache_key.encode())
        for sample_id in sorted(str(value) for value in sample_ids):
            path = self._path(sample_id)
            if not path.is_file():
                raise CacheError(f"Cached sample does not exist: {sample_id}")
            digest.update(sample_id.encode())
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()

    def write(self, bundle: ObservationBundle, *, overwrite: bool = False) -> Path:
        """Atomically write one bundle with deterministic gzip metadata."""

        bundle.validate()
        destination = self._path(bundle.sample_id)
        if destination.exists() and not overwrite:
            raise CacheError(f"Refusing to overwrite cached sample: {bundle.sample_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            _bundle_to_mapping(bundle, self.cache_key),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(payload)
        temporary.replace(destination)
        return destination

    def read(self, sample_id: str) -> ObservationBundle:
        path = self._path(sample_id)
        if not path.is_file():
            raise CacheError(f"Cached sample does not exist: {sample_id}")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                raw = json.load(handle)
            bundle = _bundle_from_mapping(raw, self.cache_key)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise CacheError(f"Corrupt cached sample {sample_id}: {error}") from error
        if bundle.sample_id != sample_id:
            raise CacheError("Hashed cache filename contains the wrong sample")
        return bundle

    def _target_path(self, sample_id: str) -> Path:
        safe_name = hashlib.sha256(sample_id.encode()).hexdigest()
        return self.targets / f"{safe_name}.json.gz"

    @staticmethod
    def _write_gzip_json(destination: Path, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(encoded)
        temporary.replace(destination)

    def write_target(
        self,
        sample_id: str,
        payload: dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist a canonical center-view target for auditing and cost-mode use."""

        destination = self._target_path(sample_id)
        if destination.exists() and not overwrite:
            raise CacheError(f"Refusing to overwrite cached target: {sample_id}")
        wrapped = {
            "schema_version": 1,
            "cache_key": self.cache_key,
            "sample_id": sample_id,
            "target": payload,
        }
        self._write_gzip_json(destination, wrapped)
        return destination

    def read_target(self, sample_id: str) -> dict[str, Any]:
        """Read and verify one canonical cached target."""

        try:
            with gzip.open(self._target_path(sample_id), "rt", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, TypeError, ValueError) as error:
            raise CacheError(f"Corrupt cached target {sample_id}: {error}") from error
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != 1
            or raw.get("cache_key") != self.cache_key
            or raw.get("sample_id") != sample_id
            or not isinstance(raw.get("target"), dict)
        ):
            raise CacheError("Cached target schema, key, or sample ID mismatch")
        return dict(raw["target"])
