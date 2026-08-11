"""Content-addressed corrupted observation caches for robustness experiment E12."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace

from mcformer.auxiliary.cache import ObservationBundle, ObservationCache, configuration_digest
from mcformer.auxiliary.corruptions import (
    add_wrist_noise,
    drop_object_detections,
    nearest_alternative_track,
    occlude_track,
)
from mcformer.auxiliary.trajectories import dominant_hand_trajectory, primary_object_trajectory
from mcformer.auxiliary.types import ObjectFrame


class CorruptionCacheError(ValueError):
    """Raised when a corruption specification is invalid or cannot be realized."""


@dataclass(frozen=True, slots=True)
class CorruptionSpec:
    name: str
    value: float
    seed: int = 2026

    def validate(self) -> None:
        if self.name == "wrist_noise" and self.value in {0.02, 0.05}:
            return
        if self.name == "missed_detections" and self.value in {0.10, 0.20}:
            return
        if self.name == "object_occlusion" and self.value == 4:
            return
        if self.name == "track_swap" and self.value in {0.10, 0.20}:
            return
        raise CorruptionCacheError(f"Unsupported E12 corruption setting: {self.name}={self.value}")


def _rng(seed: int, sample_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _selected_and_alternative(bundle: ObservationBundle) -> tuple[int | None, int | None]:
    hand = dominant_hand_trajectory(
        bundle.poses,
        width=bundle.width,
        height=bundle.height,
    )
    selected = primary_object_trajectory(
        bundle.objects,
        hand,
        width=bundle.width,
        height=bundle.height,
    )
    if selected is None:
        return None, None
    alternative = nearest_alternative_track(
        bundle.objects,
        hand,
        selected_track_id=selected.track_id,
        width=bundle.width,
        height=bundle.height,
    )
    return selected.track_id, alternative


def _only_track(frames: tuple[ObjectFrame, ...], track_id: int) -> tuple[ObjectFrame, ...]:
    return tuple(
        ObjectFrame(
            frame_index=frame.frame_index,
            objects=tuple(item for item in frame.objects if item.track_id == track_id),
        )
        for frame in frames
    )


def build_corrupted_cache(
    source: ObservationCache,
    destination: str,
    *,
    sample_ids: tuple[str, ...],
    spec: CorruptionSpec,
) -> tuple[ObservationCache, dict[str, object]]:
    """Materialize one fixed corruption realization and return its audit report."""

    spec.validate()
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise CorruptionCacheError("Corruption sample IDs must be non-empty and unique")
    selected_tracks: dict[str, tuple[int | None, int | None]] = {}
    if spec.name in {"object_occlusion", "track_swap"}:
        for sample_id in sample_ids:
            selected_tracks[sample_id] = _selected_and_alternative(source.read(sample_id))
    swapped_ids: set[str] = set()
    if spec.name == "track_swap":
        eligible = [
            sample_id for sample_id in sample_ids if selected_tracks[sample_id][1] is not None
        ]
        ranked = sorted(
            eligible,
            key=lambda sample_id: hashlib.sha256(f"{spec.seed}:{sample_id}".encode()).digest(),
        )
        count = round(len(ranked) * spec.value)
        swapped_ids = set(ranked[:count])
    identity = {
        "source_cache_key": source.cache_key,
        "source_content_sha256": source.content_digest(sample_ids),
        "sample_ids": sorted(sample_ids),
        "corruption": {"name": spec.name, "value": spec.value, "seed": spec.seed},
        "swapped_sample_ids": sorted(swapped_ids),
    }
    cache = ObservationCache(destination, cache_key=configuration_digest(identity))
    cache.initialize(identity)
    changed = 0
    no_selected_track = 0
    no_alternative_track = 0
    realization: dict[str, object] = {}
    for sample_id in sorted(sample_ids):
        bundle = source.read(sample_id)
        if bundle.mode != "native_frames":
            raise CorruptionCacheError("E12 corruptions require a native_frames source cache")
        generator = _rng(spec.seed, sample_id)
        corrupted = bundle
        if spec.name == "wrist_noise":
            corrupted = replace(
                bundle,
                poses=add_wrist_noise(
                    bundle.poses,
                    sigma_diagonal=spec.value,
                    width=bundle.width,
                    height=bundle.height,
                    rng=generator,
                ),
            )
            changed += 1
            realization[sample_id] = {"pose_frame_indices": list(bundle.frame_indices)}
        elif spec.name == "missed_detections":
            objects = drop_object_detections(bundle.objects, probability=spec.value, rng=generator)
            corrupted = replace(bundle, objects=objects)
            changed += objects != bundle.objects
            realization[sample_id] = {
                "removed": [
                    {"frame_index": before.frame_index, "track_id": item.track_id}
                    for before, after in zip(bundle.objects, objects, strict=True)
                    for item in before.objects
                    if item not in after.objects
                ]
            }
        elif spec.name == "object_occlusion":
            selected, _ = selected_tracks[sample_id]
            if selected is None or len(bundle.objects) < int(spec.value):
                no_selected_track += 1
            else:
                corrupted = replace(
                    bundle,
                    objects=occlude_track(
                        bundle.objects,
                        track_id=selected,
                        length=int(spec.value),
                        rng=generator,
                    ),
                )
                changed += 1
                realization[sample_id] = {
                    "selected_track_id": selected,
                    "removed_frame_indices": [
                        before.frame_index
                        for before, after in zip(bundle.objects, corrupted.objects, strict=True)
                        if before != after
                    ],
                }
        elif spec.name == "track_swap":
            _, alternative = selected_tracks[sample_id]
            if alternative is None:
                no_alternative_track += 1
            elif sample_id in swapped_ids:
                corrupted = replace(bundle, objects=_only_track(bundle.objects, alternative))
                changed += 1
                realization[sample_id] = {
                    "selected_track_id": selected_tracks[sample_id][0],
                    "replacement_track_id": alternative,
                }
        corrupted.validate()
        cache.write(corrupted)
    report: dict[str, object] = {
        "schema_version": 1,
        **identity,
        "cache_key": cache.cache_key,
        "samples": len(sample_ids),
        "changed_samples": changed,
        "no_selected_track_samples": no_selected_track,
        "no_alternative_track_samples": no_alternative_track,
        "realization": realization,
    }
    return cache, report
