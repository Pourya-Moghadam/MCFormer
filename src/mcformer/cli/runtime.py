"""Shared validated construction for training and evaluation commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mcformer.auxiliary.pipeline import TargetSettings
from mcformer.config import ResolvedConfig
from mcformer.data.dataset import VideoClipDataset
from mcformer.data.manifest import Manifest
from mcformer.data.protocols import ProtocolSplit


class RuntimeConfigurationError(ValueError):
    """Raised when a resolved experiment cannot drive an executable run."""


def section(config: ResolvedConfig, name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise RuntimeConfigurationError(f"Configuration section {name!r} must be a mapping")
    return value


def selected_seed(config: ResolvedConfig, seed: int | None) -> int:
    values = section(config, "reproducibility").get("seeds")
    assert isinstance(values, list)
    selected = int(values[0]) if seed is None else seed
    if selected not in values:
        raise RuntimeConfigurationError(
            f"Seed {selected} is not one of configured reproducibility.seeds={values}"
        )
    return selected


def load_data_contract(
    config: ResolvedConfig,
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
) -> tuple[Manifest, ProtocolSplit]:
    model = section(config, "model")
    classes = model.get("num_classes")
    if not isinstance(classes, int):
        raise RuntimeConfigurationError("model.num_classes must be an integer")
    manifest = Manifest.read_jsonl(manifest_path, expected_classes=classes)
    split = ProtocolSplit.read_json(protocol_path, manifest)
    configured_dataset = section(config, "data").get("dataset")
    configured_protocol = section(config, "data").get("protocol")
    if (
        manifest[0].dataset != configured_dataset
        or split.protocol.casefold() != str(configured_protocol).casefold()
    ):
        raise RuntimeConfigurationError("Manifest/split identity disagrees with the configuration")
    return manifest, split


def target_settings(config: ResolvedConfig) -> TargetSettings:
    values = section(config, "auxiliary")
    settings = TargetSettings(
        pose_confidence=float(values["pose_confidence"]),
        pose_max_gap=int(values["pose_max_gap"]),
        gaussian_sigma_frames=float(values["gaussian_sigma_frames"]),
        object_max_gap=int(values["object_max_gap"]),
        minimum_track_coverage=float(values["minimum_track_coverage"]),
        minimum_track_mean_confidence=float(values["minimum_track_mean_confidence"]),
        distance_threshold=float(values["distance_threshold"]),
        epsilon=float(values["epsilon"]),
    )
    settings.validate()
    return settings


def make_dataset(
    config: ResolvedConfig,
    manifest: Manifest,
    sample_ids: Sequence[str],
    *,
    root: str | Path,
    training: bool,
    seed: int,
    target_builder: Any = None,
) -> VideoClipDataset:
    data = section(config, "data")
    return VideoClipDataset(
        manifest=manifest,
        sample_ids=sample_ids,
        root=root,
        training=training,
        seed=seed,
        clip_length=int(data["num_frames"]),
        stride=int(data["temporal_stride"]),
        input_size=int(data["input_size"]),
        resize_short_side=int(data["resize_short_side"]),
        target_builder=target_builder,
    )
