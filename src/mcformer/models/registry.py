"""Validated construction of release model variants."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import nn

from mcformer.config import ResolvedConfig
from mcformer.models.checkpoints import CheckpointReport, load_local_checkpoint
from mcformer.models.classifier import (
    AuxiliaryFormer,
    AuxiliaryHeadDefinition,
    MCFormer,
    VideoClassifier,
)
from mcformer.models.common import VideoBackbone
from mcformer.models.timesformer import TimeSformerBackbone
from mcformer.models.torchvision_backbones import MViTv2SmallBackbone, VideoSwinTinyBackbone


class ModelConfigurationError(ValueError):
    """Raised when a model configuration violates the frozen architecture contract."""


def _mapping(config: ResolvedConfig | Mapping[str, Any]) -> Mapping[str, Any]:
    return config.values if isinstance(config, ResolvedConfig) else config


def _model_settings(config: ResolvedConfig | Mapping[str, Any]) -> Mapping[str, Any]:
    values = _mapping(config)
    model = values.get("model")
    if not isinstance(model, Mapping):
        raise ModelConfigurationError("Configuration requires a model mapping")
    return model


def build_backbone(config: ResolvedConfig | Mapping[str, Any]) -> VideoBackbone:
    """Build the exact configured backbone with no network or checkpoint side effect."""

    settings = _model_settings(config)
    name = settings.get("name")
    if name in {"video_swin_t", "mcformer_video_swin_t"}:
        expected = {
            "patch_size": [2, 4, 4],
            "window_size": [8, 7, 7],
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "stochastic_depth": 0.2,
        }
        differences = {
            key: (settings.get(key), value)
            for key, value in expected.items()
            if settings.get(key) != value
        }
        if differences:
            raise ModelConfigurationError(f"Video Swin-T settings disagree: {differences}")
        mcim = settings.get("mcim", {})
        insertion_stage = int(mcim.get("insertion_stage", 4)) if isinstance(mcim, Mapping) else 4
        return VideoSwinTinyBackbone(insertion_stage=insertion_stage)
    if name == "timesformer_base_divided_space_time":
        return TimeSformerBackbone(
            image_size=224,
            patch_size=int(settings.get("patch_size", 16)),
            num_frames=int(_mapping(config).get("data", {}).get("num_frames", 32)),
            embed_dim=int(settings.get("embed_dim", 768)),
            depth=int(settings.get("depth", 12)),
            num_heads=int(settings.get("num_heads", 12)),
            dropout=float(settings.get("dropout", 0.0)),
            attention_dropout=float(settings.get("attention_dropout", 0.0)),
            stochastic_depth=float(settings.get("stochastic_depth", 0.1)),
        )
    if name == "mvitv2_small":
        data = _mapping(config).get("data")
        if not isinstance(data, Mapping):
            raise ModelConfigurationError("Configuration requires a data mapping")
        return MViTv2SmallBackbone(
            num_frames=int(data.get("num_frames", 32)),
            image_size=int(data.get("input_size", 224)),
        )
    raise ModelConfigurationError(f"Unsupported model name: {name!r}")


def build_model(
    config: ResolvedConfig | Mapping[str, Any],
    *,
    initialization_checkpoint: str | Path | None = None,
    initialization_sha256: str | None = None,
    allow_random_initialization: bool = False,
) -> tuple[nn.Module, CheckpointReport | None]:
    """Build a classifier/MC-Former and load verified local initialization weights."""

    settings = _model_settings(config)
    num_classes = settings.get("num_classes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 1:
        raise ModelConfigurationError("model.num_classes must be an integer greater than one")
    backbone = build_backbone(config)
    report: CheckpointReport | None = None
    if initialization_checkpoint is not None:
        if initialization_sha256 is None:
            raise ModelConfigurationError("Initialization checkpoint requires its SHA-256")
        report = load_local_checkpoint(
            backbone,
            initialization_checkpoint,
            expected_sha256=initialization_sha256,
            strict=True,
        )
    elif not allow_random_initialization:
        raise ModelConfigurationError(
            "Paper models require a local initialization checkpoint and SHA-256"
        )
    rgb_model = VideoClassifier(backbone, num_classes)
    raw_heads = settings.get("auxiliary_heads")
    if raw_heads is not None:
        if not isinstance(raw_heads, list) or not raw_heads:
            raise ModelConfigurationError("model.auxiliary_heads must be a non-empty list")
        definitions: list[AuxiliaryHeadDefinition] = []
        allowed_targets = {"temporal_gated", "temporal_ungated", "spatial", "hallucination"}
        output_frames = int(_mapping(config).get("data", {}).get("num_frames", 32))
        for raw in raw_heads:
            if not isinstance(raw, Mapping):
                raise ModelConfigurationError("Each auxiliary head must be a mapping")
            target = str(raw.get("target"))
            kind = str(raw.get("kind"))
            if target not in allowed_targets:
                raise ModelConfigurationError(f"Unsupported auxiliary target: {target!r}")
            default_dimension = 256 if kind == "vector" else output_frames
            definitions.append(
                AuxiliaryHeadDefinition(
                    name=str(raw.get("name")),
                    target=target,
                    kind=kind,
                    output_dim=int(raw.get("output_dim", default_dimension)),
                    weight=float(raw.get("weight", 1.0)),
                )
            )
        return AuxiliaryFormer(rgb_model, tuple(definitions), output_frames=output_frames), report
    mcim = settings.get("mcim")
    enabled = isinstance(mcim, Mapping) and mcim.get("enabled") is True
    if enabled:
        output_frames = int(_mapping(config).get("data", {}).get("num_frames", 32))
        return MCFormer(rgb_model, output_frames=output_frames), report
    return rgb_model, report
