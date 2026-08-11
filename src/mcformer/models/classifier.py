"""RGB classifiers, MC-Former training graph, and RGB-only export."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from mcformer.models.common import BackboneOutput, VideoBackbone
from mcformer.models.mcim import MCIM


@dataclass(frozen=True)
class ClassificationOutput:
    logits: Tensor
    temporal_tokens: Tensor
    pooled: Tensor


@dataclass(frozen=True)
class MCFormerOutput:
    logits: Tensor
    coupling: Tensor
    temporal_tokens: Tensor
    pooled: Tensor


@dataclass(frozen=True)
class AuxiliaryHeadDefinition:
    """One named auxiliary regression head and its matching batch target."""

    name: str
    target: str
    kind: str
    output_dim: int
    weight: float


@dataclass(frozen=True)
class AuxiliaryFormerOutput:
    logits: Tensor
    auxiliary_predictions: dict[str, Tensor]
    auxiliary_targets: dict[str, str]
    auxiliary_weights: dict[str, float]
    temporal_tokens: Tensor
    pooled: Tensor


class VideoClassifier(nn.Module):
    """Linear classification head over a video backbone's pooled representation."""

    def __init__(self, backbone: VideoBackbone, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.backbone = backbone
        self.num_classes = num_classes
        self.classifier = nn.Linear(backbone.output_dim, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, video: Tensor) -> ClassificationOutput:
        features: BackboneOutput = self.backbone(video)
        return ClassificationOutput(
            logits=self.classifier(features.pooled),
            temporal_tokens=features.temporal_tokens,
            pooled=features.pooled,
        )


class MCFormer(nn.Module):
    """Training graph whose MCIM branch never contributes to classification logits."""

    def __init__(self, rgb_model: VideoClassifier, *, output_frames: int = 32) -> None:
        super().__init__()
        self.rgb_model = rgb_model
        self.mcim = MCIM(rgb_model.backbone.temporal_dim, output_frames=output_frames)

    def forward(self, video: Tensor) -> MCFormerOutput:
        rgb = self.rgb_model(video)
        return MCFormerOutput(
            logits=rgb.logits,
            coupling=self.mcim(rgb.temporal_tokens),
            temporal_tokens=rgb.temporal_tokens,
            pooled=rgb.pooled,
        )

    def export_rgb_model(self) -> VideoClassifier:
        """Return a detached module tree containing no MCIM parameters or dependency."""

        exported = copy.deepcopy(self.rgb_model)
        exported.train(self.training)
        return exported


class AuxiliaryFormer(nn.Module):
    """Configurable training-only auxiliary heads for E06--E10 ablations."""

    def __init__(
        self,
        rgb_model: VideoClassifier,
        definitions: tuple[AuxiliaryHeadDefinition, ...],
        *,
        output_frames: int = 32,
    ) -> None:
        super().__init__()
        if not definitions or len({definition.name for definition in definitions}) != len(
            definitions
        ):
            raise ValueError("Auxiliary head definitions must be non-empty with unique names")
        self.rgb_model = rgb_model
        self.definitions = definitions
        heads: dict[str, nn.Module] = {}
        for definition in definitions:
            if definition.weight < 0 or definition.output_dim <= 0:
                raise ValueError("Auxiliary head weight/dimension is invalid")
            if definition.kind == "temporal":
                if definition.output_dim != output_frames:
                    raise ValueError("Temporal head output_dim must equal sampled frame count")
                heads[definition.name] = MCIM(
                    rgb_model.backbone.temporal_dim, output_frames=output_frames
                )
            elif definition.kind == "vector":
                heads[definition.name] = nn.Sequential(
                    nn.Linear(rgb_model.backbone.output_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, definition.output_dim),
                )
            else:
                raise ValueError("Auxiliary head kind must be temporal or vector")
        self.heads = nn.ModuleDict(heads)

    def forward(self, video: Tensor) -> AuxiliaryFormerOutput:
        rgb = self.rgb_model(video)
        predictions = {
            definition.name: (
                self.heads[definition.name](rgb.temporal_tokens)
                if definition.kind == "temporal"
                else self.heads[definition.name](rgb.pooled)
            )
            for definition in self.definitions
        }
        return AuxiliaryFormerOutput(
            logits=rgb.logits,
            auxiliary_predictions=predictions,
            auxiliary_targets={
                definition.name: definition.target for definition in self.definitions
            },
            auxiliary_weights={
                definition.name: definition.weight for definition in self.definitions
            },
            temporal_tokens=rgb.temporal_tokens,
            pooled=rgb.pooled,
        )

    def export_rgb_model(self) -> VideoClassifier:
        exported = copy.deepcopy(self.rgb_model)
        exported.train(self.training)
        return exported


def export_rgb_checkpoint(
    model: MCFormer | AuxiliaryFormer,
    path: str | Path,
    *,
    config_sha256: str | None = None,
    seed: int | None = None,
) -> None:
    """Serialize only the deployed RGB graph's state dictionary."""

    exported = model.export_rgb_model()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "rgb_only",
            "state_dict": exported.state_dict(),
            "config_sha256": config_sha256,
            "seed": seed,
        },
        destination,
    )
