"""Shared feature contracts for video backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn


class ModelInputError(ValueError):
    """Raised when a video tensor violates the public model contract."""


@dataclass(frozen=True)
class BackboneOutput:
    """Final temporal tokens and the representation used for classification."""

    temporal_tokens: Tensor
    pooled: Tensor


class VideoBackbone(nn.Module, ABC):
    """Backbone interface accepting videos in ``B,T,C,H,W`` layout."""

    output_dim: int
    temporal_dim: int

    @abstractmethod
    def forward(self, video: Tensor) -> BackboneOutput:
        """Extract temporal and pooled video representations."""

    @staticmethod
    def validate_video(video: Tensor) -> None:
        if video.ndim != 5:
            raise ModelInputError(f"Expected B,T,C,H,W video, got shape {tuple(video.shape)}")
        if video.shape[2] != 3:
            raise ModelInputError(f"Expected three RGB channels, got {video.shape[2]}")
        if min(video.shape) <= 0:
            raise ModelInputError("Video dimensions must be positive")
