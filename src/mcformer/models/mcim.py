"""Training-only Motion Coupling Induction Module."""

from __future__ import annotations

from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class MCIM(nn.Module):
    """Predict one coupling scalar per sampled frame from temporal RGB tokens."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int | None = None,
        output_frames: int = 32,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_frames <= 0:
            raise ValueError("input_dim and output_frames must be positive")
        effective_hidden = hidden_dim if hidden_dim is not None else input_dim // 2
        if effective_hidden <= 0:
            raise ValueError("hidden_dim must be positive")
        self.input_dim = input_dim
        self.hidden_dim = effective_hidden
        self.output_frames = output_frames
        self.layers = nn.Sequential(
            nn.Linear(input_dim, effective_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(effective_hidden, 1, bias=True),
        )

    def forward(self, temporal_tokens: Tensor) -> Tensor:
        if temporal_tokens.ndim != 3 or temporal_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected B,T,{self.input_dim} temporal tokens, got "
                f"{tuple(temporal_tokens.shape)}"
            )
        prediction = self.layers(temporal_tokens).squeeze(-1)
        if prediction.shape[1] != self.output_frames:
            prediction = functional.interpolate(
                prediction.unsqueeze(1),
                size=self.output_frames,
                mode="linear",
                align_corners=True,
            ).squeeze(1)
        return cast(Tensor, prediction)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
