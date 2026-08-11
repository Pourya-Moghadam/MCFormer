"""Self-contained divided space-time TimeSformer backbone."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mcformer.models.common import BackboneOutput, ModelInputError, VideoBackbone


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("Drop-path probability must lie in [0,1)")
        self.probability = probability

    def forward(self, value: Tensor) -> Tensor:
        if not self.training or self.probability == 0:
            return value
        keep = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        random = value.new_empty(shape).bernoulli_(keep)
        return value * random / keep


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return cast(Tensor, self.layers(value))


class DividedSpaceTimeBlock(nn.Module):
    """Temporal attention per patch followed by spatial attention per frame."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("Embedding dimension must be divisible by number of heads")
        self.temporal_norm = nn.LayerNorm(dim, eps=1e-6)
        self.temporal_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.temporal_projection = nn.Linear(dim, dim)
        self.spatial_norm = nn.LayerNorm(dim, eps=1e-6)
        self.spatial_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, class_token: Tensor, patches: Tensor) -> tuple[Tensor, Tensor]:
        batch, frames, spatial_tokens, dim = patches.shape
        temporal_input = patches.permute(0, 2, 1, 3).reshape(batch * spatial_tokens, frames, dim)
        normalized_temporal = self.temporal_norm(temporal_input)
        temporal_output = self.temporal_attention(
            normalized_temporal,
            normalized_temporal,
            normalized_temporal,
            need_weights=False,
        )[0]
        temporal_output = self.temporal_projection(temporal_output)
        patches = patches + self.drop_path(
            temporal_output.reshape(batch, spatial_tokens, frames, dim).permute(0, 2, 1, 3)
        )

        repeated_class = class_token[:, None].expand(-1, frames, -1, -1)
        spatial_input = torch.cat((repeated_class, patches), dim=2).reshape(
            batch * frames, spatial_tokens + 1, dim
        )
        normalized_spatial = self.spatial_norm(spatial_input)
        spatial_output = self.spatial_attention(
            normalized_spatial,
            normalized_spatial,
            normalized_spatial,
            need_weights=False,
        )[0]
        spatial_input = spatial_input + self.drop_path(spatial_output)
        spatial_input = spatial_input + self.drop_path(self.mlp(self.mlp_norm(spatial_input)))
        spatial_input = spatial_input.reshape(batch, frames, spatial_tokens + 1, dim)
        return spatial_input[:, :, :1].mean(dim=1), spatial_input[:, :, 1:]


class TimeSformerBackbone(VideoBackbone):
    """TimeSformer with divided temporal and spatial attention."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 32,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size or min(image_size, patch_size, num_frames, depth) <= 0:
            raise ValueError("Invalid TimeSformer spatial, temporal, or depth setting")
        if not 0 <= stochastic_depth < 1:
            raise ValueError("stochastic_depth must lie in [0,1)")
        self.output_dim = embed_dim
        self.temporal_dim = embed_dim
        self.patch_size = patch_size
        self.base_grid_size = image_size // patch_size
        self.base_frames = num_frames
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
        )
        spatial_tokens = self.base_grid_size**2
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_position = nn.Parameter(torch.zeros(1, spatial_tokens, embed_dim))
        self.class_position = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.temporal_position = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        drop_rates = torch.linspace(0, stochastic_depth, depth).tolist()
        self.blocks = nn.ModuleList(
            DividedSpaceTimeBlock(
                embed_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_rates[index],
            )
            for index in range(depth)
        )
        self.position_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.class_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _spatial_positions(self, height: int, width: int) -> Tensor:
        if height == self.base_grid_size and width == self.base_grid_size:
            return self.spatial_position
        position = self.spatial_position.reshape(
            1, self.base_grid_size, self.base_grid_size, self.output_dim
        ).permute(0, 3, 1, 2)
        position = functional.interpolate(
            position, size=(height, width), mode="bicubic", align_corners=False
        )
        return position.permute(0, 2, 3, 1).reshape(1, height * width, self.output_dim)

    def _temporal_positions(self, frames: int) -> Tensor:
        if frames == self.base_frames:
            return self.temporal_position
        return cast(
            Tensor,
            functional.interpolate(
                self.temporal_position.transpose(1, 2),
                size=frames,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2),
        )

    def forward(self, video: Tensor) -> BackboneOutput:
        self.validate_video(video)
        batch, frames, channels, height, width = video.shape
        if height % self.patch_size or width % self.patch_size:
            raise ModelInputError("Video height and width must be divisible by patch_size")
        frame_batch = video.reshape(batch * frames, channels, height, width)
        embedded = self.patch_embed(frame_batch)
        patch_height, patch_width = embedded.shape[-2:]
        patches = (
            embedded.flatten(2)
            .transpose(1, 2)
            .reshape(batch, frames, patch_height * patch_width, self.output_dim)
        )
        patches = patches + self._spatial_positions(patch_height, patch_width)[:, None]
        patches = patches + self._temporal_positions(frames)[:, :, None]
        patches = self.position_dropout(patches)
        class_token = (self.class_token + self.class_position).expand(batch, -1, -1)
        for block in self.blocks:
            class_token, patches = block(class_token, patches)
        class_token = self.norm(class_token)
        patches = self.norm(patches)
        temporal_tokens = patches.mean(dim=2)
        pooled = class_token[:, 0]
        return BackboneOutput(temporal_tokens=temporal_tokens, pooled=pooled)
