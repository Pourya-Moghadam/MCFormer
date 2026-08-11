"""Pinned torchvision adapters for Video Swin-T and MViTv2-S."""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor, nn

from mcformer.models.common import BackboneOutput, ModelInputError, VideoBackbone


class TorchvisionBackboneError(RuntimeError):
    """Raised when the pinned torchvision video API is unavailable or incompatible."""


def _require_torchvision() -> Any:
    try:
        import torchvision
    except (ImportError, RuntimeError) as error:
        raise TorchvisionBackboneError(
            "Video Swin-T and MViTv2-S require torchvision==0.19.1"
        ) from error
    if torchvision.__version__.split("+")[0] != "0.19.1":
        raise TorchvisionBackboneError(
            f"Expected torchvision 0.19.1, found {torchvision.__version__}"
        )
    return torchvision


class VideoSwinTinyBackbone(VideoBackbone):
    """Video Swin-T feature adapter with a selectable auxiliary insertion stage."""

    output_dim = 768
    temporal_dim = 768

    def __init__(self, *, insertion_stage: int = 4) -> None:
        super().__init__()
        if insertion_stage not in {1, 2, 3, 4}:
            raise ValueError("insertion_stage must be one of 1, 2, 3, or 4")
        torchvision = _require_torchvision()
        self.model = torchvision.models.video.swin3d_t(weights=None, stochastic_depth_prob=0.2)
        self.model.head = nn.Identity()
        self.insertion_stage = insertion_stage
        self.auxiliary_dim = 96 * 2 ** (insertion_stage - 1)
        self.auxiliary_norm = (
            nn.Identity() if insertion_stage == 4 else nn.LayerNorm(self.auxiliary_dim)
        )
        self.temporal_dim = self.auxiliary_dim

    def forward(self, video: Tensor) -> BackboneOutput:
        self.validate_video(video)
        value = video.permute(0, 2, 1, 3, 4).contiguous()
        value = self.model.patch_embed(value)
        value = self.model.pos_drop(value)
        auxiliary: Tensor | None = None
        stage = 0
        for index, layer in enumerate(self.model.features):
            value = layer(value)
            if index % 2 == 0:
                stage += 1
                if stage == self.insertion_stage:
                    auxiliary = value
        value = self.model.norm(value)
        if self.insertion_stage == 4:
            auxiliary = value
        if auxiliary is None:
            raise TorchvisionBackboneError("Could not locate requested Video Swin stage")
        normalized_auxiliary = cast(Tensor, self.auxiliary_norm(auxiliary))
        temporal_tokens = normalized_auxiliary.mean(dim=(2, 3))
        pooled = value.mean(dim=(1, 2, 3))
        return BackboneOutput(temporal_tokens=temporal_tokens, pooled=pooled)


class MViTv2SmallBackbone(VideoBackbone):
    """MViTv2-S adapter exposing final multiscale temporal tokens."""

    output_dim = 768
    temporal_dim = 768

    def __init__(self, *, num_frames: int = 32, image_size: int = 224) -> None:
        super().__init__()
        if num_frames != 32 or image_size != 224:
            raise ValueError("Pinned MViTv2-S requires 32 frames at 224x224")
        torchvision = _require_torchvision()
        self.model = torchvision.models.video.mvit_v2_s(weights=None)
        self.model.head = nn.Identity()
        self.num_frames = num_frames
        self.image_size = image_size

    def forward(self, video: Tensor) -> BackboneOutput:
        self.validate_video(video)
        if video.shape[1] != self.num_frames or video.shape[3:] != (
            self.image_size,
            self.image_size,
        ):
            raise ModelInputError(
                f"MViTv2-S expects B,{self.num_frames},3,{self.image_size},{self.image_size}"
            )
        value = video.permute(0, 2, 1, 3, 4).contiguous()
        value = self.model.conv_proj(value)
        thw = (value.shape[2], value.shape[3], value.shape[4])
        value = value.flatten(2).transpose(1, 2)
        value = self.model.pos_encoding(value)
        for block in self.model.blocks:
            value, thw = block(value, thw)
        value = self.model.norm(value)
        expected_patches = thw[0] * thw[1] * thw[2]
        if value.shape[1] != expected_patches + 1:
            raise TorchvisionBackboneError("MViTv2 token count disagrees with final THW shape")
        patches = value[:, 1:].reshape(value.shape[0], thw[0], thw[1], thw[2], value.shape[-1])
        temporal_tokens = patches.mean(dim=(2, 3))
        return BackboneOutput(temporal_tokens=temporal_tokens, pooled=value[:, 0])
