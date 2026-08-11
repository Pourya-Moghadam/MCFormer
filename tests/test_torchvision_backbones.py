from __future__ import annotations

import types
import unittest
from unittest.mock import patch

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - dependency-light environment
    torch = None
    nn = None

if torch is not None:
    from mcformer.models.torchvision_backbones import (
        MViTv2SmallBackbone,
        VideoSwinTinyBackbone,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchvisionBackboneAdapterTests(unittest.TestCase):
    if torch is not None:

        class ExpandLast(nn.Module):
            def __init__(self, input_dim: int, output_dim: int) -> None:
                super().__init__()
                self.projection = nn.Linear(input_dim, output_dim)

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.projection(value)

        class FakeSwin(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_embed = self.PatchEmbed()
                self.pos_drop = nn.Identity()
                self.features = nn.ModuleList(
                    [
                        nn.Identity(),
                        TorchvisionBackboneAdapterTests.ExpandLast(96, 192),
                        nn.Identity(),
                        TorchvisionBackboneAdapterTests.ExpandLast(192, 384),
                        nn.Identity(),
                        TorchvisionBackboneAdapterTests.ExpandLast(384, 768),
                        nn.Identity(),
                    ]
                )
                self.norm = nn.LayerNorm(768)
                self.head = nn.Linear(768, 400)

            class PatchEmbed(nn.Module):
                def forward(self, value: torch.Tensor) -> torch.Tensor:
                    value = value.permute(0, 2, 3, 4, 1)
                    repeats = 96 // value.shape[-1]
                    return value.repeat(1, 1, 1, 1, repeats)

        class FakeMViTBlock(nn.Module):
            def forward(
                self, value: torch.Tensor, thw: tuple[int, int, int]
            ) -> tuple[torch.Tensor, tuple[int, int, int]]:
                return value, thw

        class FakeMViT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_proj = self.FakeProjection()
                self.pos_encoding = self.AddClassToken()
                self.blocks = nn.ModuleList([TorchvisionBackboneAdapterTests.FakeMViTBlock()])
                self.norm = nn.LayerNorm(768)
                self.head = nn.Linear(768, 400)

            class FakeProjection(nn.Module):
                def forward(self, value: torch.Tensor) -> torch.Tensor:
                    batch = value.shape[0]
                    return value.new_zeros(batch, 768, 16, 2, 2)

            class AddClassToken(nn.Module):
                def forward(self, value: torch.Tensor) -> torch.Tensor:
                    class_token = value.new_zeros(value.shape[0], 1, value.shape[2])
                    return torch.cat((class_token, value), dim=1)

    @staticmethod
    def fake_torchvision(swin: nn.Module | None = None, mvit: nn.Module | None = None) -> object:
        video = types.SimpleNamespace(
            swin3d_t=lambda **kwargs: swin,
            mvit_v2_s=lambda **kwargs: mvit,
        )
        return types.SimpleNamespace(models=types.SimpleNamespace(video=video))

    def test_video_swin_final_and_early_temporal_dimensions(self) -> None:
        fake = self.fake_torchvision(swin=self.FakeSwin())
        with patch("mcformer.models.torchvision_backbones._require_torchvision", return_value=fake):
            final = VideoSwinTinyBackbone(insertion_stage=4)
            early = VideoSwinTinyBackbone(insertion_stage=1)
        video = torch.randn(2, 4, 3, 2, 2)
        self.assertEqual(final(video).temporal_tokens.shape, (2, 4, 768))
        self.assertEqual(early(video).temporal_tokens.shape, (2, 4, 96))
        self.assertEqual(early(video).pooled.shape, (2, 768))

    def test_mvit_token_reshape(self) -> None:
        fake = self.fake_torchvision(mvit=self.FakeMViT())
        with patch("mcformer.models.torchvision_backbones._require_torchvision", return_value=fake):
            model = MViTv2SmallBackbone()
        output = model(torch.randn(1, 32, 3, 224, 224))
        self.assertEqual(output.temporal_tokens.shape, (1, 16, 768))
        self.assertEqual(output.pooled.shape, (1, 768))


if __name__ == "__main__":
    unittest.main()
