from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - dependency-light environment
    torch = None

if torch is not None:
    from mcformer.models.timesformer import TimeSformerBackbone


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TimeSformerTests(unittest.TestCase):
    def test_tiny_backbone_shapes_and_dynamic_positions(self) -> None:
        model = TimeSformerBackbone(
            image_size=16,
            patch_size=8,
            num_frames=4,
            embed_dim=32,
            depth=2,
            num_heads=4,
            stochastic_depth=0.0,
        )
        output = model(torch.randn(2, 3, 3, 24, 16))
        self.assertEqual(output.temporal_tokens.shape, (2, 3, 32))
        self.assertEqual(output.pooled.shape, (2, 32))

    def test_rejects_wrong_layout(self) -> None:
        model = TimeSformerBackbone(
            image_size=16,
            patch_size=8,
            num_frames=4,
            embed_dim=16,
            depth=1,
            num_heads=4,
        )
        with self.assertRaisesRegex(ValueError, "three RGB channels"):
            model(torch.randn(1, 4, 2, 16, 16))


if __name__ == "__main__":
    unittest.main()
