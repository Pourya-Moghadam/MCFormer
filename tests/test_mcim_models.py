from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    torch = None
    nn = None

if torch is not None:
    from mcformer.models.checkpoints import CheckpointError, load_local_checkpoint
    from mcformer.models.classifier import MCFormer, VideoClassifier, export_rgb_checkpoint
    from mcformer.models.common import BackboneOutput, VideoBackbone
    from mcformer.models.mcim import MCIM
    from mcformer.reproducibility import sha256_file


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MCIMModelTests(unittest.TestCase):
    if torch is not None:

        class TinyBackbone(VideoBackbone):
            output_dim = 8
            temporal_dim = 8

            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Linear(3, self.output_dim)

            def forward(self, video: torch.Tensor) -> BackboneOutput:
                self.validate_video(video)
                temporal = self.projection(video.mean(dim=(3, 4)))
                return BackboneOutput(temporal_tokens=temporal, pooled=temporal.mean(dim=1))

    def test_exact_release_parameter_count(self) -> None:
        module = MCIM(768, hidden_dim=384, output_frames=32)
        self.assertEqual(module.parameter_count, 295_681)

    def test_temporal_interpolation_aligns_endpoints(self) -> None:
        module = MCIM(1, hidden_dim=1, output_frames=5)
        with torch.no_grad():
            first = module.layers[0]
            second = module.layers[2]
            first.weight.fill_(1)
            first.bias.zero_()
            second.weight.fill_(1)
            second.bias.zero_()
        result = module(torch.tensor([[[1.0], [3.0]]]))
        torch.testing.assert_close(result, torch.tensor([[1.0, 1.5, 2.0, 2.5, 3.0]]))

    def test_mcim_does_not_change_classification_path(self) -> None:
        model = MCFormer(VideoClassifier(self.TinyBackbone(), 4), output_frames=6)
        video = torch.randn(2, 6, 3, 4, 4)
        before = model(video).logits.detach().clone()
        with torch.no_grad():
            for parameter in model.mcim.parameters():
                parameter.add_(torch.randn_like(parameter) * 100)
        after = model(video).logits.detach()
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_gradients_reach_backbone_and_mcim(self) -> None:
        model = MCFormer(VideoClassifier(self.TinyBackbone(), 4), output_frames=6)
        output = model(torch.randn(2, 6, 3, 4, 4))
        (output.logits.sum() + output.coupling.sum()).backward()
        self.assertIsNotNone(model.rgb_model.backbone.projection.weight.grad)
        self.assertIsNotNone(model.mcim.layers[0].weight.grad)

    def test_rgb_export_matches_logits_and_contains_no_mcim(self) -> None:
        model = MCFormer(VideoClassifier(self.TinyBackbone(), 4), output_frames=6).eval()
        video = torch.randn(2, 6, 3, 4, 4)
        expected = model(video).logits
        exported = model.export_rgb_model().eval()
        torch.testing.assert_close(expected, exported(video).logits, rtol=0, atol=0)
        self.assertFalse(any("mcim" in key for key in exported.state_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.pt"
            export_rgb_checkpoint(model, str(path))
            saved = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(saved["model_type"], "rgb_only")
        self.assertFalse(any("mcim" in key for key in saved["state_dict"]))

    def test_checkpoint_requires_matching_digest(self) -> None:
        source = nn.Linear(3, 2)
        target = nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"state_dict": source.state_dict()}, path)
            digest = sha256_file(path)
            report = load_local_checkpoint(target, path, expected_sha256=digest)
            self.assertEqual(report.sha256, digest)
            with self.assertRaisesRegex(CheckpointError, "mismatch"):
                load_local_checkpoint(target, path, expected_sha256="0" * 64)
        torch.testing.assert_close(source.weight, target.weight)

    def test_checkpoint_aligns_common_wrapper_prefixes(self) -> None:
        source = nn.Linear(3, 2)
        target = nn.Linear(3, 2)
        prefixed = {f"module.model.{key}": value for key, value in source.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"state_dict": prefixed}, path)
            load_local_checkpoint(target, path, expected_sha256=sha256_file(path))
        torch.testing.assert_close(source.weight, target.weight)


if __name__ == "__main__":
    unittest.main()
