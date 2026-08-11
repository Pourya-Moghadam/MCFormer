from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - dependency-light environment
    torch = None

if torch is not None:
    from mcformer.engine.losses import MCFormerLoss, masked_mse_loss
    from mcformer.models.classifier import MCFormerOutput


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelLossTests(unittest.TestCase):
    def test_masked_mse_uses_only_valid_positions(self) -> None:
        prediction = torch.tensor([[1.0, 2.0, 9.0]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0, 0.0]])
        loss, count = masked_mse_loss(prediction, target, torch.tensor([[True, True, False]]))
        self.assertAlmostEqual(float(loss.detach()), 2.5, places=5)
        self.assertEqual(float(count.detach()), 2.0)
        loss.backward()
        self.assertEqual(float(prediction.grad[0, 2]), 0.0)

    def test_empty_mask_is_differentiable_exact_zero(self) -> None:
        prediction = torch.randn(2, 3, requires_grad=True)
        loss, count = masked_mse_loss(
            prediction, torch.zeros_like(prediction), torch.zeros_like(prediction, dtype=torch.bool)
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(count.detach()), 0.0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_combined_loss(self) -> None:
        output = MCFormerOutput(
            logits=torch.tensor([[2.0, -1.0]], requires_grad=True),
            coupling=torch.tensor([[1.0, 2.0]], requires_grad=True),
            temporal_tokens=torch.empty(1, 0, 0),
            pooled=torch.empty(1, 0),
        )
        result = MCFormerLoss(coupling_weight=1.0)(
            output,
            torch.tensor([0]),
            torch.zeros(1, 2),
            torch.tensor([[True, False]]),
        )
        torch.testing.assert_close(result.total, result.classification + result.coupling)
        self.assertEqual(float(result.valid_coupling_positions.detach()), 1.0)


if __name__ == "__main__":
    unittest.main()
