from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - dependency-light environment
    torch = None

if torch is not None:
    from mcformer.models.classifier import AuxiliaryFormer, MCFormer
    from mcformer.models.registry import ModelConfigurationError, build_model


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "data": {"num_frames": 4},
            "model": {
                "name": "timesformer_base_divided_space_time",
                "num_classes": 3,
                "patch_size": 16,
                "embed_dim": 16,
                "depth": 1,
                "num_heads": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "stochastic_depth": 0.0,
                "mcim": {"enabled": True},
            },
        }

    def test_builds_mcformer_only_with_explicit_random_opt_in(self) -> None:
        model, report = build_model(self.config, allow_random_initialization=True)
        self.assertIsInstance(model, MCFormer)
        self.assertIsNone(report)

    def test_rejects_silent_random_initialization(self) -> None:
        with self.assertRaisesRegex(ModelConfigurationError, "checkpoint and SHA-256"):
            build_model(self.config)

    def test_builds_configurable_auxiliary_former(self) -> None:
        self.config["model"]["auxiliary_heads"] = [
            {
                "name": "hallucination",
                "target": "hallucination",
                "kind": "vector",
                "output_dim": 256,
                "weight": 1.0,
            }
        ]
        model, _ = build_model(self.config, allow_random_initialization=True)
        self.assertIsInstance(model, AuxiliaryFormer)


if __name__ == "__main__":
    unittest.main()
