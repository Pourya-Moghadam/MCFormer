from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcformer.auxiliary.cache import ObservationCache, configuration_digest
from mcformer.auxiliary.corruption_cache import CorruptionSpec, build_corrupted_cache
from mcformer.auxiliary.pipeline import TargetSettings, build_sample_target
from mcformer.data.dataset import ClipSample
from mcformer.data.transforms import make_spatial_transform
from mcformer.engine.data import collate_video_samples
from mcformer.experiments import load_experiment_matrix
from test_cache_pipeline import bundle

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - dependency-light environment
    np = None
    torch = None
    nn = None

if torch is not None:
    from mcformer.engine.losses import auxiliary_former_loss
    from mcformer.models.classifier import (
        AuxiliaryFormer,
        AuxiliaryHeadDefinition,
        VideoClassifier,
    )
    from mcformer.models.common import BackboneOutput, VideoBackbone


class ControlledTargetTests(unittest.TestCase):
    def test_target_mapping_contains_all_frozen_auxiliary_schemas(self) -> None:
        spatial = make_spatial_transform(
            width=100,
            height=100,
            training=False,
            resize_short_side=100,
            output_size=100,
        )
        target = build_sample_target(
            bundle(),
            frame_indices=tuple(range(6)),
            spatial_transform=spatial,
            settings=TargetSettings(gaussian_sigma_frames=0),
        ).as_mapping()
        targets = target["targets"]
        self.assertEqual(
            set(targets), {"temporal_gated", "temporal_ungated", "spatial", "hallucination"}
        )
        self.assertEqual(len(targets["spatial"]["target"]), 6)
        self.assertEqual(len(targets["hallucination"]["target"]), 48)
        self.assertTrue(all(targets["spatial"]["mask"]))

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_collation_preserves_named_targets_and_masks(self) -> None:
        spatial = make_spatial_transform(
            width=100,
            height=100,
            training=False,
            resize_short_side=100,
            output_size=100,
        )
        auxiliary = build_sample_target(
            bundle(),
            frame_indices=tuple(range(6)),
            spatial_transform=spatial,
            settings=TargetSettings(gaussian_sigma_frames=0),
        ).as_mapping()
        sample = ClipSample(
            video=np.zeros((6, 3, 2, 2), dtype=np.float32),
            label=0,
            sample_id="sample",
            frame_indices=tuple(range(6)),
            padding_mask=(False,) * 6,
            spatial_transform=spatial,
            auxiliary=auxiliary,
        )
        batch = collate_video_samples([sample, sample])
        self.assertEqual(batch.auxiliary_targets["spatial"].shape, (2, 6))
        self.assertEqual(batch.auxiliary_targets["hallucination"].shape, (2, 48))

    def test_corruption_cache_is_content_addressed_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = ObservationCache(root / "source", cache_key=configuration_digest({"raw": 1}))
            source.initialize({"raw": 1})
            source.write(bundle())
            first, first_report = build_corrupted_cache(
                source,
                str(root / "first"),
                sample_ids=(bundle().sample_id,),
                spec=CorruptionSpec("wrist_noise", 0.02),
            )
            second, second_report = build_corrupted_cache(
                source,
                str(root / "second"),
                sample_ids=(bundle().sample_id,),
                spec=CorruptionSpec("wrist_noise", 0.02),
            )
            self.assertEqual(first.cache_key, second.cache_key)
            self.assertEqual(first.read(bundle().sample_id), second.read(bundle().sample_id))
            self.assertEqual(first_report["changed_samples"], 1)
            self.assertEqual(first_report["cache_key"], second_report["cache_key"])

    def test_complete_experiment_matrix_resolves_all_variants(self) -> None:
        root = Path(__file__).resolve().parents[1]
        matrix = load_experiment_matrix(root / "configs/sweep/e06_e12.json")
        self.assertEqual(matrix.model_seeds, (17, 29, 43))
        self.assertEqual(
            {variant.experiment for variant in matrix.variants},
            {"E06", "E07", "E08", "E09", "E10", "E11", "E12"},
        )
        self.assertEqual(len(matrix.variants), 45)
        self.assertEqual(
            sum(variant.cache_corruption is not None for variant in matrix.variants), 7
        )


if torch is not None:

    class TinyBackbone(VideoBackbone):
        output_dim = 8
        temporal_dim = 8

        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(3, 8)

        def forward(self, video: torch.Tensor) -> BackboneOutput:
            temporal = self.projection(video.mean(dim=(3, 4)))
            return BackboneOutput(temporal, temporal.mean(dim=1))


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AuxiliaryFormerTests(unittest.TestCase):
    def test_combined_temporal_heads_and_loss(self) -> None:
        definitions = (
            AuxiliaryHeadDefinition("temporal", "temporal_gated", "temporal", 6, 1.0),
            AuxiliaryHeadDefinition("spatial", "spatial", "temporal", 6, 1.0),
        )
        model = AuxiliaryFormer(VideoClassifier(TinyBackbone(), 3), definitions, output_frames=6)
        output = model(torch.randn(2, 6, 3, 2, 2))
        losses = auxiliary_former_loss(
            output,
            torch.tensor([0, 1]),
            {
                "temporal_gated": torch.zeros(2, 6),
                "spatial": torch.zeros(2, 6),
            },
            {
                "temporal_gated": torch.ones(2, 6, dtype=torch.bool),
                "spatial": torch.ones(2, 6, dtype=torch.bool),
            },
        )
        losses.total.backward()
        self.assertEqual(set(losses.per_head), {"temporal", "spatial"})
        self.assertEqual(float(losses.valid_positions), 24.0)
        self.assertIsNotNone(model.rgb_model.backbone.projection.weight.grad)

    def test_hallucination_head_is_training_only(self) -> None:
        definition = AuxiliaryHeadDefinition("hallucination", "hallucination", "vector", 256, 1.0)
        model = AuxiliaryFormer(VideoClassifier(TinyBackbone(), 3), (definition,), output_frames=6)
        video = torch.randn(2, 6, 3, 2, 2)
        output = model(video)
        self.assertEqual(output.auxiliary_predictions["hallucination"].shape, (2, 256))
        exported = model.export_rgb_model()
        torch.testing.assert_close(output.logits, exported(video).logits)
        self.assertFalse(any("hallucination" in name for name, _ in exported.named_parameters()))


if __name__ == "__main__":
    unittest.main()
