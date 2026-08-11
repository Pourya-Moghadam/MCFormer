from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover - dependency-light environment
    np = None
    torch = None

if torch is not None:
    from torch import nn
    from torch.utils.data import Dataset

    from mcformer.data.dataset import ClipSample
    from mcformer.data.transforms import SpatialTransform
    from mcformer.engine.checkpointing import (
        TrainingCheckpointError,
        resume_training_checkpoint,
        save_training_checkpoint,
    )
    from mcformer.engine.data import DistributedEvalSampler, build_data_loader
    from mcformer.engine.distributed import DistributedContext
    from mcformer.engine.optim import OptimizerSettings, WarmupCosineScheduler, build_adamw
    from mcformer.engine.trainer import Trainer, TrainerSettings
    from mcformer.evaluation.evaluator import evaluate_model, write_evaluation_artifacts
    from mcformer.evaluation.metrics import ClassificationAccumulator
    from mcformer.models.classifier import ClassificationOutput


if torch is not None:

    class TinyDataset(Dataset[ClipSample]):
        def __init__(self, labels: tuple[int, ...]) -> None:
            self.labels = labels
            self.epoch = 0

        def __len__(self) -> int:
            return len(self.labels)

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __getitem__(self, index: int) -> ClipSample:
            label = self.labels[index]
            transform = SpatialTransform(1, 1, 1, 1, 0, 0, 1, 1, 1, False)
            video = np.full((2, 3, 1, 1), float(label), dtype=np.float32)
            return ClipSample(
                video=video,
                label=label,
                sample_id=f"sample-{index:02d}",
                frame_indices=(0, 1),
                padding_mask=(False, False),
                spatial_transform=transform,
                auxiliary=None,
            )

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Linear(1, 2)

        def forward(self, video: torch.Tensor) -> ClassificationOutput:
            pooled = video.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
            return ClassificationOutput(
                logits=self.classifier(pooled),
                temporal_tokens=pooled.unsqueeze(1),
                pooled=pooled,
            )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainingEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = DistributedContext(0, 1, 0, torch.device("cpu"))

    def test_metrics_report_absent_classes_and_top_k(self) -> None:
        accumulator = ClassificationAccumulator(3, torch.device("cpu"))
        accumulator.update(torch.tensor([[5.0, 0.0, -1.0], [2.0, 3.0, 1.0]]), torch.tensor([0, 0]))
        metrics = accumulator.compute(self.context)
        self.assertEqual(metrics.samples, 2)
        self.assertEqual(metrics.top1_accuracy, 0.5)
        self.assertEqual(metrics.top5_accuracy, 1.0)
        self.assertEqual(metrics.mean_class_accuracy, 0.5)
        self.assertEqual(metrics.absent_classes, (1, 2))

    def test_distributed_eval_sampler_has_no_padding_duplicates(self) -> None:
        dataset = TinyDataset((0, 1, 0, 1, 0))
        partitions = [
            list(DistributedEvalSampler(dataset, rank=rank, world_size=2)) for rank in range(2)
        ]
        self.assertEqual(sorted(partitions[0] + partitions[1]), list(range(5)))
        self.assertFalse(set(partitions[0]) & set(partitions[1]))

    def test_evaluation_writes_sorted_complete_predictions(self) -> None:
        dataset = TinyDataset((0, 1, 0))
        loader, _ = build_data_loader(
            dataset,
            batch_size=2,
            training=False,
            seed=17,
            num_workers=0,
            pin_memory=False,
            distributed=False,
        )
        model = TinyModel()
        result = evaluate_model(
            model,
            loader,
            context=self.context,
            num_classes=2,
            mixed_precision="none",
        )
        with tempfile.TemporaryDirectory() as directory:
            write_evaluation_artifacts(result, directory)
            rows = [
                json.loads(line)
                for line in (Path(directory) / "predictions.jsonl").read_text().splitlines()
            ]
            sample_ids = [row["sample_id"] for row in rows]
            self.assertEqual(sample_ids, sorted(sample_ids))
            metrics = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(metrics["samples"], 3)

    def test_checkpoint_round_trip_and_identity_guards(self) -> None:
        model = TinyModel()
        optimizer = build_adamw(model.parameters(), OptimizerSettings())
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_steps=4,
            warmup_steps=1,
            base_learning_rate=1e-4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            digest = save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch_completed=0,
                best_metric=0.5,
                best_epoch=0,
                config_sha256="a" * 64,
                seed=17,
            )
            with torch.no_grad():
                model.classifier.weight.zero_()
            state = resume_training_checkpoint(
                path,
                expected_sha256=digest,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                config_sha256="a" * 64,
                seed=17,
            )
            self.assertEqual(state.next_epoch, 1)
            self.assertNotEqual(float(model.classifier.weight.detach().abs().sum()), 0.0)
            with self.assertRaises(TrainingCheckpointError):
                resume_training_checkpoint(
                    path,
                    expected_sha256="0" * 64,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    config_sha256="a" * 64,
                    seed=17,
                )

    def test_trainer_runs_and_exports_complete_checkpoints(self) -> None:
        dataset = TinyDataset((0, 1, 0, 1))
        loader, _ = build_data_loader(
            dataset,
            batch_size=2,
            training=True,
            seed=17,
            num_workers=0,
            pin_memory=False,
            distributed=False,
        )
        model = TinyModel()
        optimizer = build_adamw(model.parameters(), OptimizerSettings())
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_steps=2,
            warmup_steps=0,
            base_learning_rate=1e-4,
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            settings=TrainerSettings(1, 1, 5.0, "none", 1.0, 20, "top1_accuracy"),
            context=self.context,
            num_classes=2,
            logger=logging.getLogger("mcformer-test"),
        )
        with tempfile.TemporaryDirectory() as directory:
            history = trainer.fit(
                train_loader=loader,
                validation_loader=loader,
                train_dataset=dataset,
                train_sampler=None,
                output_dir=directory,
                config_sha256="a" * 64,
                seed=17,
            )
            self.assertEqual(len(history), 1)
            checkpoints = Path(directory) / "checkpoints"
            self.assertTrue((checkpoints / "last.pt").is_file())
            self.assertTrue((checkpoints / "final.pt").is_file())
            final_metadata = json.loads((checkpoints / "final.json").read_text())
            self.assertEqual(len(final_metadata["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
