from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcformer.cli.aggregate_sweep import main as aggregate_sweep_main
from mcformer.cli.analyze import main as analyze_main
from mcformer.data.manifest import Manifest, SampleRecord
from mcformer.data.protocols import ProtocolSplit
from mcformer.evaluation.diagnostics import pair_diagnostic, selected_confusion
from mcformer.evaluation.predictions import (
    PredictionArtifactError,
    ensemble_predictions,
    load_predictions,
)
from mcformer.evaluation.statistics import paired_bootstrap_gain, summarize_runs


def _write_predictions(
    path: Path,
    labels: tuple[int, ...],
    predictions: tuple[int, ...],
    *,
    prefix: str = "test",
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, (label, prediction) in enumerate(zip(labels, predictions, strict=True)):
            probabilities = [0.05, 0.05, 0.05]
            probabilities[prediction] = 0.90
            row = {
                "sample_id": f"{prefix}-{index:02d}",
                "label": label,
                "prediction": prediction,
                "logits": [0.0, 0.0, 0.0],
                "probabilities": probabilities,
                "valid_frames": 32,
                "padded_frames": 0,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _manifest_and_split(directory: Path) -> tuple[Path, Path]:
    records: list[SampleRecord] = []
    for partition, count in (("train", 3), ("validation", 3), ("test", 6)):
        for index in range(count):
            label = index % 3
            records.append(
                SampleRecord(
                    sample_id=f"{partition}-{index:02d}",
                    dataset="toyota_smarthome",
                    rgb_path=f"{partition}-{index:02d}.mp4",
                    label_id=label,
                    label_name=("Action A", "Action B", "Action C")[label],
                )
            )
    manifest = Manifest(records, expected_classes=3)
    manifest_path = directory / "manifest.jsonl"
    manifest.write_jsonl(manifest_path)
    split = ProtocolSplit(
        dataset="toyota_smarthome",
        protocol="cs",
        train=tuple(f"train-{index:02d}" for index in range(3)),
        validation=tuple(f"validation-{index:02d}" for index in range(3)),
        test=tuple(f"test-{index:02d}" for index in range(6)),
        validation_strategy="fixture",
    )
    split_path = directory / "split.json"
    split.write_json(split_path)
    return manifest_path, split_path


class PaperAnalysisTests(unittest.TestCase):
    def test_loader_rejects_unsorted_and_ensemble_validates_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            _write_predictions(first, (0, 1, 2), (0, 1, 2))
            _write_predictions(second, (0, 1, 2), (0, 2, 2))
            ensemble = ensemble_predictions([load_predictions(first), load_predictions(second)])
            self.assertEqual(len(ensemble.records), 3)
            rows = first.read_text(encoding="utf-8").splitlines()
            first.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
            with self.assertRaises(PredictionArtifactError):
                load_predictions(first)

    def test_repeated_summary_and_bootstrap_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.jsonl"
            method_path = root / "method.jsonl"
            labels = (0, 0, 1, 1, 2, 2)
            _write_predictions(baseline_path, labels, (0, 1, 1, 0, 2, 0))
            _write_predictions(method_path, labels, (0, 0, 1, 1, 2, 0))
            baseline = load_predictions(baseline_path)
            method = load_predictions(method_path)
            summary = summarize_runs([baseline, baseline, baseline], metric="mca")
            self.assertAlmostEqual(summary.mean, 0.5)
            self.assertEqual(summary.sample_standard_deviation, 0.0)
            first = paired_bootstrap_gain(baseline, method, metric="mca", resamples=100, seed=2026)
            second = paired_bootstrap_gain(baseline, method, metric="mca", resamples=100, seed=2026)
            self.assertEqual(first.draws, second.draws)
            self.assertAlmostEqual(first.point_gain, 1 / 3)
            self.assertTrue(first.stratified)

    def test_pair_and_selected_confusion_preserve_other_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            _write_predictions(path, (0, 0, 1, 1, 2, 2), (0, 2, 1, 0, 2, 2))
            predictions = load_predictions(path)
            pair = pair_diagnostic(predictions, (0, 1))
            self.assertEqual(pair.confusion_with_other, ((1, 0, 1), (1, 1, 0)))
            matrix = selected_confusion(predictions, (0, 1))
            self.assertEqual(matrix[0], (0.5, 0.0))
            self.assertEqual(matrix[1], (0.5, 0.5))

    def test_analysis_cli_generates_e02_to_e05_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, split = _manifest_and_split(root)
            labels = (0, 1, 2, 0, 1, 2)
            baseline_arguments: list[str] = []
            method_arguments: list[str] = []
            for seed in (17, 29, 43):
                baseline = root / f"baseline-{seed}.jsonl"
                method = root / f"method-{seed}.jsonl"
                _write_predictions(baseline, labels, (0, 2, 2, 1, 1, 0))
                _write_predictions(method, labels, (0, 1, 2, 0, 1, 0))
                baseline_arguments.extend(("--baseline", f"{seed}={baseline}"))
                method_arguments.extend(("--method", f"{seed}={method}"))
            statistics_output = root / "statistics"
            result = analyze_main(
                [
                    "statistics",
                    "--manifest",
                    str(manifest),
                    "--protocol-split",
                    str(split),
                    *baseline_arguments,
                    *method_arguments,
                    "--metric",
                    "mca",
                    "--bootstrap-resamples",
                    "100",
                    "--output",
                    str(statistics_output),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue((statistics_output / "confidence_interval.json").is_file())
            subsets = root / "subsets.json"
            subsets.write_text(
                json.dumps(
                    {
                        "manipulation_actions": ["Action A", "Action B", "Action C"],
                        "same_object_pair_1": ["Action A", "Action B"],
                        "same_object_pair_2": ["Action A", "Action C"],
                        "same_object_pair_3": ["Action B", "Action C"],
                        "confusion_matrix_actions": ["Action A", "Action B"],
                    }
                ),
                encoding="utf-8",
            )
            diagnostics_output = root / "diagnostics"
            result = analyze_main(
                [
                    "diagnostics",
                    "--manifest",
                    str(manifest),
                    "--protocol-split",
                    str(split),
                    *baseline_arguments,
                    *method_arguments,
                    "--subsets",
                    str(subsets),
                    "--task",
                    "subset",
                    "--task",
                    "pairs",
                    "--task",
                    "confusion",
                    "--no-plot",
                    "--output",
                    str(diagnostics_output),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue((diagnostics_output / "same_object_pairs.csv").is_file())
            self.assertTrue((diagnostics_output / "method_confusion.csv").is_file())

    def test_controlled_sweep_aggregation_requires_three_seed_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, split = _manifest_and_split(root)
            labels = (0, 1, 2, 0, 1, 2)
            arguments: list[str] = []
            for variant, predictions in (
                ("baseline", (0, 2, 2, 1, 1, 0)),
                ("gated", (0, 1, 2, 0, 1, 0)),
            ):
                for seed in (17, 29, 43):
                    path = root / f"{variant}-{seed}.jsonl"
                    _write_predictions(path, labels, predictions)
                    arguments.extend(("--run", f"{variant}:{seed}={path}"))
            output = root / "sweep"
            result = aggregate_sweep_main(
                [
                    "--manifest",
                    str(manifest),
                    "--protocol-split",
                    str(split),
                    "--metric",
                    "mca",
                    *arguments,
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            rows = (output / "sweep_summary.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)
            self.assertTrue((output / "sweep_rows.tex").is_file())


if __name__ == "__main__":
    unittest.main()
