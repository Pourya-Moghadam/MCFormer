from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mcformer.benchmarks.timing import BenchmarkError, TimingAccumulator
from mcformer.benchmarks.training_cost import (
    aggregate_cache_inventories,
    aggregate_epoch_histories,
    aggregate_preprocessing_trials,
    compare_training_profiles,
)
from mcformer.evaluation.features import FeatureArchive, FeatureArtifactError
from mcformer.visualization.qualitative import select_qualitative_frames
from mcformer.visualization.tsne import select_paired_features


def test_timing_accumulator_reports_sample_sd() -> None:
    timing = TimingAccumulator(samples=[1.0, 2.0, 3.0])
    summary = timing.summary()
    assert summary["mean"] == 2.0
    assert summary["sample_sd"] == 1.0
    with pytest.raises(BenchmarkError):
        timing.add(0.0)


def test_preprocessing_cost_requires_identical_five_trials(tmp_path: Path) -> None:
    paths: list[Path] = []
    for trial in range(1, 6):
        path = tmp_path / f"trial_{trial}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "trial": trial,
                    "sample_id": "a",
                    "frames": 32,
                    "hrnet_seconds": 2.0,
                    "yolov8_inference_seconds": 4.0,
                    "object_pipeline_seconds": 5.0,
                    "target_generation_seconds": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    result = aggregate_preprocessing_trials(paths)
    assert result["trials"] == 5
    assert result["stages"]["hrnet_seconds"]["mean_frames_per_second"] == 16.0
    with pytest.raises(BenchmarkError):
        aggregate_preprocessing_trials(paths[:4])


def test_epoch_cost_excludes_warmup(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "epochs": [
                    {"training_seconds": value, "peak_training_memory_bytes": value * 100}
                    for value in (100, 60, 120, 180)
                ]
            }
        ),
        encoding="utf-8",
    )
    result = aggregate_epoch_histories([path])
    assert result["minutes"] == [1.0, 2.0, 3.0]
    assert result["peak_memory_bytes"] == 18_000


def test_cache_inventory_uses_decimal_gb(tmp_path: Path) -> None:
    paths: list[Path] = []
    for trial in range(5):
        path = tmp_path / f"inventory_{trial}.json"
        path.write_text(
            json.dumps(
                {"trial": trial + 1, "samples": 2, "frames": 64, "cache_bytes": 1_000_000_000}
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = aggregate_cache_inventories(paths)
    assert result["mean_cache_gb_decimal"] == 1.0


def test_training_profile_comparison(tmp_path: Path) -> None:
    paths: list[Path] = []
    for name, macs, parameters in (("baseline", 100.0, 20), ("method", 110.0, 25)):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "precision": "fp16",
                    "macs_per_clip": macs,
                    "parameters": {"trainable": parameters},
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = compare_training_profiles(paths[0], paths[1])
    assert result["training_only_macs_per_clip"] == 10.0
    assert result["training_only_parameters"] == 5


def test_paired_feature_selection_is_lexical_and_capped() -> None:
    archive = FeatureArchive(
        sample_ids=("a", "b", "c", "d"),
        labels=(0, 0, 1, 1),
        features=np.arange(12, dtype=np.float32).reshape(4, 3),
    )
    baseline, method = select_paired_features(archive, archive, class_ids=(0, 1), cap_per_class=1)
    assert baseline.sample_ids == method.sample_ids == ("a", "c")
    incompatible = FeatureArchive(
        sample_ids=("a", "b", "c", "x"), labels=archive.labels, features=archive.features
    )
    with pytest.raises(FeatureArtifactError):
        select_paired_features(archive, incompatible, class_ids=(0,), cap_per_class=1)


def test_qualitative_frame_rule() -> None:
    target = {
        "coupling": {
            "target": [0.0, 0.1, 0.0, 0.8, 0.3],
            "gate": [False, True, True, True, True],
        }
    }
    assert select_qualitative_frames(target) == (1, 1, 3, 4)
