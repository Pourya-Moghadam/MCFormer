"""Aggregate raw E14 preprocessing and steady-state epoch measurements."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from mcformer.benchmarks.timing import BenchmarkError


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BenchmarkError(f"{path}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise BenchmarkError(f"Timing file is empty: {path}")
    return rows


def aggregate_preprocessing_trials(paths: list[str | Path]) -> dict[str, Any]:
    """Validate five independent full-partition trials and summarize stage throughput."""

    if len(paths) != 5:
        raise BenchmarkError("E14 requires exactly five preprocessing timing trials")
    per_stage: dict[str, list[float]] = defaultdict(list)
    end_to_end_hours: list[float] = []
    identities: tuple[str, ...] | None = None
    frames_per_trial: int | None = None
    for raw_path in paths:
        rows = _jsonl(Path(raw_path).expanduser().resolve())
        sample_ids = tuple(str(row["sample_id"]) for row in rows)
        frames = sum(int(row["frames"]) for row in rows)
        if identities is None:
            identities, frames_per_trial = sample_ids, frames
        elif sample_ids != identities or frames != frames_per_trial:
            raise BenchmarkError("Preprocessing trials do not contain identical ordered clips")
        stage_names = sorted(
            key for key in rows[0] if key.endswith("_seconds") and key != "decode_seconds"
        )
        for stage in stage_names:
            duration = sum(float(row[stage]) for row in rows)
            if duration <= 0:
                raise BenchmarkError(f"Stage {stage} has a non-positive duration")
            per_stage[stage].append(frames / duration)
        required = ("hrnet_seconds", "object_pipeline_seconds", "target_generation_seconds")
        if not all(all(stage in row for stage in required) for row in rows):
            raise BenchmarkError(
                "Every E14 row must contain HRNet, object-pipeline, and target time"
            )
        end_to_end_hours.append(
            sum(sum(float(row[stage]) for stage in required) for row in rows) / 3600.0
        )
    assert identities is not None and frames_per_trial is not None
    return {
        "schema_version": 1,
        "trials": 5,
        "samples_per_trial": len(identities),
        "frames_per_trial": frames_per_trial,
        "end_to_end_hours": end_to_end_hours,
        "mean_end_to_end_hours": statistics.fmean(end_to_end_hours),
        "sample_sd_end_to_end_hours": statistics.stdev(end_to_end_hours),
        "stages": {
            stage: {
                "frames_per_second": values,
                "mean_frames_per_second": statistics.fmean(values),
                "sample_sd_frames_per_second": statistics.stdev(values),
            }
            for stage, values in sorted(per_stage.items())
        },
    }


def aggregate_cache_inventories(paths: list[str | Path]) -> dict[str, Any]:
    """Validate and summarize five deterministic gzip-cache inventories."""

    if len(paths) != 5:
        raise BenchmarkError("E14 requires exactly five cache inventories")
    values: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or int(value.get("samples", 0)) <= 0:
            raise BenchmarkError(f"Invalid cache inventory: {path}")
        values.append(value)
    identities = {(int(value["samples"]), int(value["frames"])) for value in values}
    if len(identities) != 1:
        raise BenchmarkError("Cache inventories do not describe identical partitions")
    sizes = [int(value["cache_bytes"]) for value in values]
    if any(size <= 0 for size in sizes):
        raise BenchmarkError("Cache inventory size must be positive")
    return {
        "trials": 5,
        "cache_bytes": sizes,
        "cache_gb_decimal": [size / 1_000_000_000 for size in sizes],
        "mean_cache_gb_decimal": statistics.fmean(sizes) / 1_000_000_000,
        "sample_sd_cache_gb_decimal": statistics.stdev(sizes) / 1_000_000_000,
    }


def aggregate_epoch_histories(paths: list[str | Path]) -> dict[str, Any]:
    """Use epoch zero as warm-up and summarize exactly three measured epochs."""

    if not paths:
        raise BenchmarkError("At least one history is required")
    timings: list[float] = []
    memory: list[int] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        epochs = value.get("epochs") if isinstance(value, dict) else None
        if not isinstance(epochs, list) or len(epochs) != 4:
            raise BenchmarkError(f"{path} must contain one warm-up plus three timed epochs")
        timings.extend(float(epoch["training_seconds"]) for epoch in epochs[1:])
        memory.extend(int(epoch["peak_training_memory_bytes"]) for epoch in epochs[1:])
    return {
        "timed_epochs": len(timings),
        "minutes": [value / 60.0 for value in timings],
        "mean_minutes": statistics.fmean(timings) / 60.0,
        "sample_sd_minutes": statistics.stdev(timings) / 60.0,
        "peak_memory_bytes": max(memory),
    }


def compare_training_profiles(baseline_path: str | Path, method_path: str | Path) -> dict[str, Any]:
    """Compare separately measured baseline and auxiliary training graphs."""

    values: list[dict[str, Any]] = []
    for raw_path in (baseline_path, method_path):
        path = Path(raw_path).expanduser().resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("precision") != "fp16":
            raise BenchmarkError(f"Invalid FP16 training profile: {path}")
        values.append(value)
    baseline, method = values
    baseline_parameters = int(baseline["parameters"]["trainable"])
    method_parameters = int(method["parameters"]["trainable"])
    baseline_macs = float(baseline["macs_per_clip"])
    method_macs = float(method["macs_per_clip"])
    if method_parameters < baseline_parameters or method_macs < baseline_macs:
        raise BenchmarkError("Auxiliary training graph is smaller than the paired baseline")
    return {
        "baseline_macs_per_clip": baseline_macs,
        "mcformer_macs_per_clip": method_macs,
        "training_only_macs_per_clip": method_macs - baseline_macs,
        "baseline_trainable_parameters": baseline_parameters,
        "mcformer_trainable_parameters": method_parameters,
        "training_only_parameters": method_parameters - baseline_parameters,
    }
