"""Aggregate three-seed E06--E12 controlled experiment predictions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from mcformer.data.manifest import Manifest
from mcformer.data.protocols import ProtocolSplit
from mcformer.data.subsets import load_subset_names, resolve_label_ids
from mcformer.evaluation.artifacts import (
    RawLatex,
    write_analysis_provenance,
    write_csv,
    write_latex_rows,
)
from mcformer.evaluation.diagnostics import subset_mca
from mcformer.evaluation.predictions import PredictionArtifactError, PredictionSet, load_predictions
from mcformer.evaluation.statistics import accuracy, mean_class_accuracy
from mcformer.reproducibility import sha256_file

PAPER_SEEDS = (17, 29, 43)


def _run(value: str) -> tuple[str, int, Path]:
    try:
        identity, path = value.split("=", 1)
        variant, seed_text = identity.rsplit(":", 1)
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected VARIANT:SEED=PATH") from error
    if not variant or not path:
        raise argparse.ArgumentTypeError("Variant and prediction path must be non-empty")
    return variant, seed, Path(path).expanduser().resolve()


def _history(value: str) -> tuple[str, int, Path]:
    return _run(value)


def _mean_sd(values: tuple[float, ...]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol-split", required=True)
    parser.add_argument("--run", action="append", required=True, type=_run)
    parser.add_argument("--history", action="append", default=[], type=_history)
    parser.add_argument("--metric", choices=("accuracy", "mca"), required=True)
    parser.add_argument("--subsets")
    parser.add_argument("--subset-name", default="manipulation_actions")
    parser.add_argument("--output", required=True)
    return parser


def _group_runs(
    values: list[tuple[str, int, Path]],
) -> dict[str, dict[int, Path]]:
    grouped: dict[str, dict[int, Path]] = {}
    for variant, seed, path in values:
        seeds = grouped.setdefault(variant, {})
        if seed in seeds:
            raise PredictionArtifactError(f"Duplicate {variant!r} seed {seed}")
        seeds[seed] = path
    for variant, seeds in grouped.items():
        if set(seeds) != set(PAPER_SEEDS):
            raise PredictionArtifactError(f"{variant!r} requires exactly seeds {PAPER_SEEDS}")
    return grouped


def _coverage(path: Path) -> float:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        epochs = value["epochs"]
        coverage = epochs[-1]["coupling_coverage"]
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise PredictionArtifactError(f"Invalid history artifact {path}: {error}") from error
    return float(coverage)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = Manifest.read_jsonl(args.manifest)
    split = ProtocolSplit.read_json(args.protocol_split, manifest)
    expected_metric = "mca" if split.dataset == "toyota_smarthome" else "accuracy"
    if args.metric != expected_metric:
        raise PredictionArtifactError(f"{split.dataset} requires metric={expected_metric}")
    grouped = _group_runs(args.run)
    histories = _group_runs(args.history) if args.history else {}
    if histories and set(histories) != set(grouped):
        raise PredictionArtifactError("History variants must exactly match prediction variants")
    subset_ids: tuple[int, ...] | None = None
    if args.subsets:
        subset_ids = resolve_label_ids(manifest, load_subset_names(args.subsets, args.subset_name))
    expected_ids = tuple(sorted(split.test))
    detail_rows: list[tuple[object, ...]] = []
    summary_rows: list[tuple[object, ...]] = []
    latex_rows: list[tuple[object, ...]] = []
    sources: dict[str, object] = {}
    reference_labels: tuple[int, ...] | None = None
    for variant in sorted(grouped):
        predictions: dict[int, PredictionSet] = {}
        for seed in PAPER_SEEDS:
            prediction = load_predictions(grouped[variant][seed])
            if tuple(record.sample_id for record in prediction.records) != expected_ids:
                raise PredictionArtifactError(f"{variant}:{seed} does not cover the official test")
            labels = tuple(record.label for record in prediction.records)
            if reference_labels is None:
                reference_labels = labels
            elif labels != reference_labels:
                raise PredictionArtifactError("Sweep variants disagree on paired test labels")
            predictions[seed] = prediction
        metric_function = accuracy if args.metric == "accuracy" else mean_class_accuracy
        values = tuple(metric_function(predictions[seed]) for seed in PAPER_SEEDS)
        subset_values = (
            tuple(subset_mca(predictions[seed], subset_ids) for seed in PAPER_SEEDS)
            if subset_ids is not None
            else (None, None, None)
        )
        coverages = (
            tuple(_coverage(histories[variant][seed]) for seed in PAPER_SEEDS)
            if histories
            else (None, None, None)
        )
        for seed, value, subset_value, coverage in zip(
            PAPER_SEEDS, values, subset_values, coverages, strict=True
        ):
            detail_rows.append((variant, seed, value, subset_value, coverage))
        mean, sd = _mean_sd(values)
        subset_mean, subset_sd = (
            _mean_sd(tuple(cast(float, value) for value in subset_values))
            if subset_ids is not None
            else (None, None)
        )
        coverage_mean, coverage_sd = (
            _mean_sd(tuple(cast(float, value) for value in coverages))
            if histories
            else (None, None)
        )
        summary_rows.append((variant, mean, sd, subset_mean, subset_sd, coverage_mean, coverage_sd))
        latex_rows.append(
            (
                variant,
                RawLatex(f"{mean * 100:.1f} $\\pm$ {sd * 100:.1f}"),
                (
                    RawLatex(f"{subset_mean * 100:.1f} $\\pm$ {subset_sd * 100:.1f}")
                    if subset_mean is not None and subset_sd is not None
                    else "--"
                ),
            )
        )
        sources[variant] = {
            str(seed): {
                "predictions": str(grouped[variant][seed]),
                "predictions_sha256": sha256_file(grouped[variant][seed]),
                "history": str(histories[variant][seed]) if histories else None,
                "history_sha256": sha256_file(histories[variant][seed]) if histories else None,
            }
            for seed in PAPER_SEEDS
        }
    destination = Path(args.output).expanduser().resolve()
    write_csv(
        destination / "sweep_runs.csv",
        ("variant", "seed", "metric_fraction", "subset_mca_fraction", "target_coverage"),
        detail_rows,
    )
    write_csv(
        destination / "sweep_summary.csv",
        (
            "variant",
            "metric_mean",
            "metric_sample_sd",
            "subset_mean",
            "subset_sample_sd",
            "coverage_mean",
            "coverage_sample_sd",
        ),
        summary_rows,
    )
    write_latex_rows(destination / "sweep_rows.tex", latex_rows)
    write_analysis_provenance(
        destination / "provenance.json",
        {
            "schema_version": 1,
            "paper_experiments": ["E06", "E07", "E08", "E09", "E10", "E11", "E12"],
            "dataset": split.dataset,
            "protocol": split.protocol,
            "metric": args.metric,
            "model_seeds": list(PAPER_SEEDS),
            "manifest_sha256": sha256_file(args.manifest),
            "protocol_sha256": sha256_file(args.protocol_split),
            "subset_metadata_sha256": sha256_file(args.subsets) if args.subsets else None,
            "sources": sources,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
