"""Regenerate E02--E05 statistics, diagnostic tables, and figures from E01 predictions."""

from __future__ import annotations

import argparse
import re
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
    write_json_lines,
    write_latex_rows,
    write_matrix,
)
from mcformer.evaluation.diagnostics import (
    classwise_gain,
    pair_diagnostic,
    selected_confusion,
    subset_mca,
)
from mcformer.evaluation.predictions import (
    PredictionArtifactError,
    PredictionSet,
    ensemble_predictions,
    load_predictions,
    validate_paired,
)
from mcformer.evaluation.statistics import (
    paired_bootstrap_gain,
    per_class_accuracy,
    quartiles,
    summarize_runs,
)
from mcformer.reproducibility import sha256_file
from mcformer.visualization.confusion import plot_confusions

PAPER_SEEDS = (17, 29, 43)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not slug:
        raise PredictionArtifactError("Method name cannot produce an artifact filename")
    return slug


def _present_accuracy(value: float | None) -> float:
    if value is None:
        raise PredictionArtifactError("Resolved diagnostic class is absent from predictions")
    return value


def _seed_file(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("Expected SEED=PATH") from error
    if not path_text:
        raise argparse.ArgumentTypeError("Prediction path cannot be empty")
    return seed, Path(path_text).expanduser().resolve()


def _method(values: list[tuple[int, Path]], name: str) -> dict[int, PredictionSet]:
    if len(values) != len(PAPER_SEEDS) or {seed for seed, _ in values} != set(PAPER_SEEDS):
        raise PredictionArtifactError(f"{name} requires exactly seeds {PAPER_SEEDS}")
    if len({seed for seed, _ in values}) != len(values):
        raise PredictionArtifactError(f"{name} contains duplicate seeds")
    return {seed: load_predictions(path) for seed, path in values}


def _data_contract(args: argparse.Namespace) -> tuple[Manifest, ProtocolSplit]:
    manifest = Manifest.read_jsonl(args.manifest)
    split = ProtocolSplit.read_json(args.protocol_split, manifest)
    return manifest, split


def _validate_test_predictions(
    prediction_sets: list[PredictionSet], manifest: Manifest, split: ProtocolSplit
) -> None:
    validate_paired(prediction_sets)
    expected_ids = tuple(sorted(split.test))
    actual_ids = tuple(record.sample_id for record in prediction_sets[0].records)
    if actual_ids != expected_ids:
        raise PredictionArtifactError("Predictions must cover the complete official test partition")
    for record in prediction_sets[0].records:
        if manifest.by_id(record.sample_id).label_id != record.label:
            raise PredictionArtifactError(f"Manifest label mismatch for {record.sample_id}")


def _label_names(manifest: Manifest) -> tuple[str, ...]:
    mapping = {record.label_id: record.label_name for record in manifest}
    return tuple(mapping[index] for index in range(len(mapping)))


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol-split", required=True)
    parser.add_argument("--baseline", action="append", required=True, type=_seed_file)
    parser.add_argument("--method", action="append", required=True, type=_seed_file)
    parser.add_argument("--baseline-name", default="Video Swin")
    parser.add_argument("--method-name", default="MC-Former")
    parser.add_argument("--output", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    statistics = subparsers.add_parser("statistics", help="Generate E02 statistics")
    _common_parser(statistics)
    statistics.add_argument("--metric", choices=("accuracy", "mca"), required=True)
    statistics.add_argument("--bootstrap-resamples", type=int, default=10_000)
    statistics.add_argument("--analysis-seed", type=int, default=2026)
    diagnostics = subparsers.add_parser("diagnostics", help="Generate E03--E05 artifacts")
    _common_parser(diagnostics)
    diagnostics.add_argument("--subsets", required=True)
    diagnostics.add_argument(
        "--task", action="append", choices=("subset", "pairs", "confusion"), required=True
    )
    diagnostics.add_argument("--no-plot", action="store_true")
    return parser


def _load_methods(
    args: argparse.Namespace, manifest: Manifest, split: ProtocolSplit
) -> tuple[dict[int, PredictionSet], dict[int, PredictionSet]]:
    baseline = _method(args.baseline, args.baseline_name)
    method = _method(args.method, args.method_name)
    all_sets = [baseline[seed] for seed in PAPER_SEEDS] + [method[seed] for seed in PAPER_SEEDS]
    _validate_test_predictions(all_sets, manifest, split)
    return baseline, method


def _provenance(
    args: argparse.Namespace,
    split: ProtocolSplit,
    baseline: dict[int, PredictionSet],
    method: dict[int, PredictionSet],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": "E02" if args.command == "statistics" else "derived_diagnostics",
        "dataset": split.dataset,
        "protocol": split.protocol,
        "model_seeds": list(PAPER_SEEDS),
        "manifest_sha256": sha256_file(args.manifest),
        "protocol_sha256": sha256_file(args.protocol_split),
        "methods": {
            args.baseline_name: {
                str(seed): {
                    "path": str(baseline[seed].source),
                    "sha256": sha256_file(baseline[seed].source),
                }
                for seed in PAPER_SEEDS
            },
            args.method_name: {
                str(seed): {
                    "path": str(method[seed].source),
                    "sha256": sha256_file(method[seed].source),
                }
                for seed in PAPER_SEEDS
            },
        },
    }


def _statistics(args: argparse.Namespace) -> int:
    manifest, split = _data_contract(args)
    expected_metric = "mca" if split.dataset == "toyota_smarthome" else "accuracy"
    if args.metric != expected_metric:
        raise PredictionArtifactError(
            f"{split.dataset} paper statistics require metric={expected_metric}"
        )
    baseline, method = _load_methods(args, manifest, split)
    baseline_runs = [baseline[seed] for seed in PAPER_SEEDS]
    method_runs = [method[seed] for seed in PAPER_SEEDS]
    baseline_summary = summarize_runs(baseline_runs, metric=args.metric)
    method_summary = summarize_runs(method_runs, metric=args.metric)
    baseline_ensemble = ensemble_predictions(baseline_runs)
    method_ensemble = ensemble_predictions(method_runs)
    bootstrap = paired_bootstrap_gain(
        baseline_ensemble,
        method_ensemble,
        metric=args.metric,
        resamples=args.bootstrap_resamples,
        seed=args.analysis_seed,
    )
    destination = Path(args.output).expanduser().resolve()
    write_csv(
        destination / "statistics.csv",
        ("method", "seed", "metric", "value_fraction"),
        [
            (name, seed, args.metric, value)
            for name, summary in (
                (args.baseline_name, baseline_summary),
                (args.method_name, method_summary),
            )
            for seed, value in zip(PAPER_SEEDS, summary.values, strict=True)
        ]
        + [
            (args.baseline_name, "mean", args.metric, baseline_summary.mean),
            (
                args.baseline_name,
                "sample_sd",
                args.metric,
                baseline_summary.sample_standard_deviation,
            ),
            (args.method_name, "mean", args.metric, method_summary.mean),
            (args.method_name, "sample_sd", args.metric, method_summary.sample_standard_deviation),
        ],
    )
    write_csv(
        destination / "bootstrap_draws.csv",
        ("draw", "method_minus_baseline_fraction"),
        [(index, value) for index, value in enumerate(bootstrap.draws)],
    )
    write_analysis_provenance(
        destination / "confidence_interval.json",
        {
            "metric": args.metric,
            "point_gain_fraction": bootstrap.point_gain,
            "lower_fraction": bootstrap.lower,
            "upper_fraction": bootstrap.upper,
            "confidence_level": 0.95,
            "estimator": "percentile",
            "paired": True,
            "class_stratified": bootstrap.stratified,
            "resamples": bootstrap.resamples,
            "seed": bootstrap.seed,
        },
    )
    names = _label_names(manifest)
    gains = classwise_gain(baseline_ensemble, method_ensemble)
    baseline_class = per_class_accuracy(baseline_ensemble)
    method_class = per_class_accuracy(method_ensemble)
    write_csv(
        destination / "classwise_gains.csv",
        ("class_id", "class_name", "baseline_accuracy", "method_accuracy", "gain_fraction"),
        [
            (class_id, names[class_id], baseline_class[class_id], method_class[class_id], gain)
            for class_id, gain in enumerate(gains)
        ],
    )
    present_gains = [gain for gain in gains if gain is not None]
    lower_quartile, median, upper_quartile = quartiles(present_gains)
    write_analysis_provenance(
        destination / "classwise_summary.json",
        {
            "positive_classes": sum(gain > 0 for gain in present_gains),
            "evaluated_classes": len(present_gains),
            "median_gain_fraction": median,
            "lower_quartile_fraction": lower_quartile,
            "upper_quartile_fraction": upper_quartile,
        },
    )
    write_latex_rows(
        destination / "statistics_rows.tex",
        [
            (
                split.dataset,
                split.protocol,
                RawLatex(
                    f"{baseline_summary.mean * 100:.1f} $\\pm$ "
                    f"{baseline_summary.sample_standard_deviation * 100:.1f}"
                ),
                RawLatex(
                    f"{method_summary.mean * 100:.1f} $\\pm$ "
                    f"{method_summary.sample_standard_deviation * 100:.1f}"
                ),
                f"[{bootstrap.lower * 100:.1f}, {bootstrap.upper * 100:.1f}]",
            )
        ],
    )
    provenance = _provenance(args, split, baseline, method)
    provenance.update(
        {"analysis_seed": args.analysis_seed, "bootstrap_resamples": args.bootstrap_resamples}
    )
    write_analysis_provenance(destination / "provenance.json", provenance)
    return 0


def _diagnostics(args: argparse.Namespace) -> int:
    manifest, split = _data_contract(args)
    if split.dataset != "toyota_smarthome":
        raise PredictionArtifactError("E03--E05 diagnostics require Toyota Smarthome")
    tasks = set(args.task)
    if len(tasks) != len(args.task):
        raise PredictionArtifactError("Diagnostic tasks cannot be repeated")
    if tasks & {"pairs", "confusion"} and split.protocol.casefold() != "cs":
        raise PredictionArtifactError("E04 pair and E05 confusion diagnostics require Toyota-CS")
    baseline, method = _load_methods(args, manifest, split)
    subsets_path = Path(args.subsets).expanduser().resolve()
    baseline_runs = [baseline[seed] for seed in PAPER_SEEDS]
    method_runs = [method[seed] for seed in PAPER_SEEDS]
    destination = Path(args.output).expanduser().resolve()
    baseline_ensemble = ensemble_predictions(baseline_runs)
    method_ensemble = ensemble_predictions(method_runs)
    resolved_definitions: dict[str, object] = {}
    if "subset" in tasks:
        manipulation_names = load_subset_names(subsets_path, "manipulation_actions")
        manipulation_ids = resolve_label_ids(manifest, manipulation_names)
        resolved_definitions["manipulation_actions"] = [
            {"class_id": class_id, "class_name": name}
            for class_id, name in zip(manipulation_ids, manipulation_names, strict=True)
        ]
        baseline_values = tuple(subset_mca(run, manipulation_ids) for run in baseline_runs)
        method_values = tuple(subset_mca(run, manipulation_ids) for run in method_runs)
        rows: list[tuple[object, ...]] = [
            (name, seed, value)
            for name, values in (
                (args.baseline_name, baseline_values),
                (args.method_name, method_values),
            )
            for seed, value in zip(PAPER_SEEDS, values, strict=True)
        ]
        latex_rows: list[tuple[object, ...]] = []
        for name, values in (
            (args.baseline_name, baseline_values),
            (args.method_name, method_values),
        ):
            mean = sum(values) / len(values)
            sample_sd = (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5
            rows.extend(((name, "mean", mean), (name, "sample_sd", sample_sd)))
            latex_rows.append((name, RawLatex(f"{mean * 100:.1f} $\\pm$ {sample_sd * 100:.1f}")))
        write_csv(destination / "manipulation_subset.csv", ("method", "seed", "mca_fraction"), rows)
        baseline_class = per_class_accuracy(baseline_ensemble)
        method_class = per_class_accuracy(method_ensemble)
        support = {
            class_id: sum(record.label == class_id for record in baseline_ensemble.records)
            for class_id in manipulation_ids
        }
        write_csv(
            destination / "manipulation_per_class.csv",
            (
                "class_id",
                "class_name",
                "support",
                "baseline_accuracy",
                "method_accuracy",
                "gain_fraction",
            ),
            [
                (
                    class_id,
                    name,
                    support[class_id],
                    baseline_class[class_id],
                    method_class[class_id],
                    _present_accuracy(method_class[class_id])
                    - _present_accuracy(baseline_class[class_id]),
                )
                for class_id, name in zip(manipulation_ids, manipulation_names, strict=True)
            ],
        )
        write_latex_rows(destination / "manipulation_subset_rows.tex", latex_rows)
    if "pairs" in tasks:
        pair_rows: list[tuple[object, ...]] = []
        sample_rows: list[dict[str, object]] = []
        for pair_name in ("same_object_pair_1", "same_object_pair_2", "same_object_pair_3"):
            pair_names = load_subset_names(subsets_path, pair_name)
            pair_ids = resolve_label_ids(manifest, pair_names)
            if len(pair_ids) != 2:
                raise PredictionArtifactError(f"{pair_name} must resolve to exactly two classes")
            resolved_definitions[pair_name] = [
                {"class_id": class_id, "class_name": name}
                for class_id, name in zip(pair_ids, pair_names, strict=True)
            ]
            baseline_pair = pair_diagnostic(baseline_ensemble, (pair_ids[0], pair_ids[1]))
            method_pair = pair_diagnostic(method_ensemble, (pair_ids[0], pair_ids[1]))
            pair_rows.append(
                (
                    pair_name,
                    " / ".join(pair_names),
                    baseline_pair.samples,
                    baseline_pair.accuracy,
                    method_pair.accuracy,
                    method_pair.accuracy - baseline_pair.accuracy,
                )
            )
            for model_name, diagnostic in (
                (args.baseline_name, baseline_pair),
                (args.method_name, method_pair),
            ):
                write_csv(
                    destination / f"{pair_name}_{_slug(model_name)}_confusion.csv",
                    ("predicted_first", "predicted_second", "predicted_other"),
                    list(diagnostic.confusion_with_other),
                )
                sample_rows.extend(
                    {"pair": pair_name, "method": model_name, "sample_id": sample_id}
                    for sample_id in diagnostic.sample_ids
                )
        write_csv(
            destination / "same_object_pairs.csv",
            (
                "pair",
                "labels",
                "samples",
                "baseline_accuracy",
                "method_accuracy",
                "gain_fraction",
            ),
            pair_rows,
        )
        write_latex_rows(
            destination / "same_object_pair_rows.tex",
            [
                (
                    row[1],
                    f"{cast(float, row[3]) * 100:.1f}",
                    f"{cast(float, row[4]) * 100:.1f}",
                    f"{cast(float, row[5]) * 100:.1f}",
                )
                for row in pair_rows
            ],
        )
        write_json_lines(destination / "same_object_pair_samples.jsonl", sample_rows)
    if "confusion" in tasks:
        confusion_names = load_subset_names(subsets_path, "confusion_matrix_actions")
        confusion_ids = resolve_label_ids(manifest, confusion_names)
        resolved_definitions["confusion_matrix_actions"] = [
            {"class_id": class_id, "class_name": name}
            for class_id, name in zip(confusion_ids, confusion_names, strict=True)
        ]
        baseline_confusion = selected_confusion(baseline_ensemble, confusion_ids)
        method_confusion = selected_confusion(method_ensemble, confusion_ids)
        write_matrix(destination / "baseline_confusion.csv", baseline_confusion)
        write_matrix(destination / "method_confusion.csv", method_confusion)
        if not args.no_plot:
            plot_confusions(
                [baseline_confusion, method_confusion],
                labels=confusion_names,
                titles=(args.baseline_name, args.method_name),
                output_stem=destination / "confusion_matrix",
            )
    provenance = _provenance(args, split, baseline, method)
    provenance.update(
        {
            "subset_metadata": str(subsets_path),
            "subset_metadata_sha256": sha256_file(subsets_path),
            "confusion_normalization": "all true-class samples; selected prediction columns",
            "seed_ensemble": "arithmetic mean class probabilities",
            "tasks": sorted(tasks),
            "paper_experiments": [
                experiment
                for task, experiment in (
                    ("subset", "E03"),
                    ("pairs", "E04"),
                    ("confusion", "E05"),
                )
                if task in tasks
            ],
            "resolved_label_definitions": resolved_definitions,
        }
    )
    write_analysis_provenance(destination / "provenance.json", provenance)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _statistics(args) if args.command == "statistics" else _diagnostics(args)


if __name__ == "__main__":
    raise SystemExit(main())
