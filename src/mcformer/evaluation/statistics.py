"""Paper-specified repeated-run summaries and paired bootstrap estimators."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from mcformer.evaluation.predictions import PredictionSet, validate_paired


class StatisticsError(ValueError):
    """Raised when a requested estimator is undefined for supplied predictions."""


def accuracy(predictions: PredictionSet, indices: list[int] | None = None) -> float:
    selected = range(len(predictions.records)) if indices is None else indices
    values = [predictions.records[index] for index in selected]
    if not values:
        raise StatisticsError("Accuracy is undefined for an empty selection")
    return sum(record.prediction == record.label for record in values) / len(values)


def mean_class_accuracy(
    predictions: PredictionSet,
    indices: list[int] | None = None,
    *,
    class_ids: tuple[int, ...] | None = None,
) -> float:
    selected = range(len(predictions.records)) if indices is None else indices
    by_class: dict[int, list[bool]] = defaultdict(list)
    allowed = set(class_ids) if class_ids is not None else None
    for index in selected:
        record = predictions.records[index]
        if allowed is None or record.label in allowed:
            by_class[record.label].append(record.prediction == record.label)
    expected = tuple(sorted(by_class)) if class_ids is None else class_ids
    absent = [class_id for class_id in expected if not by_class[class_id]]
    if absent:
        raise StatisticsError(f"Mean class accuracy selection lacks classes: {absent}")
    return sum(sum(by_class[class_id]) / len(by_class[class_id]) for class_id in expected) / len(
        expected
    )


def per_class_accuracy(predictions: PredictionSet) -> tuple[float | None, ...]:
    by_class: dict[int, list[bool]] = defaultdict(list)
    for record in predictions.records:
        by_class[record.label].append(record.prediction == record.label)
    return tuple(
        sum(by_class[class_id]) / len(by_class[class_id]) if by_class[class_id] else None
        for class_id in range(predictions.num_classes)
    )


@dataclass(frozen=True)
class RepeatedRunSummary:
    values: tuple[float, ...]
    mean: float
    sample_standard_deviation: float


def summarize_runs(predictions: list[PredictionSet], *, metric: str) -> RepeatedRunSummary:
    if len(predictions) < 2:
        raise StatisticsError("Sample standard deviation requires at least two runs")
    validate_paired(predictions)
    function = accuracy if metric == "accuracy" else mean_class_accuracy
    if metric not in {"accuracy", "mca"}:
        raise StatisticsError("metric must be accuracy or mca")
    values = tuple(function(run) for run in predictions)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return RepeatedRunSummary(values, mean, math.sqrt(variance))


@dataclass(frozen=True)
class BootstrapResult:
    draws: tuple[float, ...]
    lower: float
    upper: float
    point_gain: float
    resamples: int
    seed: int
    stratified: bool


def paired_bootstrap_gain(
    baseline: PredictionSet,
    method: PredictionSet,
    *,
    metric: str,
    resamples: int = 10_000,
    seed: int = 2026,
) -> BootstrapResult:
    """Bootstrap method-minus-baseline gain with paper-specified pairing/stratification."""

    validate_paired([baseline, method])
    if resamples <= 0:
        raise StatisticsError("resamples must be positive")
    rng = random.Random(seed)
    if metric == "accuracy":
        groups = [list(range(len(baseline.records)))]
        estimator = accuracy
        stratified = False
    elif metric == "mca":
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(baseline.records):
            grouped[record.label].append(index)
        groups = [grouped[class_id] for class_id in sorted(grouped)]
        estimator = mean_class_accuracy
        stratified = True
    else:
        raise StatisticsError("metric must be accuracy or mca")
    draws: list[float] = []
    for _ in range(resamples):
        indices = [rng.choice(group) for group in groups for _ in range(len(group))]
        draws.append(estimator(method, indices) - estimator(baseline, indices))
    point_gain = estimator(method) - estimator(baseline)
    lower, upper = np.quantile(np.asarray(draws), [0.025, 0.975], method="linear")
    return BootstrapResult(
        draws=tuple(draws),
        lower=float(lower),
        upper=float(upper),
        point_gain=point_gain,
        resamples=resamples,
        seed=seed,
        stratified=stratified,
    )


def quartiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise StatisticsError("Quartiles require at least one value")
    lower, median, upper = np.quantile(np.asarray(values), [0.25, 0.5, 0.75])
    return float(lower), float(median), float(upper)
