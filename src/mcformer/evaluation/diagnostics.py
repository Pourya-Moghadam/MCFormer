"""Toyota subset, same-object pair, and selected-class confusion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from mcformer.evaluation.predictions import PredictionSet, validate_paired
from mcformer.evaluation.statistics import mean_class_accuracy


@dataclass(frozen=True)
class PairDiagnostic:
    class_ids: tuple[int, int]
    samples: int
    accuracy: float
    confusion_with_other: tuple[tuple[int, int, int], tuple[int, int, int]]
    sample_ids: tuple[str, ...]


def subset_mca(predictions: PredictionSet, class_ids: tuple[int, ...]) -> float:
    """Compute mCA on true labels in a fixed class subset without changing predictions."""

    return mean_class_accuracy(predictions, class_ids=class_ids)


def pair_diagnostic(predictions: PredictionSet, class_ids: tuple[int, int]) -> PairDiagnostic:
    """Score 31-way predictions on a true-label pair and retain out-of-pair errors."""

    if class_ids[0] == class_ids[1]:
        raise ValueError("Pair classes must be distinct")
    rows = [record for record in predictions.records if record.label in class_ids]
    if not rows or {record.label for record in rows} != set(class_ids):
        raise ValueError("Both pair classes must have test samples")
    matrix = [[0, 0, 0], [0, 0, 0]]
    for record in rows:
        true_index = class_ids.index(record.label)
        predicted_index = (
            class_ids.index(record.prediction) if record.prediction in class_ids else 2
        )
        matrix[true_index][predicted_index] += 1
    return PairDiagnostic(
        class_ids=class_ids,
        samples=len(rows),
        accuracy=sum(record.prediction == record.label for record in rows) / len(rows),
        confusion_with_other=(tuple(matrix[0]), tuple(matrix[1])),  # type: ignore[arg-type]
        sample_ids=tuple(record.sample_id for record in rows),
    )


def selected_confusion(
    predictions: PredictionSet, class_ids: tuple[int, ...]
) -> tuple[tuple[float, ...], ...]:
    """Return true-row-normalized selected columns without hiding outside-class errors."""

    if not class_ids or len(class_ids) != len(set(class_ids)):
        raise ValueError("Confusion class IDs must be non-empty and unique")
    matrix: list[tuple[float, ...]] = []
    for true_class in class_ids:
        rows = [record for record in predictions.records if record.label == true_class]
        if not rows:
            raise ValueError(f"Confusion class {true_class} has no test samples")
        matrix.append(
            tuple(
                sum(record.prediction == predicted_class for record in rows) / len(rows)
                for predicted_class in class_ids
            )
        )
    return tuple(matrix)


def classwise_gain(baseline: PredictionSet, method: PredictionSet) -> tuple[float | None, ...]:
    """Return method-minus-baseline per-class accuracy gains after strict pairing."""

    validate_paired([baseline, method])
    gains: list[float | None] = []
    for class_id in range(baseline.num_classes):
        rows = [index for index, row in enumerate(baseline.records) if row.label == class_id]
        if not rows:
            gains.append(None)
            continue
        baseline_accuracy = sum(
            baseline.records[index].prediction == class_id for index in rows
        ) / len(rows)
        method_accuracy = sum(method.records[index].prediction == class_id for index in rows) / len(
            rows
        )
        gains.append(method_accuracy - baseline_accuracy)
    return tuple(gains)
