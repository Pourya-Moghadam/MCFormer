"""Strict loading, pairing, and seed ensembling of evaluation prediction artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


class PredictionArtifactError(ValueError):
    """Raised when saved predictions are incomplete, inconsistent, or unpaired."""


@dataclass(frozen=True, slots=True)
class SavedPrediction:
    sample_id: str
    label: int
    prediction: int
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class PredictionSet:
    source: Path
    records: tuple[SavedPrediction, ...]
    num_classes: int

    @property
    def by_id(self) -> dict[str, SavedPrediction]:
        return {record.sample_id: record for record in self.records}


def _finite_vector(value: object, *, field: str, line_number: int) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise PredictionArtifactError(f"Line {line_number}: {field} must be a non-empty list")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise PredictionArtifactError(f"Line {line_number}: {field} contains non-finite values")
    return vector


def load_predictions(path: str | Path) -> PredictionSet:
    """Load evaluator JSONL and reject malformed or internally inconsistent rows."""

    source = Path(path).expanduser().resolve()
    records: list[SavedPrediction] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PredictionArtifactError(f"Cannot read predictions {source}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            sample_id = raw["sample_id"]
            label = raw["label"]
            prediction = raw["prediction"]
            logits = _finite_vector(raw["logits"], field="logits", line_number=line_number)
            probabilities = _finite_vector(
                raw["probabilities"], field="probabilities", line_number=line_number
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise PredictionArtifactError(
                f"Invalid prediction line {line_number}: {error}"
            ) from error
        if not isinstance(sample_id, str) or not sample_id:
            raise PredictionArtifactError(f"Line {line_number}: invalid sample_id")
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or not isinstance(prediction, int)
            or isinstance(prediction, bool)
        ):
            raise PredictionArtifactError(f"Line {line_number}: label/prediction must be integers")
        if len(logits) != len(probabilities) or not 0 <= label < len(logits):
            raise PredictionArtifactError(f"Line {line_number}: class dimensions are inconsistent")
        probability_prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
        if prediction != probability_prediction or not 0 <= prediction < len(logits):
            raise PredictionArtifactError(
                f"Line {line_number}: prediction is not probability argmax"
            )
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise PredictionArtifactError(f"Line {line_number}: probabilities do not sum to one")
        records.append(SavedPrediction(sample_id, label, prediction, logits, probabilities))
    if not records:
        raise PredictionArtifactError(f"Prediction file is empty: {source}")
    identifiers = [record.sample_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise PredictionArtifactError("Prediction file contains duplicate sample IDs")
    if identifiers != sorted(identifiers):
        raise PredictionArtifactError("Prediction rows must be sorted by sample_id")
    dimensions = {len(record.probabilities) for record in records}
    if len(dimensions) != 1:
        raise PredictionArtifactError("Prediction rows have inconsistent class dimensions")
    return PredictionSet(source, tuple(records), dimensions.pop())


def validate_paired(sets: list[PredictionSet]) -> tuple[str, ...]:
    """Require identical sample IDs, labels, and class dimensions across prediction sets."""

    if not sets:
        raise PredictionArtifactError("At least one prediction set is required")
    reference = sets[0]
    identifiers = tuple(record.sample_id for record in reference.records)
    labels = tuple(record.label for record in reference.records)
    for candidate in sets[1:]:
        if candidate.num_classes != reference.num_classes:
            raise PredictionArtifactError("Paired predictions have different class dimensions")
        if tuple(record.sample_id for record in candidate.records) != identifiers:
            raise PredictionArtifactError("Paired predictions have different sample IDs or order")
        if tuple(record.label for record in candidate.records) != labels:
            raise PredictionArtifactError("Paired predictions disagree on labels")
    return identifiers


def ensemble_predictions(sets: list[PredictionSet]) -> PredictionSet:
    """Average class probabilities across paired seeds and recompute multiclass predictions."""

    validate_paired(sets)
    records: list[SavedPrediction] = []
    for row_index, reference in enumerate(sets[0].records):
        probabilities = tuple(
            sum(candidate.records[row_index].probabilities[class_id] for candidate in sets)
            / len(sets)
            for class_id in range(sets[0].num_classes)
        )
        prediction = max(range(sets[0].num_classes), key=probabilities.__getitem__)
        records.append(
            SavedPrediction(
                sample_id=reference.sample_id,
                label=reference.label,
                prediction=prediction,
                logits=(),
                probabilities=probabilities,
            )
        )
    return PredictionSet(Path("<seed-mean ensemble>"), tuple(records), sets[0].num_classes)
