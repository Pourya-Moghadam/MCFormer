"""Validated controlled-experiment matrix loading for E06--E12."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcformer.auxiliary.cache import configuration_digest
from mcformer.auxiliary.corruption_cache import CorruptionSpec
from mcformer.config import load_config


class ExperimentMatrixError(ValueError):
    """Raised when the frozen controlled-experiment catalog is malformed."""


@dataclass(frozen=True, slots=True)
class SweepVariant:
    experiment: str
    name: str
    config: Path
    overrides: tuple[str, ...]
    config_sha256: str
    cache_corruption: CorruptionSpec | None


@dataclass(frozen=True)
class ExperimentMatrix:
    source: Path
    model_seeds: tuple[int, ...]
    analysis_seed: int
    variants: tuple[SweepVariant, ...]


def load_experiment_matrix(path: str | Path) -> ExperimentMatrix:
    """Load every variant and resolve its base configuration without constructing models."""

    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        seeds = tuple(int(value) for value in raw["model_seeds"])
        analysis_seed = int(raw["analysis_seed"])
        experiments = raw["experiments"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ExperimentMatrixError(f"Cannot read experiment matrix {source}: {error}") from error
    if raw.get("schema_version") != 1 or seeds != (17, 29, 43):
        raise ExperimentMatrixError("Experiment matrix requires schema 1 and seeds 17/29/43")
    if analysis_seed != 2026 or not isinstance(experiments, dict):
        raise ExperimentMatrixError("Experiment matrix requires analysis seed 2026")
    expected_experiments = {f"E{index:02d}" for index in range(6, 13)}
    if set(experiments) != expected_experiments:
        raise ExperimentMatrixError("Experiment matrix must define exactly E06 through E12")
    variants: list[SweepVariant] = []
    identities: set[tuple[str, str]] = set()
    for experiment in sorted(experiments):
        entries = experiments[experiment]
        if not isinstance(entries, list) or not entries:
            raise ExperimentMatrixError(f"{experiment} variants must be a non-empty list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ExperimentMatrixError(f"{experiment} variant must be a mapping")
            name = entry.get("name")
            config_value = entry.get("config")
            overrides = entry.get("overrides")
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(config_value, str)
                or not isinstance(overrides, list)
                or not all(isinstance(value, str) for value in overrides)
            ):
                raise ExperimentMatrixError(f"Malformed {experiment} variant")
            identity = (experiment, name)
            if identity in identities:
                raise ExperimentMatrixError(f"Duplicate variant: {experiment}/{name}")
            identities.add(identity)
            config_path = (source.parent / config_value).resolve()
            effective_overrides = (
                f'experiment.id="{experiment}"',
                f'experiment.name="{name}"',
                *(str(value) for value in overrides),
            )
            resolved = load_config(config_path, effective_overrides)
            corruption: CorruptionSpec | None = None
            cache = entry.get("cache")
            if cache is not None:
                if not isinstance(cache, dict):
                    raise ExperimentMatrixError("cache must be null or a mapping")
                corruption_name, corruption_value = cache.get("corruption"), cache.get("value")
                if not isinstance(corruption_name, str) or not isinstance(
                    corruption_value, int | float
                ):
                    raise ExperimentMatrixError(
                        "cache corruption requires a name and numeric value"
                    )
                corruption = CorruptionSpec(
                    name=corruption_name,
                    value=float(corruption_value),
                    seed=analysis_seed,
                )
                corruption.validate()
            variants.append(
                SweepVariant(
                    experiment=experiment,
                    name=name,
                    config=config_path,
                    overrides=effective_overrides,
                    config_sha256=configuration_digest(resolved.as_dict()),
                    cache_corruption=corruption,
                )
            )
    return ExperimentMatrix(source, seeds, analysis_seed, tuple(variants))


def matrix_as_mapping(matrix: ExperimentMatrix) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": str(matrix.source),
        "model_seeds": list(matrix.model_seeds),
        "analysis_seed": matrix.analysis_seed,
        "variants": [
            {
                "experiment": variant.experiment,
                "name": variant.name,
                "config": str(variant.config),
                "overrides": list(variant.overrides),
                "config_sha256": variant.config_sha256,
                "cache_corruption": (
                    {
                        "name": variant.cache_corruption.name,
                        "value": variant.cache_corruption.value,
                        "seed": variant.cache_corruption.seed,
                    }
                    if variant.cache_corruption is not None
                    else None
                ),
            }
            for variant in matrix.variants
        ],
    }
