"""Classifier-input feature extraction and portable feature archives for E16."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from mcformer.engine.data import VideoBatch
from mcformer.engine.distributed import DistributedContext, gather_objects


class FeatureArtifactError(RuntimeError):
    """Raised when feature records are incomplete, duplicated, or incompatible."""


@dataclass(frozen=True)
class FeatureArchive:
    sample_ids: tuple[str, ...]
    labels: tuple[int, ...]
    features: NDArray[np.float32]

    def validate(self) -> None:
        if (
            not self.sample_ids
            or len(self.sample_ids) != len(set(self.sample_ids))
            or len(self.labels) != len(self.sample_ids)
            or self.features.ndim != 2
            or self.features.shape[0] != len(self.sample_ids)
            or not np.isfinite(self.features).all()
        ):
            raise FeatureArtifactError("Feature archive dimensions or values are invalid")


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader[VideoBatch],
    *,
    context: DistributedContext,
) -> FeatureArchive | None:
    """Gather final pre-classifier pooled representations in sample-ID order."""

    model.eval()
    local: list[tuple[str, int, list[float]]] = []
    for batch_cpu in loader:
        batch = batch_cpu.to(context.device, non_blocking=context.device.type == "cuda")
        output = model(batch.videos)
        pooled = getattr(output, "pooled", None)
        if not isinstance(pooled, Tensor) or pooled.ndim != 2:
            raise FeatureArtifactError("Model output does not expose a BxD pooled tensor")
        for index, sample_id in enumerate(batch.sample_ids):
            local.append(
                (
                    sample_id,
                    int(batch.labels[index].item()),
                    [float(value) for value in pooled[index].float().cpu().tolist()],
                )
            )
    gathered = gather_objects(local, context)
    if not context.is_primary:
        return None
    assert gathered is not None
    rows = sorted((row for rank_rows in gathered for row in rank_rows), key=lambda row: row[0])
    archive = FeatureArchive(
        sample_ids=tuple(row[0] for row in rows),
        labels=tuple(row[1] for row in rows),
        features=np.asarray([row[2] for row in rows], dtype=np.float32),
    )
    archive.validate()
    return archive


def write_feature_archive(archive: FeatureArchive, path: str | Path) -> None:
    archive.validate()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        sample_ids=np.asarray(archive.sample_ids),
        labels=np.asarray(archive.labels, dtype=np.int64),
        features=archive.features,
    )


def read_feature_archive(path: str | Path) -> FeatureArchive:
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as value:
            archive = FeatureArchive(
                sample_ids=tuple(str(item) for item in value["sample_ids"].tolist()),
                labels=tuple(int(item) for item in value["labels"].tolist()),
                features=np.asarray(value["features"], dtype=np.float32),
            )
    except (OSError, ValueError, KeyError) as error:
        raise FeatureArtifactError(f"Cannot read feature archive {source}: {error}") from error
    archive.validate()
    return archive


def write_feature_index(archive: FeatureArchive, path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    with destination.open("w", encoding="utf-8") as handle:
        for sample_id, label in zip(archive.sample_ids, archive.labels, strict=True):
            handle.write(json.dumps({"sample_id": sample_id, "label": label}, sort_keys=True))
            handle.write("\n")
