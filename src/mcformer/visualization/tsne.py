"""Frozen joint PCA+t-SNE projection and figure generation for E16."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mcformer.evaluation.features import FeatureArchive, FeatureArtifactError
from mcformer.reproducibility import write_json_atomic


@dataclass(frozen=True)
class TSNESettings:
    cap_per_class: int = 200
    pca_components: int = 50
    perplexity: float = 30.0
    learning_rate: str = "auto"
    iterations: int = 1000
    seed: int = 2026
    standardize: bool = False


def select_paired_features(
    baseline: FeatureArchive,
    method: FeatureArchive,
    *,
    class_ids: tuple[int, ...],
    cap_per_class: int,
) -> tuple[FeatureArchive, FeatureArchive]:
    baseline.validate()
    method.validate()
    if baseline.sample_ids != method.sample_ids or baseline.labels != method.labels:
        raise FeatureArtifactError(
            "Baseline and method archives must have identical IDs and labels"
        )
    if not class_ids or len(class_ids) != len(set(class_ids)) or cap_per_class <= 0:
        raise FeatureArtifactError("Class IDs must be unique and cap_per_class positive")
    indices: list[int] = []
    labels = np.asarray(baseline.labels)
    for class_id in class_ids:
        candidates = np.flatnonzero(labels == class_id).tolist()
        if not candidates:
            raise FeatureArtifactError(f"Selected class {class_id} has no samples")
        indices.extend(candidates[:cap_per_class])
    indices.sort(key=lambda index: baseline.sample_ids[index])

    def subset(value: FeatureArchive) -> FeatureArchive:
        return FeatureArchive(
            sample_ids=tuple(value.sample_ids[index] for index in indices),
            labels=tuple(value.labels[index] for index in indices),
            features=value.features[indices],
        )

    return subset(baseline), subset(method)


def joint_tsne(
    baseline: FeatureArchive,
    method: FeatureArchive,
    settings: TSNESettings,
) -> NDArray[np.float64]:
    """Return coordinates ordered as all baseline rows followed by all method rows."""

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    combined = np.concatenate((baseline.features, method.features), axis=0)
    if combined.shape[0] <= settings.perplexity:
        raise FeatureArtifactError("t-SNE requires more samples than its perplexity")
    if min(combined.shape) < settings.pca_components:
        raise FeatureArtifactError("Combined feature matrix cannot support the fixed PCA-50 step")
    if settings.standardize:
        combined = StandardScaler().fit_transform(combined)
    reduced = PCA(n_components=settings.pca_components, svd_solver="full").fit_transform(combined)
    coordinates = TSNE(
        n_components=2,
        perplexity=settings.perplexity,
        learning_rate=settings.learning_rate,
        max_iter=settings.iterations,
        init="pca",
        metric="euclidean",
        random_state=settings.seed,
    ).fit_transform(reduced)
    return np.asarray(coordinates, dtype=np.float64)


def write_tsne_artifacts(
    baseline: FeatureArchive,
    method: FeatureArchive,
    coordinates: NDArray[np.float64],
    *,
    class_names: dict[int, str],
    settings: TSNESettings,
    output_dir: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    count = len(baseline.sample_ids)
    if method.sample_ids != baseline.sample_ids or coordinates.shape != (2 * count, 2):
        raise FeatureArtifactError("Coordinate rows do not match paired feature archives")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Any, ...]] = []
    pairs = (("Video Swin", baseline), ("MC-Former", method))
    for model_index, (name, archive) in enumerate(pairs):
        for index, (sample_id, label) in enumerate(
            zip(archive.sample_ids, archive.labels, strict=True)
        ):
            coordinate = coordinates[model_index * count + index]
            rows.append((sample_id, label, class_names[label], name, coordinate[0], coordinate[1]))
    with (destination / "coordinates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "label", "class_name", "model", "x", "y"))
        writer.writerows(rows)
    np.savez_compressed(
        destination / "selected_features.npz",
        sample_ids=np.asarray(baseline.sample_ids),
        labels=np.asarray(baseline.labels),
        baseline=baseline.features,
        mcformer=method.features,
    )
    write_json_atomic(
        destination / "projection.json",
        {
            "schema_version": 1,
            **settings.__dict__,
            "projection": "joint PCA followed by joint t-SNE",
            "distance": "euclidean",
            "row_order": "all Video Swin rows, then all MC-Former rows",
            "class_names": {str(key): value for key, value in sorted(class_names.items())},
        },
    )
    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    class_ids = sorted(class_names)
    for model_index, (model_name, marker) in enumerate((("Video Swin", "o"), ("MC-Former", "^"))):
        offset = model_index * count
        for color_index, class_id in enumerate(class_ids):
            selected = np.asarray(baseline.labels) == class_id
            axis.scatter(
                coordinates[offset : offset + count][selected, 0],
                coordinates[offset : offset + count][selected, 1],
                s=18,
                marker=marker,
                color=cmap(color_index % 20),
                alpha=0.72,
                label=f"{class_names[class_id]} — {model_name}",
            )
    axis.set(xticks=[], yticks=[], xlabel="t-SNE 1", ylabel="t-SNE 2")
    axis.legend(fontsize=6, ncol=2, frameon=False)
    figure.savefig(destination / "tsne.png", dpi=300)
    figure.savefig(destination / "tsne.svg")
    plt.close(figure)
