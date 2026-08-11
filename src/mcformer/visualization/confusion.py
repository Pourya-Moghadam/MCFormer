"""Deterministic selected-class confusion-matrix rendering."""

from __future__ import annotations

from pathlib import Path


def plot_confusions(
    matrices: list[tuple[tuple[float, ...], ...]],
    *,
    labels: tuple[str, ...],
    titles: tuple[str, ...],
    output_stem: str | Path,
) -> None:
    """Render matrices with shared [0,1] scale to both PNG and vector PDF."""

    if len(matrices) != len(titles) or not matrices:
        raise ValueError("Every confusion matrix requires one title")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised in analysis environment
        raise RuntimeError("Confusion plotting requires the analysis dependencies") from error
    figure, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.5), squeeze=False)
    image = None
    for axis, matrix, title in zip(axes[0], matrices, titles, strict=True):
        values = np.asarray(matrix)
        if values.shape != (len(labels), len(labels)):
            raise ValueError("Confusion matrix dimensions do not match labels")
        image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = values[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.5 else "black",
                )
    assert image is not None
    figure.colorbar(image, ax=list(axes[0]), label="Fraction of true-class samples")
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
