"""Deterministic E15 sample/frame selection and attention overlay rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mcformer.reproducibility import write_json_atomic


class QualitativeFigureError(ValueError):
    """Raised when an E15 selection or visualization input is invalid."""


def select_qualitative_frames(target: dict[str, Any]) -> tuple[int, int, int, int]:
    """Select first gate, first positive coupling, maximum coupling, and last gate."""

    coupling = target.get("coupling")
    if not isinstance(coupling, dict):
        raise QualitativeFigureError("Cached sample target lacks coupling data")
    values = np.asarray(coupling.get("target"), dtype=np.float64)
    gate = np.asarray(coupling.get("gate"), dtype=bool)
    if values.ndim != 1 or gate.shape != values.shape or not np.isfinite(values).all():
        raise QualitativeFigureError("Coupling target/gate must be aligned finite vectors")
    gated = np.flatnonzero(gate)
    positive = np.flatnonzero((values > 0) & gate)
    if gated.size == 0 or positive.size == 0:
        raise QualitativeFigureError("Selected sample has no gated positive-coupling frame")
    maximum = int(gated[np.argmax(values[gated])])
    return int(gated[0]), int(positive[0]), maximum, int(gated[-1])


def rgb_from_normalized(video: Any) -> NDArray[np.float32]:
    value = np.asarray(video, dtype=np.float32)
    if value.ndim != 4 or value.shape[1] != 3:
        raise QualitativeFigureError("Expected normalized T,C,H,W video")
    value = value.transpose(0, 2, 3, 1)
    value = value * np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    value = value + np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    return np.clip(value, 0.0, 1.0)


def render_qualitative_figure(
    video: NDArray[np.float32],
    activations: NDArray[np.float32],
    selected: tuple[int, int, int, int],
    *,
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    if activations.shape != video.shape[:3] or not np.isfinite(activations).all():
        raise QualitativeFigureError("Activation maps must align with T,H,W RGB frames")
    destination = Path(output_dir).expanduser().resolve()
    panels = destination / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    names = ("first_gated", "first_positive", "maximum_coupling", "last_gated")
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    for axis, name, index in zip(axes, names, selected, strict=True):
        axis.imshow(video[index])
        axis.imshow(activations[index], cmap="viridis", alpha=0.45, vmin=0.0, vmax=1.0)
        axis.set_title(name.replace("_", " "))
        axis.axis("off")
        panel, panel_axis = plt.subplots(figsize=(4, 4), constrained_layout=True)
        panel_axis.imshow(video[index])
        panel_axis.imshow(activations[index], cmap="viridis", alpha=0.45, vmin=0.0, vmax=1.0)
        panel_axis.axis("off")
        panel.savefig(panels / f"{name}.png", dpi=300)
        plt.close(panel)
    figure.savefig(destination / "interaction_sequence.png", dpi=300)
    figure.savefig(destination / "interaction_sequence.pdf")
    plt.close(figure)
    np.savez_compressed(
        destination / "activation_arrays.npz",
        activation=activations.astype(np.float32),
        selected_positions=np.asarray(selected, dtype=np.int64),
    )
    write_json_atomic(destination / "provenance.json", metadata)
    (destination / "trajectories.json").write_text(
        json.dumps(metadata["trajectories"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
