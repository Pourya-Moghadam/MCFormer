"""Atomic, resumable training checkpoints with RNG and configuration identity."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.optim import Optimizer

from mcformer.engine.optim import WarmupCosineScheduler
from mcformer.reproducibility import sha256_file


class TrainingCheckpointError(RuntimeError):
    """Raised when a checkpoint is changed, incompatible, or incomplete."""


@dataclass(frozen=True)
class ResumeState:
    next_epoch: int
    best_metric: float | None
    best_epoch: int | None
    sha256: str


def unwrap_model(model: nn.Module) -> nn.Module:
    return cast(nn.Module, model.module) if hasattr(model, "module") else model


def capture_rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        value["numpy"] = np.random.get_state()
    return value


def restore_rng_state(value: dict[str, Any]) -> None:
    try:
        random.setstate(value["python"])
        torch.set_rng_state(value["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in value:
            torch.cuda.set_rng_state_all(value["torch_cuda"])
        if "numpy" in value:
            import numpy as np

            np.random.set_state(value["numpy"])
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingCheckpointError(f"Invalid checkpoint RNG state: {error}") from error


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: Any,
    epoch_completed: int,
    best_metric: float | None,
    best_epoch: int | None,
    config_sha256: str,
    seed: int,
    rng_states: list[dict[str, Any]] | None = None,
) -> str:
    """Atomically serialize complete resume state and return its SHA-256."""

    if epoch_completed < 0 or len(config_sha256) != 64:
        raise TrainingCheckpointError("Invalid epoch or configuration digest")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "epoch_completed": epoch_completed,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "config_sha256": config_sha256,
        "seed": seed,
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_states": rng_states if rng_states is not None else [capture_rng_state()],
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(destination)


def read_checkpoint(
    path: str | Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise TrainingCheckpointError(f"Checkpoint does not exist: {source}")
    digest = sha256_file(source)
    if expected_sha256 is not None and digest != expected_sha256.casefold():
        raise TrainingCheckpointError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, found {digest}"
        )
    try:
        value = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise TrainingCheckpointError(f"Cannot read checkpoint {source}: {error}") from error
    if not isinstance(value, dict):
        raise TrainingCheckpointError("Checkpoint root must be a mapping")
    return value, digest


def load_inference_state(
    model: nn.Module,
    value: dict[str, Any],
    *,
    config_sha256: str,
    seed: int,
) -> nn.Module:
    """Load a full training checkpoint or an exported RGB-only checkpoint.

    The configuration digest and model seed are mandatory identity checks.  An
    RGB-only export is loaded into, and returns, the deployable classifier graph;
    the training-only auxiliary module is therefore physically absent.
    """

    from mcformer.models.classifier import AuxiliaryFormer, MCFormer

    if value.get("config_sha256") != config_sha256 or value.get("seed") != seed:
        raise TrainingCheckpointError("Checkpoint config digest or seed does not match inference")
    if value.get("model_type") == "rgb_only":
        if not isinstance(model, MCFormer | AuxiliaryFormer):
            raise TrainingCheckpointError(
                "RGB-only checkpoint requires an auxiliary-model configuration"
            )
        state = value.get("state_dict")
        if not isinstance(state, dict):
            raise TrainingCheckpointError("RGB-only checkpoint lacks a state dictionary")
        model.rgb_model.load_state_dict(state, strict=True)
        return model.rgb_model
    if value.get("schema_version") != 1:
        raise TrainingCheckpointError("Unsupported inference checkpoint schema")
    state = value.get("model_state")
    if not isinstance(state, dict):
        raise TrainingCheckpointError("Training checkpoint lacks model_state")
    model.load_state_dict(state, strict=True)
    return model


def resume_training_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: Any,
    config_sha256: str,
    seed: int,
    rank: int = 0,
) -> ResumeState:
    """Restore all mutable training state after strict identity validation."""

    value, digest = read_checkpoint(path, expected_sha256=expected_sha256)
    if value.get("schema_version") != 1:
        raise TrainingCheckpointError("Unsupported checkpoint schema")
    if value.get("config_sha256") != config_sha256 or value.get("seed") != seed:
        raise TrainingCheckpointError("Checkpoint config digest or seed does not match this run")
    try:
        unwrap_model(model).load_state_dict(value["model_state"], strict=True)
        optimizer.load_state_dict(value["optimizer_state"])
        scheduler.load_state_dict(value["scheduler_state"])
        saved_scaler = value.get("scaler_state")
        if scaler is not None and saved_scaler is not None:
            scaler.load_state_dict(saved_scaler)
        elif (scaler is None) != (saved_scaler is None):
            raise TrainingCheckpointError("Checkpoint AMP scaler mode does not match")
        rng_states = value["rng_states"]
        if not isinstance(rng_states, list) or not 0 <= rank < len(rng_states):
            raise TrainingCheckpointError("Checkpoint has no RNG state for this rank")
        restore_rng_state(rng_states[rank])
        epoch_completed = int(value["epoch_completed"])
        best_metric = value.get("best_metric")
        best_epoch = value.get("best_epoch")
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, TrainingCheckpointError):
            raise
        raise TrainingCheckpointError(f"Cannot restore training checkpoint: {error}") from error
    return ResumeState(
        next_epoch=epoch_completed + 1,
        best_metric=float(best_metric) if best_metric is not None else None,
        best_epoch=int(best_epoch) if best_epoch is not None else None,
        sha256=digest,
    )
