"""Content-addressed local checkpoint loading without implicit downloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mcformer.reproducibility import sha256_file


class CheckpointError(RuntimeError):
    """Raised when a local checkpoint is missing, changed, or incompatible."""


@dataclass(frozen=True)
class CheckpointReport:
    path: str
    sha256: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _state_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError("Checkpoint root must be a mapping")
    for key in ("state_dict", "model", "model_state_dict"):
        nested = value.get(key)
        if isinstance(nested, dict):
            value = nested
            break
    if not value or not all(isinstance(key, str) for key in value):
        raise CheckpointError("Checkpoint does not contain a non-empty state dictionary")
    return value


def _align_keys(state: dict[str, Any], expected_keys: set[str]) -> dict[str, Any]:
    """Choose the deterministic wrapper-prefix mapping with greatest exact overlap."""

    removable_prefixes = (
        "module.model.backbone.",
        "module.backbone.",
        "module.model.",
        "backbone.model.",
        "backbone.",
        "module.",
        "model.",
        "",
    )
    candidates: list[dict[str, Any]] = []
    for prefix in removable_prefixes:
        if not prefix or all(key.startswith(prefix) for key in state):
            candidates.append(
                {key[len(prefix) :] if prefix else key: tensor for key, tensor in state.items()}
            )
    candidates.extend(
        {f"model.{key}": tensor for key, tensor in candidate.items()}
        for candidate in list(candidates)
    )
    return max(
        candidates,
        key=lambda candidate: (
            len(set(candidate) & expected_keys),
            -len(set(candidate) - expected_keys),
        ),
    )


def load_local_checkpoint(
    module: nn.Module,
    path: str | Path,
    *,
    expected_sha256: str,
    strict: bool = True,
) -> CheckpointReport:
    """Verify SHA-256 and load a state dictionary into ``module`` on CPU."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {source}")
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256.casefold()
    ):
        raise CheckpointError("expected_sha256 must be a 64-character hexadecimal digest")
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256.casefold():
        raise CheckpointError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )
    try:
        raw = torch.load(source, map_location="cpu", weights_only=True)
        state = _align_keys(_state_dict(raw), set(module.state_dict()))
        incompatible = module.load_state_dict(state, strict=strict)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise CheckpointError(f"Cannot load checkpoint {source}: {error}") from error
    return CheckpointReport(
        path=str(source),
        sha256=actual_sha256,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
