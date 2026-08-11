"""Hierarchical, validated experiment configuration loading."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class ResolvedConfig:
    """An immutable wrapper around a resolved configuration mapping."""

    source: Path
    values: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a defensive deep copy suitable for serialization."""

        return copy.deepcopy(dict(self.values))

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Read a value using a dotted path such as ``training.epochs``."""

        current: Any = self.values
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML, with a JSON fallback for dependency-light smoke checks.

    JSON is a subset of YAML 1.2. Shipped configuration files use that subset so
    configuration resolution can be smoke-tested before optional dependencies are
    installed. Normal user-authored YAML requires PyYAML.
    """

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as error:
            raise ConfigError(
                f"{path} requires PyYAML because it is not JSON-compatible YAML"
            ) from error
    else:
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return loaded


def _deep_merge(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _load_with_defaults(path: Path, active: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if resolved_path in active:
        chain = " -> ".join(str(item) for item in (*active, resolved_path))
        raise ConfigError(f"Configuration inheritance cycle: {chain}")
    if not resolved_path.is_file():
        raise ConfigError(f"Configuration does not exist: {resolved_path}")

    raw = _read_yaml(resolved_path)
    defaults = raw.pop("defaults", [])
    if not isinstance(defaults, list) or not all(isinstance(item, str) for item in defaults):
        raise ConfigError(f"'defaults' must be a list of paths: {resolved_path}")

    merged: dict[str, Any] = {}
    for default in defaults:
        parent = (resolved_path.parent / default).resolve()
        _deep_merge(merged, _load_with_defaults(parent, (*active, resolved_path)))
    _deep_merge(merged, raw)
    return merged


def _parse_override(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _apply_override(values: MutableMapping[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"Override must have KEY=VALUE form: {override!r}")
    dotted_key, raw_value = override.split("=", 1)
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ConfigError(f"Override key is empty: {override!r}")
    current: MutableMapping[str, Any] = values
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, MutableMapping):
            raise ConfigError(f"Cannot set nested key below non-mapping: {dotted_key!r}")
        current = child
    current[parts[-1]] = _parse_override(raw_value)


def validate_config(values: Mapping[str, Any]) -> None:
    """Validate infrastructure-level fields without constraining future models."""

    required = (
        "schema_version",
        "project",
        "reproducibility",
        "data",
        "model",
        "training",
        "logging",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ConfigError(f"Missing required sections: {', '.join(missing)}")
    if values["schema_version"] != 1:
        raise ConfigError("Only schema_version=1 is supported")
    for section in required[1:]:
        if not isinstance(values[section], Mapping):
            raise ConfigError(f"{section} must be a mapping")

    seeds = values["reproducibility"].get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
    ):
        raise ConfigError("reproducibility.seeds must be a non-empty list of integers")

    training = values["training"]
    for name in ("epochs", "batch_size"):
        setting = training.get(name)
        if not isinstance(setting, int) or isinstance(setting, bool) or setting <= 0:
            raise ConfigError(f"training.{name} must be a positive integer")
    learning_rate = training.get("learning_rate")
    if (
        not isinstance(learning_rate, int | float)
        or isinstance(learning_rate, bool)
        or learning_rate <= 0
    ):
        raise ConfigError("training.learning_rate must be positive")
    for name in ("per_device_batch_size", "num_workers", "warmup_epochs", "log_every_steps"):
        if name == "log_every_steps":
            continue
        setting = training.get(name)
        if not isinstance(setting, int) or isinstance(setting, bool) or setting < 0:
            raise ConfigError(f"training.{name} must be a non-negative integer")
    if training.get("per_device_batch_size", 0) == 0:
        raise ConfigError("training.per_device_batch_size must be positive")
    if training.get("warmup_epochs", 0) >= training["epochs"]:
        raise ConfigError("training.warmup_epochs must be less than training.epochs")
    if not isinstance(training.get("pin_memory"), bool):
        raise ConfigError("training.pin_memory must be a boolean")
    fixed_choices = {
        "optimizer": "adamw",
        "schedule": "cosine",
        "amp_loss_scaling": "dynamic",
        "classification_loss": "cross_entropy",
        "coupling_loss": "masked_mse",
        "learning_rate_scaling": "none",
        "distributed_loss_reduction": "global_valid_positions",
        "gradient_accumulation_policy": "preserve_global_batch_16",
    }
    for name, expected in fixed_choices.items():
        if training.get(name) != expected:
            raise ConfigError(f"training.{name} must be {expected!r}")
    if training.get("early_stopping") is not False or training.get("label_smoothing") != 0.0:
        raise ConfigError("Early stopping and label smoothing are disabled by the release protocol")
    if training.get("mixed_precision") not in {"none", "fp16", "bf16"}:
        raise ConfigError("training.mixed_precision must be none, fp16, or bf16")
    betas = training.get("adam_betas")
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or not all(isinstance(beta, int | float) and 0 <= beta < 1 for beta in betas)
    ):
        raise ConfigError("training.adam_betas must contain two values in [0,1)")
    for name, allow_zero in (
        ("weight_decay", True),
        ("adam_epsilon", False),
        ("gradient_clip_norm", False),
        ("coupling_weight", True),
        ("minimum_learning_rate", True),
    ):
        setting = training.get(name)
        lower_ok = setting >= 0 if allow_zero and isinstance(setting, int | float) else False
        if (
            not isinstance(setting, int | float)
            or isinstance(setting, bool)
            or not (lower_ok or setting > 0)
        ):
            raise ConfigError(f"training.{name} has an invalid numeric value")
    if training["minimum_learning_rate"] > training["learning_rate"]:
        raise ConfigError("training.minimum_learning_rate cannot exceed training.learning_rate")
    log_every = values["logging"].get("log_every_steps")
    if not isinstance(log_every, int) or isinstance(log_every, bool) or log_every <= 0:
        raise ConfigError("logging.log_every_steps must be a positive integer")

    auxiliary_heads = values["model"].get("auxiliary_heads")
    if auxiliary_heads is not None:
        if not isinstance(auxiliary_heads, list) or not auxiliary_heads:
            raise ConfigError("model.auxiliary_heads must be a non-empty list")
        names: list[str] = []
        for head in auxiliary_heads:
            if not isinstance(head, Mapping):
                raise ConfigError("Every auxiliary head must be a mapping")
            head_name, target, kind = head.get("name"), head.get("target"), head.get("kind")
            if not isinstance(head_name, str) or not head_name:
                raise ConfigError("Every auxiliary head requires a non-empty name")
            names.append(head_name)
            expected_kind = "vector" if target == "hallucination" else "temporal"
            if (
                target
                not in {
                    "temporal_gated",
                    "temporal_ungated",
                    "spatial",
                    "hallucination",
                }
                or kind != expected_kind
            ):
                raise ConfigError(f"Auxiliary head {head_name!r} has an invalid target/kind")
            weight = head.get("weight")
            if not isinstance(weight, int | float) or isinstance(weight, bool) or weight < 0:
                raise ConfigError(f"Auxiliary head {head_name!r} weight must be non-negative")
            if kind == "vector" and head.get("output_dim") != 256:
                raise ConfigError("Hallucination head output_dim must be 256")
            if kind == "vector" and values["data"].get("num_frames") != 32:
                raise ConfigError("The fixed 256-D hallucination target requires 32 frames")
            if kind == "temporal" and head.get(
                "output_dim", values["data"].get("num_frames")
            ) != values["data"].get("num_frames"):
                raise ConfigError("Temporal auxiliary output_dim must equal data.num_frames")
        if len(names) != len(set(names)):
            raise ConfigError("Auxiliary head names must be unique")

    device = values["project"].get("device", "auto")
    if device not in {"auto", "cpu", "cuda", "mps"} and not str(device).startswith("cuda:"):
        raise ConfigError(f"Unsupported project.device: {device!r}")


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ResolvedConfig:
    """Load defaults, apply dotted overrides, validate, and return a config."""

    source = Path(path).expanduser().resolve()
    values = _load_with_defaults(source)
    for override in overrides:
        _apply_override(values, override)
    validate_config(values)
    return ResolvedConfig(source=source, values=values)
