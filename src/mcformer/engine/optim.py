"""Frozen AdamW and stepwise warm-up/cosine schedule."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer


class OptimizerConfigurationError(ValueError):
    """Raised when optimizer or schedule settings violate the release contract."""


@dataclass(frozen=True)
class OptimizerSettings:
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    warmup_epochs: int = 5
    minimum_learning_rate: float = 0.0


def build_adamw(parameters: Iterable[Tensor], settings: OptimizerSettings) -> AdamW:
    if settings.learning_rate <= 0 or settings.weight_decay < 0 or settings.epsilon <= 0:
        raise OptimizerConfigurationError("Invalid AdamW learning rate, decay, or epsilon")
    if not all(0 <= beta < 1 for beta in settings.betas):
        raise OptimizerConfigurationError("AdamW betas must lie in [0,1)")
    return AdamW(
        parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        betas=settings.betas,
        eps=settings.epsilon,
    )


class WarmupCosineScheduler:
    """Set the LR for every optimizer update, including zero endpoint."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        base_learning_rate: float,
        minimum_learning_rate: float = 0.0,
    ) -> None:
        if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
            raise OptimizerConfigurationError("Require 0 <= warmup_steps < total_steps")
        if not 0 <= minimum_learning_rate <= base_learning_rate:
            raise OptimizerConfigurationError("minimum LR must lie between zero and base LR")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_learning_rate = base_learning_rate
        self.minimum_learning_rate = minimum_learning_rate
        self.step_index = 0
        self._set_learning_rate(self.learning_rate_at(0))

    def learning_rate_at(self, step: int) -> float:
        if not 0 <= step < self.total_steps:
            raise OptimizerConfigurationError(f"Step {step} is outside the schedule")
        if self.warmup_steps and step < self.warmup_steps:
            return self.base_learning_rate * (step + 1) / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        decay_index = step - self.warmup_steps
        progress = 1.0 if decay_steps == 1 else decay_index / (decay_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            self.minimum_learning_rate
            + (self.base_learning_rate - self.minimum_learning_rate) * cosine
        )

    def _set_learning_rate(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self) -> None:
        if self.step_index >= self.total_steps:
            raise OptimizerConfigurationError("Schedule advanced beyond total_steps")
        self.step_index += 1
        if self.step_index < self.total_steps:
            self._set_learning_rate(self.learning_rate_at(self.step_index))

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_learning_rate": self.base_learning_rate,
            "minimum_learning_rate": self.minimum_learning_rate,
            "step_index": self.step_index,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        expected = {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_learning_rate": self.base_learning_rate,
            "minimum_learning_rate": self.minimum_learning_rate,
        }
        actual = {key: value.get(key) for key in expected}
        if actual != expected:
            raise OptimizerConfigurationError(
                f"Checkpoint schedule does not match current schedule: {actual} != {expected}"
            )
        step_index = int(value["step_index"])
        if not 0 <= step_index <= self.total_steps:
            raise OptimizerConfigurationError("Invalid restored schedule step")
        self.step_index = step_index
        if step_index < self.total_steps:
            self._set_learning_rate(self.learning_rate_at(step_index))
