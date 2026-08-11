"""Classification and globally reduced masked coupling losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mcformer.models.classifier import AuxiliaryFormerOutput, MCFormerOutput


def _distributed_sum(value: Tensor) -> Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    from torch.distributed.nn.functional import all_reduce

    return cast(
        Tensor,
        all_reduce(value, op=torch.distributed.ReduceOp.SUM),  # type: ignore[no-untyped-call]
    )


def masked_mse_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Return global masked MSE and global valid count, with differentiable empty loss."""

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    weights = mask.to(dtype=prediction.dtype)
    local_squared_error = ((prediction - target) ** 2 * weights).sum()
    local_count = weights.sum()
    squared_error = _distributed_sum(local_squared_error)
    count = _distributed_sum(local_count)
    if count.detach().item() == 0:
        return prediction.sum() * 0.0, count
    return squared_error / (count + epsilon), count


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    classification: Tensor
    coupling: Tensor
    valid_coupling_positions: Tensor


class MCFormerLoss(nn.Module):
    """Cross-entropy plus weighted gated coupling regression."""

    def __init__(self, coupling_weight: float = 1.0, epsilon: float = 1e-6) -> None:
        super().__init__()
        if coupling_weight < 0:
            raise ValueError("coupling_weight must be non-negative")
        self.coupling_weight = coupling_weight
        self.epsilon = epsilon

    def forward(
        self,
        output: MCFormerOutput,
        labels: Tensor,
        coupling_target: Tensor,
        coupling_mask: Tensor,
    ) -> LossOutput:
        classification = functional.cross_entropy(output.logits, labels)
        coupling, count = masked_mse_loss(
            output.coupling,
            coupling_target,
            coupling_mask,
            epsilon=self.epsilon,
        )
        return LossOutput(
            total=classification + self.coupling_weight * coupling,
            classification=classification,
            coupling=coupling,
            valid_coupling_positions=count,
        )


@dataclass(frozen=True)
class AuxiliaryLossOutput:
    total: Tensor
    classification: Tensor
    auxiliary: Tensor
    valid_positions: Tensor
    possible_positions: Tensor
    per_head: dict[str, Tensor]


def auxiliary_former_loss(
    output: AuxiliaryFormerOutput,
    labels: Tensor,
    targets: dict[str, Tensor],
    masks: dict[str, Tensor],
) -> AuxiliaryLossOutput:
    """Combine CE with each configured globally masked auxiliary MSE."""

    classification = functional.cross_entropy(output.logits, labels)
    auxiliary = classification.detach() * 0.0
    valid = torch.zeros((), dtype=classification.dtype, device=classification.device)
    possible = torch.zeros((), dtype=classification.dtype, device=classification.device)
    per_head: dict[str, Tensor] = {}
    for name, prediction in output.auxiliary_predictions.items():
        target_name = output.auxiliary_targets[name]
        if target_name not in targets or target_name not in masks:
            raise ValueError(f"Batch lacks auxiliary target {target_name!r} for head {name!r}")
        loss, _ = masked_mse_loss(prediction, targets[target_name], masks[target_name])
        per_head[name] = loss
        auxiliary = auxiliary + output.auxiliary_weights[name] * loss
        valid = valid + masks[target_name].sum()
        possible = possible + masks[target_name].numel()
    return AuxiliaryLossOutput(
        total=classification + auxiliary,
        classification=classification,
        auxiliary=auxiliary,
        valid_positions=valid,
        possible_positions=possible,
        per_head=per_head,
    )
