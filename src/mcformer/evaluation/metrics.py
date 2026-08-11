"""Exact action-recognition metrics with absent-class reporting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mcformer.engine.distributed import DistributedContext, reduce_sum


@dataclass(frozen=True)
class ClassificationMetrics:
    samples: int
    top1_accuracy: float
    top5_accuracy: float
    mean_class_accuracy: float
    per_class_accuracy: tuple[float | None, ...]
    class_support: tuple[int, ...]
    absent_classes: tuple[int, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "top1_accuracy": self.top1_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "mean_class_accuracy": self.mean_class_accuracy,
            "per_class_accuracy": list(self.per_class_accuracy),
            "class_support": list(self.class_support),
            "absent_classes": list(self.absent_classes),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


class ClassificationAccumulator:
    """Accumulate top-k and per-class counts without retaining model outputs."""

    def __init__(self, num_classes: int, device: torch.device) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.num_classes = num_classes
        self.device = device
        self.counts = torch.zeros(num_classes, 2, dtype=torch.long, device=device)
        self.top1 = torch.zeros((), dtype=torch.long, device=device)
        self.top5 = torch.zeros((), dtype=torch.long, device=device)
        self.samples = torch.zeros((), dtype=torch.long, device=device)
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    def update(self, logits: Tensor, labels: Tensor) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError("Logits have the wrong class dimension")
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("Labels must align with logits")
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
            raise ValueError("Labels are outside the configured class range")
        prediction = logits.argmax(dim=1)
        correct = prediction.eq(labels)
        top_k = min(5, self.num_classes)
        top5_correct = logits.topk(top_k, dim=1).indices.eq(labels[:, None]).any(dim=1)
        self.top1 += correct.sum()
        self.top5 += top5_correct.sum()
        self.samples += labels.numel()
        self.counts[:, 0] += torch.bincount(labels, minlength=self.num_classes)
        self.counts[:, 1] += torch.bincount(labels[correct], minlength=self.num_classes)
        flat_confusion = labels * self.num_classes + prediction
        self.confusion += torch.bincount(flat_confusion, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self, context: DistributedContext) -> ClassificationMetrics:
        samples = reduce_sum(self.samples, context).cpu()
        top1 = reduce_sum(self.top1, context).cpu()
        top5 = reduce_sum(self.top5, context).cpu()
        counts = reduce_sum(self.counts, context).cpu()
        confusion = reduce_sum(self.confusion, context).cpu()
        sample_count = int(samples.item())
        if sample_count == 0:
            raise ValueError("Cannot compute metrics for zero samples")
        support = counts[:, 0]
        correct = counts[:, 1]
        valid = support > 0
        per_class: list[float | None] = []
        for index in range(self.num_classes):
            per_class.append(float(correct[index] / support[index]) if bool(valid[index]) else None)
        mean_class = float((correct[valid].float() / support[valid].float()).mean())
        return ClassificationMetrics(
            samples=sample_count,
            top1_accuracy=float(top1 / samples),
            top5_accuracy=float(top5 / samples),
            mean_class_accuracy=mean_class,
            per_class_accuracy=tuple(per_class),
            class_support=tuple(int(value) for value in support.tolist()),
            absent_classes=tuple(
                index for index, value in enumerate(support.tolist()) if value == 0
            ),
            confusion_matrix=tuple(
                tuple(int(value) for value in row) for row in confusion.tolist()
            ),
        )
