"""Deterministic validation/test inference and auditable prediction artifacts."""

from __future__ import annotations

import contextlib
import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.data import DataLoader

from mcformer.engine.data import VideoBatch
from mcformer.engine.distributed import DistributedContext, gather_objects, reduce_sum
from mcformer.evaluation.metrics import ClassificationAccumulator, ClassificationMetrics
from mcformer.reproducibility import write_json_atomic


class EvaluationError(RuntimeError):
    """Raised when inference cannot produce a complete, unique prediction set."""


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    label: int
    prediction: int
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    valid_frames: int
    padded_frames: int

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "prediction": self.prediction,
            "logits": list(self.logits),
            "probabilities": list(self.probabilities),
            "valid_frames": self.valid_frames,
            "padded_frames": self.padded_frames,
        }


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    metrics: ClassificationMetrics
    predictions: tuple[Prediction, ...] | None


def _logits(output: Any) -> Tensor:
    value = getattr(output, "logits", None)
    if not isinstance(value, Tensor):
        raise EvaluationError("Model output does not contain tensor logits")
    return value


def _autocast(device: torch.device, precision: str) -> Any:
    if precision == "none":
        return contextlib.nullcontext()
    if precision not in {"fp16", "bf16"}:
        raise EvaluationError("mixed_precision must be none, fp16, or bf16")
    if precision == "fp16" and device.type != "cuda":
        raise EvaluationError("FP16 evaluation requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[VideoBatch],
    *,
    context: DistributedContext,
    num_classes: int,
    mixed_precision: str,
) -> EvaluationResult:
    """Evaluate once and gather sample predictions on rank zero."""

    model.eval()
    accumulator = ClassificationAccumulator(num_classes, context.device)
    totals = torch.zeros(2, dtype=torch.float64, device=context.device)
    local: list[dict[str, object]] = []
    for batch_cpu in loader:
        batch = batch_cpu.to(context.device, non_blocking=context.device.type == "cuda")
        with _autocast(context.device, mixed_precision):
            logits = _logits(model(batch.videos))
            loss = functional.cross_entropy(logits, batch.labels)
        accumulator.update(logits, batch.labels)
        probabilities = logits.float().softmax(dim=1)
        predictions = logits.argmax(dim=1)
        for index, sample_id in enumerate(batch.sample_ids):
            local.append(
                Prediction(
                    sample_id=sample_id,
                    label=int(batch.labels[index].item()),
                    prediction=int(predictions[index].item()),
                    logits=tuple(float(value) for value in logits[index].float().cpu().tolist()),
                    probabilities=tuple(
                        float(value) for value in probabilities[index].cpu().tolist()
                    ),
                    valid_frames=int((~batch.padding_mask[index]).sum().item()),
                    padded_frames=int(batch.padding_mask[index].sum().item()),
                ).as_dict()
            )
        totals += torch.tensor(
            [float(loss.detach()) * batch.labels.numel(), batch.labels.numel()],
            dtype=torch.float64,
            device=context.device,
        )
    reduced = reduce_sum(totals, context)
    count = int(reduced[1].item())
    if count == 0:
        raise EvaluationError("Evaluation loader produced zero samples")
    gathered = gather_objects(local, context)
    complete: tuple[Prediction, ...] | None = None
    if context.is_primary:
        assert gathered is not None
        mappings = [item for rank_items in gathered for item in rank_items]
        identifiers = [str(item["sample_id"]) for item in mappings]
        if len(identifiers) != len(set(identifiers)):
            raise EvaluationError("Evaluation produced duplicate sample IDs")
        complete = tuple(
            Prediction(
                sample_id=str(item["sample_id"]),
                label=int(item["label"]),
                prediction=int(item["prediction"]),
                logits=tuple(float(value) for value in item["logits"]),
                probabilities=tuple(float(value) for value in item["probabilities"]),
                valid_frames=int(item["valid_frames"]),
                padded_frames=int(item["padded_frames"]),
            )
            for item in sorted(mappings, key=lambda value: str(value["sample_id"]))
        )
        if len(complete) != count:
            raise EvaluationError("Gathered predictions do not match the reduced sample count")
    return EvaluationResult(
        loss=float(reduced[0].item() / count),
        metrics=accumulator.compute(context),
        predictions=complete,
    )


def write_evaluation_artifacts(result: EvaluationResult, output_dir: str | Path) -> None:
    """Atomically write metrics JSON and canonical sample-sorted JSONL predictions."""

    if result.predictions is None:
        raise EvaluationError("Only the primary process may write evaluation artifacts")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metrics_value = {"loss": result.loss, **result.metrics.as_dict()}
    write_json_atomic(destination / "metrics.json", metrics_value)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination, delete=False
    ) as handle:
        for prediction in result.predictions:
            handle.write(json.dumps(prediction.as_dict(), sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination / "predictions.jsonl")
    with (destination / "per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class_id", "support", "accuracy"))
        for class_id, (support, accuracy) in enumerate(
            zip(result.metrics.class_support, result.metrics.per_class_accuracy, strict=True)
        ):
            writer.writerow((class_id, support, "" if accuracy is None else accuracy))
    with (destination / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(result.metrics.confusion_matrix)
