"""Deterministic single- or multi-GPU training and validation loops."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from mcformer.data.dataset import VideoClipDataset
from mcformer.engine.checkpointing import capture_rng_state, save_training_checkpoint, unwrap_model
from mcformer.engine.data import VideoBatch
from mcformer.engine.distributed import (
    DistributedContext,
    barrier,
    gather_objects,
    reduce_max,
    reduce_sum,
)
from mcformer.engine.losses import MCFormerLoss, auxiliary_former_loss
from mcformer.engine.optim import WarmupCosineScheduler
from mcformer.evaluation.metrics import ClassificationAccumulator, ClassificationMetrics
from mcformer.models.classifier import (
    AuxiliaryFormer,
    AuxiliaryFormerOutput,
    MCFormer,
    MCFormerOutput,
    export_rgb_checkpoint,
)
from mcformer.reproducibility import sha256_file, write_json_atomic


class TrainingError(RuntimeError):
    """Raised when a batch or runtime setting cannot produce a valid training run."""


class EpochSampler(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@dataclass(frozen=True)
class TrainerSettings:
    epochs: int
    accumulation_steps: int
    gradient_clip_norm: float
    mixed_precision: str
    coupling_weight: float
    log_every_steps: int
    primary_metric: str

    def validate(self, device: torch.device) -> None:
        if self.epochs <= 0 or self.accumulation_steps <= 0 or self.log_every_steps <= 0:
            raise TrainingError("Epoch, accumulation, and logging settings must be positive")
        if self.gradient_clip_norm <= 0 or self.coupling_weight < 0:
            raise TrainingError("Invalid gradient clipping or coupling weight")
        if self.mixed_precision not in {"none", "fp16", "bf16"}:
            raise TrainingError("mixed_precision must be none, fp16, or bf16")
        if self.mixed_precision == "fp16" and device.type != "cuda":
            raise TrainingError("FP16 AMP requires CUDA; use mixed_precision=none on CPU")
        if self.primary_metric not in {"top1_accuracy", "mean_class_accuracy"}:
            raise TrainingError("Unsupported primary validation metric")


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    training_loss: float
    classification_loss: float
    coupling_loss: float
    coupling_valid_positions: int
    coupling_coverage: float
    auxiliary_head_losses: dict[str, float]
    learning_rate: float
    gradient_norm: float
    train_metrics: ClassificationMetrics
    validation_loss: float
    validation_metrics: ClassificationMetrics
    training_seconds: float
    peak_training_memory_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "training_loss": self.training_loss,
            "classification_loss": self.classification_loss,
            "coupling_loss": self.coupling_loss,
            "auxiliary_loss": self.coupling_loss,
            "coupling_valid_positions": self.coupling_valid_positions,
            "coupling_coverage": self.coupling_coverage,
            "auxiliary_head_losses": self.auxiliary_head_losses,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "train_metrics": self.train_metrics.as_dict(),
            "validation_loss": self.validation_loss,
            "validation_metrics": self.validation_metrics.as_dict(),
            "training_seconds": self.training_seconds,
            "peak_training_memory_bytes": self.peak_training_memory_bytes,
        }


def _autocast(device: torch.device, precision: str) -> Any:
    if precision == "none":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def create_grad_scaler(device: torch.device, precision: str) -> Any:
    if precision != "fp16":
        return None
    return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def _logits(output: Any) -> Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TrainingError("Model output does not contain tensor logits")
    return logits


class Trainer:
    """Own model optimization while keeping data preparation and artifacts explicit."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: WarmupCosineScheduler,
        settings: TrainerSettings,
        context: DistributedContext,
        num_classes: int,
        logger: logging.Logger,
        scaler: Any = None,
    ) -> None:
        settings.validate(context.device)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.settings = settings
        self.context = context
        self.num_classes = num_classes
        self.logger = logger
        self.scaler = scaler
        self.mc_loss = MCFormerLoss(coupling_weight=settings.coupling_weight)

    def _loss(
        self, output: Any, batch: VideoBatch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if isinstance(output, AuxiliaryFormerOutput):
            losses = auxiliary_former_loss(
                output,
                batch.labels,
                batch.auxiliary_targets,
                batch.auxiliary_masks,
            )
            return (
                losses.total,
                losses.classification,
                losses.auxiliary,
                losses.valid_positions,
                losses.possible_positions,
                losses.per_head,
            )
        if isinstance(output, MCFormerOutput):
            if batch.coupling_target is None or batch.coupling_mask is None:
                raise TrainingError("MC-Former training requires coupling targets for every sample")
            mc_losses = self.mc_loss(
                output,
                batch.labels,
                batch.coupling_target,
                batch.coupling_mask,
            )
            return (
                mc_losses.total,
                mc_losses.classification,
                mc_losses.coupling,
                batch.coupling_mask.sum(),
                torch.tensor(batch.coupling_mask.numel(), device=batch.labels.device),
                {"mcim": mc_losses.coupling},
            )
        classification = functional.cross_entropy(_logits(output), batch.labels)
        return (
            classification,
            classification,
            classification.detach() * 0.0,
            torch.zeros((), device=batch.labels.device),
            torch.tensor(batch.frame_indices.numel(), device=batch.labels.device),
            {},
        )

    def train_epoch(
        self, loader: DataLoader[VideoBatch], epoch: int
    ) -> tuple[
        float,
        float,
        float,
        int,
        float,
        dict[str, float],
        float,
        float,
        ClassificationMetrics,
    ]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = ClassificationAccumulator(self.num_classes, self.context.device)
        totals = torch.zeros(6, dtype=torch.float64, device=self.context.device)
        head_totals: dict[str, Tensor] = {}
        maximum_gradient_norm = 0.0
        last_learning_rate = self.scheduler.learning_rate
        loader_length = len(loader)
        for step, batch_cpu in enumerate(loader):
            batch = batch_cpu.to(
                self.context.device, non_blocking=self.context.device.type == "cuda"
            )
            group_start = step - step % self.settings.accumulation_steps
            group_size = min(self.settings.accumulation_steps, loader_length - group_start)
            update = (step + 1) % self.settings.accumulation_steps == 0 or step + 1 == loader_length
            synchronization = (
                self.model.no_sync()
                if isinstance(self.model, DistributedDataParallel) and not update
                else contextlib.nullcontext()
            )
            with synchronization, _autocast(self.context.device, self.settings.mixed_precision):
                output = self.model(batch.videos)
                (
                    total,
                    classification,
                    coupling,
                    valid_coupling,
                    possible_coupling,
                    head_losses,
                ) = self._loss(output, batch)
                backward_loss = total / group_size
            if self.scaler is None:
                backward_loss.backward()  # type: ignore[no-untyped-call]
            else:
                self.scaler.scale(backward_loss).backward()
            metrics.update(_logits(output).detach(), batch.labels)
            batch_size = batch.labels.numel()
            totals += torch.tensor(
                [
                    float(total.detach()) * batch_size,
                    float(classification.detach()) * batch_size,
                    float(coupling.detach()) * batch_size,
                    batch_size,
                    float(valid_coupling.detach()),
                    float(possible_coupling.detach()),
                ],
                dtype=torch.float64,
                device=self.context.device,
            )
            if set(head_totals) not in (set(), set(head_losses)):
                raise TrainingError("Auxiliary head set changed between training batches")
            for name, value in head_losses.items():
                head_totals[name] = (
                    head_totals.get(
                        name, torch.zeros((), dtype=torch.float64, device=self.context.device)
                    )
                    + value.detach().to(torch.float64) * batch_size
                )
            if update:
                last_learning_rate = self.scheduler.learning_rate
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.settings.gradient_clip_norm
                )
                maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
                if self.scaler is None:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            if self.context.is_primary and (step + 1) % self.settings.log_every_steps == 0:
                self.logger.info(
                    "epoch=%d step=%d/%d loss=%.6f",
                    epoch,
                    step + 1,
                    loader_length,
                    float(total.detach()),
                    extra={"event": "train_step", "epoch": epoch, "step": step + 1},
                )
        reduced = reduce_sum(totals, self.context)
        denominator = float(reduced[3].item())
        if denominator == 0:
            raise TrainingError("Training loader produced zero samples")
        epoch_metrics = metrics.compute(self.context)
        reduced_head_losses = {
            name: float(reduce_sum(value, self.context).item() / denominator)
            for name, value in head_totals.items()
        }
        global_gradient_norm = float(
            reduce_max(
                torch.tensor(maximum_gradient_norm, device=self.context.device), self.context
            ).item()
        )
        return (
            float(reduced[0] / denominator),
            float(reduced[1] / denominator),
            float(reduced[2] / denominator),
            int(reduced[4].item()),
            float(reduced[4].item() / reduced[5].item()),
            reduced_head_losses,
            last_learning_rate,
            global_gradient_norm,
            epoch_metrics,
        )

    @torch.no_grad()
    def validate(self, loader: DataLoader[VideoBatch]) -> tuple[float, ClassificationMetrics]:
        self.model.eval()
        metrics = ClassificationAccumulator(self.num_classes, self.context.device)
        totals = torch.zeros(2, dtype=torch.float64, device=self.context.device)
        for batch_cpu in loader:
            batch = batch_cpu.to(
                self.context.device, non_blocking=self.context.device.type == "cuda"
            )
            with _autocast(self.context.device, self.settings.mixed_precision):
                output = self.model(batch.videos)
                loss = functional.cross_entropy(_logits(output), batch.labels)
            metrics.update(_logits(output), batch.labels)
            totals += torch.tensor(
                [float(loss.detach()) * batch.labels.numel(), batch.labels.numel()],
                dtype=torch.float64,
                device=self.context.device,
            )
        reduced = reduce_sum(totals, self.context)
        if float(reduced[1].item()) == 0:
            raise TrainingError("Validation loader produced zero samples")
        return float(reduced[0] / reduced[1]), metrics.compute(self.context)

    def fit(
        self,
        *,
        train_loader: DataLoader[VideoBatch],
        validation_loader: DataLoader[VideoBatch],
        train_dataset: VideoClipDataset | Any,
        train_sampler: EpochSampler | None,
        output_dir: str | Path,
        config_sha256: str,
        seed: int,
        start_epoch: int = 0,
        best_metric: float | None = None,
        best_epoch: int | None = None,
    ) -> list[EpochResult]:
        destination = Path(output_dir).expanduser().resolve()
        checkpoints = destination / "checkpoints"
        if self.context.is_primary:
            checkpoints.mkdir(parents=True, exist_ok=True)
        barrier(self.context)
        history: list[EpochResult] = []
        persisted_history: list[object] = []
        history_path = destination / "history.json"
        if self.context.is_primary and start_epoch > 0 and history_path.is_file():
            existing_history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(existing_history, dict) or not isinstance(
                existing_history.get("epochs"), list
            ):
                raise TrainingError("Existing history.json is invalid")
            persisted_history = existing_history["epochs"]
        for epoch in range(start_epoch, self.settings.epochs):
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if self.context.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.context.device)
                torch.cuda.synchronize(self.context.device)
            training_start = time.perf_counter()
            (
                train_loss,
                classification,
                coupling,
                coupling_valid_positions,
                coupling_coverage,
                auxiliary_head_losses,
                learning_rate,
                gradient_norm,
                train_metrics,
            ) = self.train_epoch(train_loader, epoch)
            if self.context.device.type == "cuda":
                torch.cuda.synchronize(self.context.device)
                peak_training_memory_bytes = torch.cuda.max_memory_allocated(self.context.device)
            else:
                peak_training_memory_bytes = 0
            training_seconds = time.perf_counter() - training_start
            validation_loss, validation_metrics = self.validate(validation_loader)
            current = float(getattr(validation_metrics, self.settings.primary_metric))
            improved = best_metric is None or current > best_metric
            if improved:
                best_metric, best_epoch = current, epoch
            result = EpochResult(
                epoch=epoch,
                training_loss=train_loss,
                classification_loss=classification,
                coupling_loss=coupling,
                coupling_valid_positions=coupling_valid_positions,
                coupling_coverage=coupling_coverage,
                auxiliary_head_losses=auxiliary_head_losses,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm,
                train_metrics=train_metrics,
                validation_loss=validation_loss,
                validation_metrics=validation_metrics,
                training_seconds=training_seconds,
                peak_training_memory_bytes=peak_training_memory_bytes,
            )
            history.append(result)
            gathered_rng = gather_objects(capture_rng_state(), self.context)
            if self.context.is_primary:
                assert gathered_rng is not None
                last_sha = save_training_checkpoint(
                    checkpoints / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch_completed=epoch,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    config_sha256=config_sha256,
                    seed=seed,
                    rng_states=gathered_rng,
                )
                if improved:
                    best_sha = save_training_checkpoint(
                        checkpoints / "best_validation.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch_completed=epoch,
                        best_metric=best_metric,
                        best_epoch=best_epoch,
                        config_sha256=config_sha256,
                        seed=seed,
                        rng_states=gathered_rng,
                    )
                    write_json_atomic(
                        checkpoints / "best_validation.json",
                        {"epoch": epoch, "metric": current, "sha256": best_sha},
                    )
                write_json_atomic(
                    history_path,
                    {"epochs": persisted_history + [entry.as_dict() for entry in history]},
                )
                self.logger.info(
                    "epoch=%d train_loss=%.6f validation_%s=%.6f",
                    epoch,
                    train_loss,
                    self.settings.primary_metric,
                    current,
                    extra={"event": "epoch_complete", "epoch": epoch, "value": current},
                )
                write_json_atomic(
                    checkpoints / "last.json",
                    {"epoch": epoch, "sha256": last_sha},
                )
            barrier(self.context)
        final_rng = gather_objects(capture_rng_state(), self.context) if history else None
        if self.context.is_primary and history:
            assert final_rng is not None
            final_sha = save_training_checkpoint(
                checkpoints / "final.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch_completed=history[-1].epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                config_sha256=config_sha256,
                seed=seed,
                rng_states=final_rng,
            )
            write_json_atomic(
                checkpoints / "final.json",
                {"epoch": history[-1].epoch, "sha256": final_sha},
            )
            raw_model = unwrap_model(self.model)
            if isinstance(raw_model, MCFormer | AuxiliaryFormer):
                rgb_checkpoint = checkpoints / "rgb_only.pt"
                export_rgb_checkpoint(
                    raw_model, rgb_checkpoint, config_sha256=config_sha256, seed=seed
                )
                write_json_atomic(
                    checkpoints / "rgb_only.json",
                    {
                        "sha256": sha256_file(rgb_checkpoint),
                        "model_type": "rgb_only",
                        "config_sha256": config_sha256,
                        "seed": seed,
                    },
                )
        barrier(self.context)
        return history


def optimizer_updates_per_epoch(loader_length: int, accumulation_steps: int) -> int:
    if loader_length <= 0 or accumulation_steps <= 0:
        raise TrainingError("Loader length and accumulation steps must be positive")
    return math.ceil(loader_length / accumulation_steps)
