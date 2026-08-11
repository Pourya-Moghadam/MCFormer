"""Architecture-only E14 FP16 training graph parameter/FLOP profiler."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor, nn

from mcformer.benchmarks.inference import (
    InferenceBenchmarkSettings,
    environment_report,
    module_listing,
    parameter_report,
)


def _training_objective(output: Any) -> Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise RuntimeError("Training profile model output lacks logits")
    objective = logits.float().sum()
    coupling = getattr(output, "coupling", None)
    if isinstance(coupling, Tensor):
        objective = objective + coupling.float().sum()
    predictions = getattr(output, "auxiliary_predictions", None)
    if isinstance(predictions, dict):
        for value in predictions.values():
            if not isinstance(value, Tensor):
                raise RuntimeError("Auxiliary prediction is not a tensor")
            objective = objective + value.float().sum()
    return objective


def profile_training_graph(
    model: nn.Module,
    *,
    device: torch.device,
    settings: InferenceBenchmarkSettings | None = None,
) -> dict[str, Any]:
    """Measure supported forward+backward operations once and report per clip.

    The scalar objective touches classification and every training-only head. It
    changes no operation count relative to CE/MSE reductions in a material way,
    while avoiding a dependency on dataset targets for this architecture audit.
    """

    effective = settings or InferenceBenchmarkSettings()
    effective.validate()
    if device.type != "cuda":
        raise RuntimeError("The reference E14 FP16 training profile requires CUDA")
    generator = torch.Generator(device="cpu").manual_seed(effective.seed)
    inputs = torch.randn(
        effective.batch_size,
        effective.frames,
        3,
        effective.image_size,
        effective.image_size,
        generator=generator,
    ).to(device)
    model.train().to(device)
    model.zero_grad(set_to_none=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, with_flops=True) as profile:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(inputs)
            objective = _training_objective(output)
        objective.backward()  # type: ignore[no-untyped-call]
    macs = float(sum(event.flops for event in profile.key_averages())) / 2.0
    return {
        "schema_version": 1,
        "settings": asdict(effective),
        "precision": "fp16",
        "scope": "forward and backward; optimizer step excluded",
        "macs_per_clip": macs / effective.batch_size,
        "flop_convention": "one multiply-add equals one FLOP",
        "parameters": parameter_report(model),
        "environment": environment_report(device),
        "modules": module_listing(model),
    }
