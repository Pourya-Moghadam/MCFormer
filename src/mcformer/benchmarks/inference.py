"""Reference E13 deployed-model inference benchmark implementation."""

from __future__ import annotations

import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from mcformer.benchmarks.timing import BenchmarkError


@dataclass(frozen=True)
class InferenceBenchmarkSettings:
    """Frozen manuscript timing protocol; non-defaults are explicit CLI choices."""

    frames: int = 32
    image_size: int = 224
    batch_size: int = 1
    warmup_iterations: int = 50
    timed_iterations: int = 200
    runs: int = 5
    seed: int = 2026

    def validate(self) -> None:
        values = (
            self.frames,
            self.image_size,
            self.batch_size,
            self.warmup_iterations,
            self.timed_iterations,
            self.runs,
        )
        if any(value <= 0 for value in values):
            raise BenchmarkError("Benchmark dimensions and iteration counts must be positive")


def parameter_report(model: nn.Module) -> dict[str, int]:
    """Count unique deployed parameters without double-counting shared tensors."""

    parameters = {parameter.data_ptr(): parameter for parameter in model.parameters()}
    return {
        "total": sum(parameter.numel() for parameter in parameters.values()),
        "trainable": sum(
            parameter.numel() for parameter in parameters.values() if parameter.requires_grad
        ),
    }


def module_listing(model: nn.Module) -> list[dict[str, str]]:
    return [
        {"name": name or "<root>", "type": f"{type(module).__module__}.{type(module).__name__}"}
        for name, module in model.named_modules()
    ]


def environment_report(device: torch.device) -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        value.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
            }
        )
    return value


def profiler_macs(model: nn.Module, inputs: Tensor, device: torch.device) -> float:
    """Count profiler FLOPs and convert the conventional 2-FLOP MAC to one MAC.

    PyTorch's profiler counts a multiply and add separately for supported
    operators.  The paper uses one multiply-add as one FLOP, so the stored value
    is half of the profiler total. Unsupported operators are listed in provenance
    by the profiler version and this method name; the result is never hard-coded.
    """

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with (
        torch.inference_mode(),
        torch.profiler.profile(activities=activities, with_flops=True) as profile,
    ):
        model(inputs)
    return float(sum(event.flops for event in profile.key_averages())) / 2.0


def run_inference_benchmark(
    model: nn.Module,
    *,
    device: torch.device,
    settings: InferenceBenchmarkSettings,
) -> dict[str, Any]:
    """Measure pre-transferred FP32 clips, excluding decode and host transfer."""

    settings.validate()
    if device.type != "cuda":
        raise BenchmarkError("The E13 reference benchmark requires a CUDA device")
    generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    inputs = torch.randn(
        settings.batch_size,
        settings.frames,
        3,
        settings.image_size,
        settings.image_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    model.eval().to(device=device, dtype=torch.float32)
    run_seconds: list[float] = []
    iteration_seconds: list[list[float]] = []
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(settings.warmup_iterations):
            model(inputs)
        torch.cuda.synchronize(device)
        for _ in range(settings.runs):
            samples: list[float] = []
            for _ in range(settings.timed_iterations):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(inputs)
                torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - start)
            iteration_seconds.append(samples)
            run_seconds.append(sum(samples))
    throughputs = [
        settings.timed_iterations * settings.batch_size / duration for duration in run_seconds
    ]
    return {
        "schema_version": 1,
        "settings": asdict(settings),
        "precision": "fp32",
        "includes_io": False,
        "iteration_seconds": iteration_seconds,
        "run_seconds": run_seconds,
        "clips_per_second": throughputs,
        "clips_per_second_mean": statistics.fmean(throughputs),
        "clips_per_second_sample_sd": statistics.stdev(throughputs),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "parameters": parameter_report(model),
        "macs_per_clip": profiler_macs(model, inputs[:1], device),
        "flop_convention": "one multiply-add equals one FLOP",
        "environment": environment_report(device),
        "modules": module_listing(model),
    }
