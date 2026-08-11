"""Small, testable timing primitives used by preprocessing and GPU benchmarks."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field


class BenchmarkError(RuntimeError):
    """Raised when a benchmark contract is invalid or incomplete."""


@dataclass
class TimingAccumulator:
    """Collect positive wall-clock samples and expose paper-ready summaries."""

    unit: str = "seconds"
    samples: list[float] = field(default_factory=list)

    def add(self, duration: float) -> None:
        if not math.isfinite(duration) or duration <= 0:
            raise BenchmarkError("Timing samples must be finite and positive")
        self.samples.append(duration)

    def measure(self, operation: Callable[[], object], synchronize: Callable[[], None]) -> object:
        synchronize()
        start = time.perf_counter()
        result = operation()
        synchronize()
        self.add(time.perf_counter() - start)
        return result

    def summary(self) -> dict[str, float | int | str]:
        if not self.samples:
            raise BenchmarkError("Cannot summarize an empty timing accumulator")
        return {
            "unit": self.unit,
            "count": len(self.samples),
            "mean": statistics.fmean(self.samples),
            "sample_sd": statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0,
            "minimum": min(self.samples),
            "maximum": max(self.samples),
        }
