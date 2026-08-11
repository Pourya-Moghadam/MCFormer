"""CPU/GPU device selection kept independent from model code."""

from __future__ import annotations


class DeviceError(RuntimeError):
    """Raised when a requested accelerator is unavailable."""


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto`` or validate a requested PyTorch device string."""

    normalized = requested.lower()
    try:
        import torch
    except ImportError:
        if normalized in {"auto", "cpu"}:
            return "cpu"
        raise DeviceError(f"Cannot use {requested!r}: PyTorch is not installed") from None

    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise DeviceError(f"Requested {requested!r}, but CUDA is unavailable")
    if normalized == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise DeviceError("Requested 'mps', but MPS is unavailable")
    if normalized != "cpu" and normalized != "mps" and not normalized.startswith("cuda"):
        raise DeviceError(f"Unsupported device: {requested!r}")
    return normalized
