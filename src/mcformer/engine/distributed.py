"""Small torchrun-compatible distributed runtime helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch


class DistributedError(RuntimeError):
    """Raised when distributed environment variables are incomplete or invalid."""


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    requested_device: str, *, timeout_minutes: int = 30
) -> DistributedContext:
    """Initialize from torchrun variables, or return a single-process context."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size or local_rank < 0:
        raise DistributedError("Invalid WORLD_SIZE/RANK/LOCAL_RANK environment")
    normalized = requested_device.casefold()
    if world_size > 1:
        if not torch.cuda.is_available():
            raise DistributedError("Multi-process training requires CUDA/NCCL")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=timeout_minutes),
            )
        return DistributedContext(rank, world_size, local_rank, device)
    if normalized == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DistributedError("CUDA was requested but is unavailable")
    return DistributedContext(rank=0, world_size=1, local_rank=0, device=device)


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        torch.distributed.barrier()


def reduce_sum(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    result = value.detach().clone()
    if context.enabled:
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result


def reduce_max(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    result = value.detach().clone()
    if context.enabled:
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MAX)
    return result


def gather_objects(value: Any, context: DistributedContext) -> list[Any] | None:
    """Gather serializable rank payloads on rank zero."""

    if not context.enabled:
        return [value]
    output: list[Any] | None = [None] * context.world_size if context.is_primary else None
    torch.distributed.gather_object(value, output, dst=0)
    return output


def shutdown_distributed(context: DistributedContext) -> None:
    if context.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
