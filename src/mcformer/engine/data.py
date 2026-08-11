"""Typed collation and deterministic PyTorch DataLoader construction."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sized
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from mcformer.data.dataset import ClipSample


class BatchError(ValueError):
    """Raised when samples cannot form a scientifically valid batch."""


@dataclass(frozen=True)
class VideoBatch:
    videos: Tensor
    labels: Tensor
    sample_ids: tuple[str, ...]
    frame_indices: Tensor
    padding_mask: Tensor
    coupling_target: Tensor | None
    coupling_mask: Tensor | None
    auxiliary_targets: dict[str, Tensor]
    auxiliary_masks: dict[str, Tensor]

    def to(self, device: torch.device, *, non_blocking: bool = False) -> VideoBatch:
        return VideoBatch(
            videos=self.videos.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            sample_ids=self.sample_ids,
            frame_indices=self.frame_indices.to(device, non_blocking=non_blocking),
            padding_mask=self.padding_mask.to(device, non_blocking=non_blocking),
            coupling_target=(
                self.coupling_target.to(device, non_blocking=non_blocking)
                if self.coupling_target is not None
                else None
            ),
            coupling_mask=(
                self.coupling_mask.to(device, non_blocking=non_blocking)
                if self.coupling_mask is not None
                else None
            ),
            auxiliary_targets={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.auxiliary_targets.items()
            },
            auxiliary_masks={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.auxiliary_masks.items()
            },
        )


def _coupling(sample: ClipSample) -> tuple[Tensor, Tensor] | None:
    if sample.auxiliary is None:
        return None
    coupling = sample.auxiliary.get("coupling")
    if not isinstance(coupling, dict):
        raise BatchError(f"Sample {sample.sample_id} has no coupling mapping")
    target = coupling.get("target")
    gate = coupling.get("gate")
    if not isinstance(target, list | tuple) or not isinstance(gate, list | tuple):
        raise BatchError(f"Sample {sample.sample_id} has invalid coupling target/mask")
    if len(target) != len(sample.frame_indices) or len(gate) != len(sample.frame_indices):
        raise BatchError(f"Sample {sample.sample_id} coupling length is not clip-aligned")
    return torch.tensor(target, dtype=torch.float32), torch.tensor(gate, dtype=torch.bool)


def _targets(sample: ClipSample) -> dict[str, tuple[Tensor, Tensor]]:
    if sample.auxiliary is None:
        return {}
    raw_targets = sample.auxiliary.get("targets")
    if raw_targets is None:
        coupling = _coupling(sample)
        return {"temporal_gated": coupling} if coupling is not None else {}
    if not isinstance(raw_targets, dict):
        raise BatchError(f"Sample {sample.sample_id} targets must be a mapping")
    result: dict[str, tuple[Tensor, Tensor]] = {}
    for name, raw in raw_targets.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise BatchError(f"Sample {sample.sample_id} has an invalid auxiliary target")
        target, mask = raw.get("target"), raw.get("mask")
        if not isinstance(target, list | tuple) or not isinstance(mask, list | tuple):
            raise BatchError(f"Sample {sample.sample_id} target {name!r} is malformed")
        if not target or len(target) != len(mask):
            raise BatchError(f"Sample {sample.sample_id} target {name!r} has invalid lengths")
        result[name] = (
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
        )
    return result


def collate_video_samples(samples: list[ClipSample]) -> VideoBatch:
    """Collate clips while requiring all-or-none auxiliary targets."""

    if not samples:
        raise BatchError("Cannot collate an empty batch")
    videos = torch.stack([torch.as_tensor(sample.video, dtype=torch.float32) for sample in samples])
    labels = torch.tensor([sample.label for sample in samples], dtype=torch.long)
    frame_indices = torch.tensor([sample.frame_indices for sample in samples], dtype=torch.long)
    padding_mask = torch.tensor([sample.padding_mask for sample in samples], dtype=torch.bool)
    auxiliary = [_coupling(sample) for sample in samples]
    present = [value is not None for value in auxiliary]
    if any(present) and not all(present):
        raise BatchError("A batch cannot mix samples with and without auxiliary targets")
    coupling_target: Tensor | None = None
    coupling_mask: Tensor | None = None
    if all(present):
        values = [value for value in auxiliary if value is not None]
        coupling_target = torch.stack([value[0] for value in values])
        coupling_mask = torch.stack([value[1] for value in values])
    target_sets = [_targets(sample) for sample in samples]
    target_names = set(target_sets[0])
    if any(set(values) != target_names for values in target_sets[1:]):
        raise BatchError("Every sample in a batch must provide the same auxiliary targets")
    auxiliary_targets = {
        name: torch.stack([values[name][0] for values in target_sets]) for name in target_names
    }
    auxiliary_masks = {
        name: torch.stack([values[name][1] for values in target_sets]) for name in target_names
    }
    return VideoBatch(
        videos=videos,
        labels=labels,
        sample_ids=tuple(sample.sample_id for sample in samples),
        frame_indices=frame_indices,
        padding_mask=padding_mask,
        coupling_target=coupling_target,
        coupling_mask=coupling_mask,
        auxiliary_targets=auxiliary_targets,
        auxiliary_masks=auxiliary_masks,
    )


def seed_worker(worker_id: int) -> None:
    """Derive Python and NumPy worker state from PyTorch's assigned worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(worker_seed)


class DistributedEvalSampler(Sampler[int]):
    """Partition evaluation indices without DistributedSampler's padding duplicates."""

    def __init__(self, dataset: Dataset[Any], *, rank: int, world_size: int) -> None:
        if world_size <= 1 or not 0 <= rank < world_size:
            raise BatchError("Invalid distributed evaluation rank or world size")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(cast(Sized, self.dataset)), self.world_size))

    def __len__(self) -> int:
        remaining = len(cast(Sized, self.dataset)) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def build_data_loader(
    dataset: Dataset[ClipSample],
    *,
    batch_size: int,
    training: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader[VideoBatch], Sampler[int] | None]:
    """Build a seeded loader and an epoch-aware distributed sampler when required."""

    if batch_size <= 0 or num_workers < 0:
        raise BatchError("batch_size must be positive and num_workers non-negative")
    sampler: Sampler[int] | None = None
    if distributed:
        sampler = (
            DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=False)
            if training
            else DistributedEvalSampler(dataset, rank=rank, world_size=world_size)
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = cast(
        DataLoader[VideoBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # Workers must be recreated so VideoClipDataset.set_epoch is visible in each process.
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=generator,
            collate_fn=collate_video_samples,
            drop_last=False,
        ),
    )
    return loader, sampler
