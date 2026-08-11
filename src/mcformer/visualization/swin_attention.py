"""Exact final-stage Video Swin attention capture and residual rollout."""

from __future__ import annotations

import itertools
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from mcformer.models.torchvision_backbones import VideoSwinTinyBackbone


class AttentionRolloutError(RuntimeError):
    """Raised when the pinned Video Swin attention surface is incompatible."""


def _window_and_shift(module: Any, size: tuple[int, int, int]) -> tuple[list[int], list[int]]:
    window = [int(value) for value in module.window_size]
    shift = [int(value) for value in module.shift_size]
    for index, extent in enumerate(size):
        if extent <= window[index]:
            window[index], shift[index] = extent, 0
    return window, shift


def _partition(value: Tensor, window: list[int]) -> Tensor:
    batch, frames, height, width = value.shape[:4]
    trailing = value.shape[4:]
    reshaped = value.reshape(
        batch,
        frames // window[0],
        window[0],
        height // window[1],
        window[1],
        width // window[2],
        window[2],
        *trailing,
    )
    permutation = (0, 1, 3, 5, 2, 4, 6, *range(7, reshaped.ndim))
    return reshaped.permute(permutation).reshape(-1, window[0] * window[1] * window[2], *trailing)


def _shift_mask(
    padded_size: tuple[int, int, int], window: list[int], shift: list[int], reference: Tensor
) -> Tensor | None:
    if not any(shift):
        return None
    mask = reference.new_zeros((1, *padded_size, 1))

    def slices(extent: int, window_extent: int, shift_extent: int) -> tuple[slice, ...]:
        if shift_extent == 0:
            return (slice(0, extent),)
        return (
            slice(0, -window_extent),
            slice(-window_extent, -shift_extent),
            slice(-shift_extent, None),
        )

    dimensions = [
        slices(extent, window_extent, shift_extent)
        for extent, window_extent, shift_extent in zip(padded_size, window, shift, strict=True)
    ]
    for counter, (temporal, vertical, horizontal) in enumerate(itertools.product(*dimensions)):
        mask[:, temporal, vertical, horizontal, :] = counter
    windows = _partition(mask, window).squeeze(-1)
    differences = windows.unsqueeze(1) - windows.unsqueeze(2)
    return differences.masked_fill(differences != 0, -100.0).masked_fill(differences == 0, 0.0)


def _attention_and_indices(
    value: Tensor, module: Any
) -> tuple[Tensor, Tensor, tuple[int, int, int]]:
    if value.ndim != 5:
        raise AttentionRolloutError("Video Swin attention input must be B,T,H,W,C")
    batch, frames, height, width, channels = value.shape
    size = (frames, height, width)
    window, shift = _window_and_shift(module, size)
    padding = tuple(
        (window[index] - size[index] % window[index]) % window[index] for index in range(3)
    )
    padded = functional.pad(value, (0, 0, 0, padding[2], 0, padding[1], 0, padding[0]))
    padded_size = (int(padded.shape[1]), int(padded.shape[2]), int(padded.shape[3]))
    if any(shift):
        padded = torch.roll(padded, shifts=tuple(-item for item in shift), dims=(1, 2, 3))
    windows = _partition(padded, window)
    tokens = windows.shape[1]
    heads = int(module.num_heads)
    qkv = functional.linear(windows, module.qkv.weight, module.qkv.bias)
    qkv = qkv.reshape(windows.shape[0], tokens, 3, heads, channels // heads).permute(2, 0, 3, 1, 4)
    query, key = qkv[0], qkv[1]
    query = query * (channels // heads) ** -0.5
    attention = query @ key.transpose(-2, -1)
    try:
        relative_bias = module.get_relative_position_bias(window)
    except TypeError:
        relative_bias = module.get_relative_position_bias()
    attention = attention + relative_bias
    mask = _shift_mask(padded_size, window, shift, attention)
    windows_per_sample = attention.shape[0] // batch
    if mask is not None:
        attention = attention.reshape(batch, windows_per_sample, heads, tokens, tokens)
        attention = attention + mask[None, :, None]
        attention = attention.reshape(-1, heads, tokens, tokens)
    attention = attention.softmax(dim=-1).mean(dim=1)

    indices = torch.arange(frames * height * width, device=value.device).reshape(
        1, frames, height, width, 1
    )
    indices = functional.pad(indices, (0, 0, 0, padding[2], 0, padding[1], 0, padding[0]), value=-1)
    if any(shift):
        indices = torch.roll(indices, shifts=tuple(-item for item in shift), dims=(1, 2, 3))
    return (
        attention.reshape(batch, windows_per_sample, tokens, tokens),
        _partition(indices, window).reshape(windows_per_sample, tokens),
        size,
    )


def _global_attention(attention: Tensor, indices: Tensor, token_count: int) -> Tensor:
    batch = attention.shape[0]
    result = attention.new_zeros((batch, token_count, token_count))
    for window_index in range(indices.shape[0]):
        valid = indices[window_index] >= 0
        selected = indices[window_index, valid].long()
        for batch_index in range(batch):
            result[batch_index][selected[:, None], selected[None, :]] = attention[
                batch_index, window_index
            ][valid][:, valid]
    identity = torch.eye(token_count, dtype=result.dtype, device=result.device)[None]
    result = result + identity
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def final_stage_attention_rollout(
    backbone: VideoSwinTinyBackbone, video: Tensor
) -> tuple[Any, Tensor]:
    """Run the model and return mean-received residual attention at input resolution.

    Video Swin has no class token. The unambiguous release rule is therefore the
    mean attention received by every token after multiplying the two final-stage
    residual attention matrices. Heads are averaged before rollout.
    """

    if video.shape[0] != 1:
        raise AttentionRolloutError("Qualitative rollout requires batch size one")
    try:
        final_stage = backbone.model.features[6]
        modules = [block.attn for block in final_stage]
    except (AttributeError, IndexError, TypeError) as error:
        raise AttentionRolloutError("Pinned torchvision final attention stage changed") from error
    captured: list[tuple[Tensor, Tensor, tuple[int, int, int]]] = []
    handles: list[Any] = []

    def hook(module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], Tensor):
            raise AttentionRolloutError("Attention hook did not receive a tensor")
        captured.append(_attention_and_indices(inputs[0], module))

    for module in modules:
        handles.append(module.register_forward_pre_hook(hook))
    try:
        output = backbone(video)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(modules) or not captured:
        raise AttentionRolloutError("Did not capture every final-stage attention block")
    size = captured[0][2]
    token_count = size[0] * size[1] * size[2]
    rollout = torch.eye(token_count, device=video.device, dtype=captured[0][0].dtype)[None]
    for attention, indices, block_size in captured:
        if block_size != size:
            raise AttentionRolloutError("Final-stage token grid changed inside rollout")
        rollout = _global_attention(attention, indices, token_count) @ rollout
    received = rollout.mean(dim=1).reshape(1, 1, *size)
    upsampled = functional.interpolate(
        received,
        size=(video.shape[1], video.shape[3], video.shape[4]),
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    flat = upsampled.flatten(1)
    minimum, maximum = flat.min(dim=1).values, flat.max(dim=1).values
    normalized = (upsampled - minimum[:, None, None]) / (maximum - minimum).clamp_min(1e-12)[
        :, None, None
    ]
    return output, normalized
