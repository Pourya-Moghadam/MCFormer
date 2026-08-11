"""Spatially aligned RGB, point, and bounding-box transforms."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


class TransformError(ValueError):
    """Raised when spatial input or configuration is invalid."""


@dataclass(frozen=True, slots=True)
class SpatialTransform:
    """Affine resize/crop/flip description shared by RGB and auxiliary geometry."""

    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    output_size: int
    horizontal_flip: bool

    @property
    def resize_x(self) -> float:
        return self.resized_width / self.source_width

    @property
    def resize_y(self) -> float:
        return self.resized_height / self.source_height

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """Map one source-image point to output pixel coordinates."""

        x = (point[0] * self.resize_x - self.crop_left) * self.output_size / self.crop_width
        y = (point[1] * self.resize_y - self.crop_top) * self.output_size / self.crop_height
        if self.horizontal_flip:
            x = self.output_size - 1 - x
        return x, y

    def transform_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Map and order an ``x1,y1,x2,y2`` source-image box."""

        x1 = (box[0] * self.resize_x - self.crop_left) * self.output_size / self.crop_width
        x2 = (box[2] * self.resize_x - self.crop_left) * self.output_size / self.crop_width
        y1 = (box[1] * self.resize_y - self.crop_top) * self.output_size / self.crop_height
        y2 = (box[3] * self.resize_y - self.crop_top) * self.output_size / self.crop_height
        if self.horizontal_flip:
            x1, x2 = self.output_size - x2, self.output_size - x1
        return (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        )


def _resize_dimensions(width: int, height: int, short_side: int) -> tuple[int, int]:
    if min(width, height) <= 0 or short_side <= 0:
        raise TransformError("Image dimensions and short side must be positive")
    scale = short_side / min(width, height)
    return max(round(width * scale), 1), max(round(height * scale), 1)


def _random_crop(
    width: int,
    height: int,
    *,
    scale_range: tuple[float, float],
    ratio_range: tuple[float, float],
    rng: random.Random,
) -> tuple[int, int, int, int]:
    area = width * height
    log_ratio = (math.log(ratio_range[0]), math.log(ratio_range[1]))
    for _ in range(10):
        target_area = area * rng.uniform(*scale_range)
        aspect = math.exp(rng.uniform(*log_ratio))
        crop_width = round(math.sqrt(target_area * aspect))
        crop_height = round(math.sqrt(target_area / aspect))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = rng.randint(0, width - crop_width)
            top = rng.randint(0, height - crop_height)
            return left, top, crop_width, crop_height
    input_ratio = width / height
    if input_ratio < ratio_range[0]:
        crop_width = width
        crop_height = round(crop_width / ratio_range[0])
    elif input_ratio > ratio_range[1]:
        crop_height = height
        crop_width = round(crop_height * ratio_range[1])
    else:
        crop_width, crop_height = width, height
    return (
        (width - crop_width) // 2,
        (height - crop_height) // 2,
        crop_width,
        crop_height,
    )


def make_spatial_transform(
    *,
    width: int,
    height: int,
    training: bool,
    rng: random.Random | None = None,
    resize_short_side: int = 256,
    output_size: int = 224,
    crop_scale: tuple[float, float] = (0.8, 1.0),
    crop_ratio: tuple[float, float] = (0.75, 4 / 3),
    flip_probability: float = 0.5,
) -> SpatialTransform:
    """Sample the manuscript's training transform or deterministic center crop."""

    if not 0 <= flip_probability <= 1:
        raise TransformError("flip_probability must lie in [0,1]")
    if not 0 < crop_scale[0] <= crop_scale[1] <= 1:
        raise TransformError("crop_scale must be ordered within (0,1]")
    if not 0 < crop_ratio[0] <= crop_ratio[1]:
        raise TransformError("crop_ratio must be positive and ordered")
    resized_width, resized_height = _resize_dimensions(width, height, resize_short_side)
    generator = rng if rng is not None else random.Random()
    if training:
        left, top, crop_width, crop_height = _random_crop(
            resized_width,
            resized_height,
            scale_range=crop_scale,
            ratio_range=crop_ratio,
            rng=generator,
        )
        flip = generator.random() < flip_probability
    else:
        crop_width = min(output_size, resized_width)
        crop_height = min(output_size, resized_height)
        left = (resized_width - crop_width) // 2
        top = (resized_height - crop_height) // 2
        flip = False
    return SpatialTransform(
        source_width=width,
        source_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        crop_left=left,
        crop_top=top,
        crop_width=crop_width,
        crop_height=crop_height,
        output_size=output_size,
        horizontal_flip=flip,
    )


def apply_rgb_transform(
    frames: Any,
    transform: SpatialTransform,
    *,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Any:
    """Apply a shared spatial transform and return normalized ``T,C,H,W`` float32."""

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise TransformError("RGB transforms require NumPy and OpenCV") from error
    if getattr(frames, "ndim", None) != 4 or frames.shape[-1] != 3:
        raise TransformError("frames must have shape T,H,W,3")
    if frames.shape[2] != transform.source_width or frames.shape[1] != transform.source_height:
        raise TransformError("Frame dimensions do not match the spatial transform")
    output: list[Any] = []
    for frame in frames:
        resized = cv2.resize(
            frame,
            (transform.resized_width, transform.resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        cropped = resized[
            transform.crop_top : transform.crop_top + transform.crop_height,
            transform.crop_left : transform.crop_left + transform.crop_width,
        ]
        final = cv2.resize(
            cropped,
            (transform.output_size, transform.output_size),
            interpolation=cv2.INTER_LINEAR,
        )
        if transform.horizontal_flip:
            final = np.ascontiguousarray(final[:, ::-1])
        output.append(final)
    array = np.stack(output).astype(np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(array, (0, 3, 1, 2))
