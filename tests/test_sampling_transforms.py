from __future__ import annotations

import random
import unittest

from mcformer.data.sampling import sample_frame_indices
from mcformer.data.transforms import make_spatial_transform


class SamplingTransformTests(unittest.TestCase):
    def test_center_sampling_and_padding(self) -> None:
        centered = sample_frame_indices(100, clip_length=32, stride=2, training=False)
        self.assertEqual(centered.start, 18)
        self.assertEqual(centered.indices[0], 18)
        self.assertEqual(centered.indices[-1], 80)
        padded = sample_frame_indices(3, clip_length=4, stride=2, training=False)
        self.assertEqual(padded.indices, (0, 2, 2, 2))
        self.assertEqual(padded.padding_mask, (False, False, True, True))

    def test_training_sampling_is_rng_deterministic(self) -> None:
        first = sample_frame_indices(200, training=True, rng=random.Random(17))
        second = sample_frame_indices(200, training=True, rng=random.Random(17))
        self.assertEqual(first, second)

    def test_spatial_geometry_center_and_flip(self) -> None:
        center = make_spatial_transform(
            width=640,
            height=480,
            training=False,
            resize_short_side=256,
            output_size=224,
        )
        x, y = center.transform_point((320, 240))
        self.assertAlmostEqual(x, 112.0, delta=0.6)
        self.assertAlmostEqual(y, 112.0, delta=0.6)
        flipped = make_spatial_transform(
            width=100,
            height=100,
            training=True,
            rng=random.Random(1),
            resize_short_side=100,
            output_size=100,
            crop_scale=(1.0, 1.0),
            crop_ratio=(1.0, 1.0),
            flip_probability=1.0,
        )
        self.assertEqual(flipped.transform_point((10, 20)), (89.0, 20.0))
        self.assertEqual(flipped.transform_box((0, 0, 100, 100)), (0.0, 0.0, 100.0, 100.0))


if __name__ == "__main__":
    unittest.main()
