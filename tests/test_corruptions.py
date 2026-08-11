from __future__ import annotations

import random
import unittest

from mcformer.auxiliary.corruptions import (
    add_wrist_noise,
    drop_object_detections,
    occlude_track,
)
from mcformer.auxiliary.types import (
    ObjectFrame,
    PersonPose,
    PoseFrame,
    TrackedObject,
    Wrist,
)


class CorruptionTests(unittest.TestCase):
    def test_wrist_noise_is_repeatable(self) -> None:
        frames = (PoseFrame(0, (PersonPose("a", Wrist((1, 2), 1), None),)),)
        first = add_wrist_noise(
            frames,
            sigma_diagonal=0.02,
            width=100,
            height=100,
            rng=random.Random(2026),
        )
        second = add_wrist_noise(
            frames,
            sigma_diagonal=0.02,
            width=100,
            height=100,
            rng=random.Random(2026),
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, frames)

    def test_drop_and_contiguous_occlusion(self) -> None:
        frames = tuple(
            ObjectFrame(index, (TrackedObject(1, 0, "object", 1.0, (0, 0, 1, 1)),))
            for index in range(8)
        )
        dropped = drop_object_detections(frames, probability=1.0, rng=random.Random(1))
        self.assertFalse(any(frame.objects for frame in dropped))
        occluded = occlude_track(frames, track_id=1, length=4, rng=random.Random(1))
        missing = [index for index, frame in enumerate(occluded) if not frame.objects]
        self.assertEqual(len(missing), 4)
        self.assertEqual(missing, list(range(missing[0], missing[0] + 4)))


if __name__ == "__main__":
    unittest.main()
