from __future__ import annotations

import unittest

from mcformer.auxiliary.coupling import compute_coupling_target
from mcformer.auxiliary.trajectories import ObjectTrajectory, PositionTrajectory


def hand(points: tuple[tuple[float, float], ...]) -> PositionTrajectory:
    valid = tuple(True for _ in points)
    return PositionTrajectory(points=points, valid=valid, observed=valid)


def object_track(points: tuple[tuple[float, float], ...]) -> ObjectTrajectory:
    valid = tuple(True for _ in points)
    return ObjectTrajectory(
        track_id=1,
        class_id=1,
        class_name="object",
        points=points,
        boxes=tuple((x - 0.01, y - 0.01, x + 0.01, y + 0.01) for x, y in points),
        valid=valid,
        observed=valid,
        confidences=tuple(1.0 for _ in points),
        coverage=1.0,
        mean_confidence=1.0,
        median_hand_box_distance=0.0,
    )


class CouplingTests(unittest.TestCase):
    def test_opposite_motion_is_negative(self) -> None:
        target = compute_coupling_target(
            hand(((0.10, 0.10), (0.12, 0.10))),
            object_track(((0.13, 0.10), (0.11, 0.10))),
        )
        self.assertTrue(target.gate[1])
        self.assertLess(target.raw[1], -0.99)

    def test_stationary_motion_is_zero_but_gated(self) -> None:
        target = compute_coupling_target(
            hand(((0.10, 0.10), (0.10, 0.10))),
            object_track(((0.11, 0.10), (0.11, 0.10))),
        )
        self.assertTrue(target.gate[1])
        self.assertEqual(target.raw[1], 0.0)
        self.assertEqual(target.target[1], 0.0)

    def test_distance_masks_otherwise_aligned_motion(self) -> None:
        target = compute_coupling_target(
            hand(((0.1, 0.1), (0.2, 0.1))),
            object_track(((0.5, 0.5), (0.6, 0.5))),
            distance_threshold=0.15,
        )
        self.assertFalse(target.gate[1])
        self.assertGreater(target.raw[1], 0.99)
        self.assertEqual(target.target[1], 0.0)


if __name__ == "__main__":
    unittest.main()
