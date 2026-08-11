from __future__ import annotations

import unittest

from mcformer.auxiliary.coupling import compute_coupling_target
from mcformer.auxiliary.trajectories import (
    dominant_hand_trajectory,
    primary_object_trajectory,
)
from mcformer.auxiliary.types import (
    ObjectFrame,
    PersonPose,
    PoseFrame,
    TrackedObject,
    Wrist,
)


class TrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.poses = tuple(
            PoseFrame(
                frame_index=index,
                people=(
                    PersonPose(
                        actor_id="actor",
                        left_wrist=(None if index == 2 else Wrist((10.0 + index, 10.0), 1.0)),
                        right_wrist=Wrist((50.0, 50.0), 1.0),
                    ),
                ),
            )
            for index in range(6)
        )

    def test_dominant_hand_interpolation(self) -> None:
        hand = dominant_hand_trajectory(
            self.poses,
            width=100,
            height=100,
            max_gap=1,
            gaussian_sigma=0,
        )
        self.assertEqual(hand.selected_wrist, "left")
        self.assertTrue(hand.valid[2])
        self.assertFalse(hand.observed[2])
        self.assertAlmostEqual(hand.points[2][0], 12 / (2**0.5 * 100))

    def test_primary_object_and_aligned_coupling(self) -> None:
        hand = dominant_hand_trajectory(
            self.poses,
            width=100,
            height=100,
            max_gap=1,
            gaussian_sigma=0,
        )
        objects = tuple(
            ObjectFrame(
                frame_index=index,
                objects=(
                    TrackedObject(1, 39, "bottle", 0.9, (9 + index, 9, 13 + index, 13)),
                    TrackedObject(2, 41, "cup", 0.9, (80, 80, 90, 90)),
                ),
            )
            for index in range(6)
        )
        selected = primary_object_trajectory(objects, hand, width=100, height=100)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.track_id, 1)
        coupling = compute_coupling_target(hand, selected)
        self.assertFalse(coupling.gate[0])
        self.assertTrue(all(coupling.gate[1:]))
        self.assertTrue(all(value > 0.97 for value in coupling.raw[1:]))

    def test_no_eligible_track_masks_target(self) -> None:
        hand = dominant_hand_trajectory(self.poses, width=100, height=100, gaussian_sigma=0)
        objects = tuple(ObjectFrame(index, ()) for index in range(6))
        selected = primary_object_trajectory(objects, hand, width=100, height=100)
        self.assertIsNone(selected)
        coupling = compute_coupling_target(hand, selected)
        self.assertEqual(coupling.coverage, 0)
        self.assertFalse(any(coupling.gate))


if __name__ == "__main__":
    unittest.main()
