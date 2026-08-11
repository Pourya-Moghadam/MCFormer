from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcformer.auxiliary.pose import (
    PoseError,
    associate_people,
    read_ntu_skeleton,
    read_projected_3d_pose,
)
from mcformer.auxiliary.types import PersonPose, PoseFrame, Wrist


def joint_row(color_x: float, color_y: float) -> str:
    return f"0 0 0 0 0 {color_x} {color_y} 0 0 0 0\n"


class PoseTests(unittest.TestCase):
    def test_reads_ntu_projected_wrists(self) -> None:
        lines = ["1\n", "1\n", "123 0 0 0 0 0 0 0 0 0\n", "25\n"]
        for index in range(25):
            if index == 6:
                lines.append(joint_row(10, 20))
            elif index == 10:
                lines.append(joint_row(30, 40))
            else:
                lines.append(joint_row(1, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.skeleton"
            path.write_text("".join(lines), encoding="utf-8")
            frames = read_ntu_skeleton(path)
        self.assertEqual(frames[0].people[0].actor_id, "123")
        self.assertEqual(frames[0].people[0].left_wrist.point, (10, 20))
        self.assertEqual(frames[0].people[0].right_wrist.point, (30, 40))

    def test_associates_people_across_order_changes(self) -> None:
        first = PoseFrame(
            0,
            (
                PersonPose("0", Wrist((10, 10), 1), None),
                PersonPose("1", Wrist((90, 90), 1), None),
            ),
        )
        second = PoseFrame(
            1,
            (
                PersonPose("0", Wrist((91, 90), 1), None),
                PersonPose("1", Wrist((11, 10), 1), None),
            ),
        )
        associated = associate_people((first, second), maximum_distance=10)
        self.assertEqual(associated[0].people[0].actor_id, associated[1].people[1].actor_id)
        self.assertEqual(associated[0].people[1].actor_id, associated[1].people[0].actor_id)

    def test_projects_portable_3d_wrists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calibration.json").write_text(
                '{"projection_matrix":[[100,0,10,0],[0,100,20,0],[0,0,1,0]]}',
                encoding="utf-8",
            )
            (root / "pose.json").write_text(
                '{"frames":[{"frame_index":3,"people":[{"actor_id":"subject-1",'
                '"left_wrist":{"xyz":[1,2,2],"confidence":0.8},'
                '"right_wrist":null}]}]}',
                encoding="utf-8",
            )
            frames = read_projected_3d_pose(root / "pose.json", root / "calibration.json")
        self.assertEqual(frames[0].frame_index, 3)
        self.assertEqual(frames[0].people[0].left_wrist.point, (60.0, 120.0))
        self.assertIsNone(frames[0].people[0].right_wrist)

    def test_rejects_invalid_projection_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calibration.json").write_text(
                '{"projection_matrix":[[1,0],[0,1]]}', encoding="utf-8"
            )
            (root / "pose.json").write_text('{"frames":[]}', encoding="utf-8")
            with self.assertRaisesRegex(PoseError, "shape 3x4"):
                read_projected_3d_pose(root / "pose.json", root / "calibration.json")


if __name__ == "__main__":
    unittest.main()
