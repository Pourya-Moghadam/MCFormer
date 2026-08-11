from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcformer.auxiliary.preprocess import extract_observation_bundle
from mcformer.auxiliary.types import ObjectFrame, PoseFrame
from mcformer.data.manifest import SampleRecord


class FakeFrames:
    shape = (4, 100, 100, 3)

    def __len__(self) -> int:
        return self.shape[0]


class FakePoseBackend:
    def infer(self, frames: object, frame_indices: tuple[int, ...]) -> tuple[PoseFrame, ...]:
        return tuple(PoseFrame(index, ()) for index in frame_indices)


class FakeObjectBackend:
    def track(self, frames: object, frame_indices: tuple[int, ...]) -> tuple[ObjectFrame, ...]:
        return tuple(ObjectFrame(index, ()) for index in frame_indices)


class PreprocessTests(unittest.TestCase):
    @patch("mcformer.auxiliary.preprocess.decode_all_video_frames", return_value=FakeFrames())
    def test_native_frame_bundle_alignment(self, decode: object) -> None:
        record = SampleRecord(
            sample_id="sample",
            dataset="synthetic",
            rgb_path="video.mp4",
            label_id=0,
            label_name="action",
            num_frames=4,
            fps=30,
            width=100,
            height=100,
        )
        result = extract_observation_bundle(
            record,
            root=".",
            object_backend=FakeObjectBackend(),
            pose_backend=FakePoseBackend(),
            pose_source="hrnet",
            mode="native_frames",
        )
        self.assertEqual(result.frame_indices, (0, 1, 2, 3))
        self.assertEqual(len(result.poses), 4)
        self.assertEqual(len(result.objects), 4)

    @patch("mcformer.auxiliary.preprocess.decode_all_video_frames", return_value=FakeFrames())
    def test_portable_projected_3d_source(self, decode: object) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "poses.json").write_text(
                '{"frames":[{"frame_index":0,"people":[{"actor_id":"a",'
                '"left_wrist":{"xyz":[1,2,2],"confidence":1},"right_wrist":null}]}]}',
                encoding="utf-8",
            )
            (root / "calibration.json").write_text(
                '{"projection_matrix":[[100,0,0,0],[0,100,0,0],[0,0,1,0]]}',
                encoding="utf-8",
            )
            record = SampleRecord(
                sample_id="sample",
                dataset="synthetic",
                rgb_path="video.mp4",
                label_id=0,
                label_name="action",
                num_frames=4,
                fps=30,
                width=100,
                height=100,
                skeleton_path="poses.json",
                calibration_path="calibration.json",
            )
            result = extract_observation_bundle(
                record,
                root=root,
                object_backend=FakeObjectBackend(),
                pose_source="projected_3d_json",
            )
        wrist = result.poses[0].people[0].left_wrist
        self.assertIsNotNone(wrist)
        assert wrist is not None
        self.assertEqual(wrist.point, (50.0, 100.0))


if __name__ == "__main__":
    unittest.main()
