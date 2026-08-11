from __future__ import annotations

import unittest
from unittest.mock import patch

from mcformer.data.dataset import VideoClipDataset
from mcformer.data.manifest import Manifest, SampleRecord


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = SampleRecord(
            sample_id="sample",
            dataset="synthetic",
            rgb_path="video.mp4",
            label_id=0,
            label_name="action",
            num_frames=200,
            fps=30,
            width=640,
            height=480,
        )
        self.manifest = Manifest([self.record], expected_classes=1)

    @patch("mcformer.data.dataset.apply_rgb_transform", return_value="normalized-video")
    @patch("mcformer.data.dataset.decode_video_frames", return_value="decoded-frames")
    def test_deterministic_clip_and_target_alignment(self, decode: object, apply: object) -> None:
        calls: list[tuple[int, ...]] = []

        def target_builder(
            record: object, indices: tuple[int, ...], spatial: object
        ) -> dict[str, object]:
            calls.append(indices)
            return {"indices": indices, "spatial": spatial}

        dataset = VideoClipDataset(
            manifest=self.manifest,
            sample_ids=["sample"],
            root=".",
            training=True,
            seed=17,
            target_builder=target_builder,
        )
        first = dataset[0]
        second = dataset[0]
        self.assertEqual(first.frame_indices, second.frame_indices)
        self.assertEqual(first.spatial_transform, second.spatial_transform)
        self.assertEqual(first.video, "normalized-video")
        self.assertEqual(calls, [first.frame_indices, first.frame_indices])
        dataset.set_epoch(1)
        third = dataset[0]
        self.assertNotEqual(first.frame_indices, third.frame_indices)

    @patch("mcformer.data.dataset.apply_rgb_transform", return_value="normalized-video")
    @patch("mcformer.data.dataset.decode_video_frames", return_value="decoded-frames")
    def test_evaluation_clip_is_centered(self, decode: object, apply: object) -> None:
        dataset = VideoClipDataset(
            manifest=self.manifest,
            sample_ids=["sample"],
            root=".",
            training=False,
            seed=17,
        )
        sample = dataset[0]
        self.assertEqual(sample.frame_indices[0], 68)
        self.assertEqual(sample.frame_indices[-1], 130)


if __name__ == "__main__":
    unittest.main()
