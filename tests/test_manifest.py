from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcformer.data.manifest import Manifest, ManifestError, SampleRecord


def record(sample_id: str, label: int = 0) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        dataset="synthetic",
        rgb_path=f"videos/{sample_id}.mp4",
        label_id=label,
        label_name=f"class-{label}",
        subject_id="1",
        num_frames=64,
        fps=30.0,
        width=640,
        height=480,
    )


class ManifestTests(unittest.TestCase):
    def test_canonical_round_trip(self) -> None:
        manifest = Manifest([record("b", 1), record("a", 0)], expected_classes=2)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            first_hash = manifest.write_jsonl(first)
            loaded = Manifest.read_jsonl(first, expected_classes=2)
            second_hash = loaded.write_jsonl(second)
            self.assertEqual([item.sample_id for item in loaded], ["a", "b"])
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_rejects_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ManifestError, "Duplicate"):
            Manifest([record("same"), record("same")])

    def test_rejects_duplicate_video_paths(self) -> None:
        first = record("first")
        second = SampleRecord(
            sample_id="second",
            dataset=first.dataset,
            rgb_path=first.rgb_path,
            label_id=first.label_id,
            label_name=first.label_name,
        )
        with self.assertRaisesRegex(ManifestError, "Duplicate RGB"):
            Manifest([first, second])

    def test_rejects_noncontiguous_labels(self) -> None:
        with self.assertRaisesRegex(ManifestError, "contiguous"):
            Manifest([record("a", 0), record("b", 2)], expected_classes=3)


if __name__ == "__main__":
    unittest.main()
