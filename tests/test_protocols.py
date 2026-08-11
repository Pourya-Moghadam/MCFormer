from __future__ import annotations

import unittest

from mcformer.data.manifest import Manifest, SampleRecord
from mcformer.data.protocols import build_protocol_split, official_ntu_train_test


def sample(
    sample_id: str,
    label: int,
    *,
    dataset: str = "toyota_smarthome",
    subject: int = 1,
    camera: int = 1,
    setup: int = 1,
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        dataset=dataset,
        rgb_path=f"{sample_id}.mp4",
        label_id=label,
        label_name=f"class-{label}",
        subject_id=str(subject),
        camera_or_view_id=str(camera),
        setup_id=str(setup),
    )


class ProtocolTests(unittest.TestCase):
    def test_explicit_split_with_deterministic_stratified_validation(self) -> None:
        records = [
            sample(f"train-{label}-{index}", label) for label in range(2) for index in range(5)
        ]
        records += [sample(f"test-{label}", label) for label in range(2)]
        manifest = Manifest(records, expected_classes=2)
        source = {
            "train": [item.sample_id for item in records if item.sample_id.startswith("train")],
            "test": [item.sample_id for item in records if item.sample_id.startswith("test")],
        }
        first = build_protocol_split(manifest, protocol="cs", explicit_train_test=source)
        second = build_protocol_split(manifest, protocol="cs", explicit_train_test=source)
        self.assertEqual(first, second)
        self.assertEqual(len(first.validation), 2)
        self.assertEqual(first.validation_strategy, "class_stratified_clip_fallback")

    def test_ntu60_official_rules(self) -> None:
        manifest = Manifest(
            [
                sample("train", 0, dataset="ntu_rgbd_60", subject=1, camera=2),
                sample("test", 0, dataset="ntu_rgbd_60", subject=3, camera=1),
            ],
            expected_classes=1,
        )
        train, test = official_ntu_train_test(manifest, "cs")
        self.assertEqual([item.sample_id for item in train], ["train"])
        self.assertEqual([item.sample_id for item in test], ["test"])
        train, test = official_ntu_train_test(manifest, "cv")
        self.assertEqual([item.sample_id for item in train], ["train"])
        self.assertEqual([item.sample_id for item in test], ["test"])

    def test_group_aware_validation_keeps_subjects_disjoint(self) -> None:
        training = [
            sample(f"subject-{subject}-class-{label}", label, subject=subject)
            for subject in range(1, 7)
            for label in range(2)
        ]
        testing = [sample(f"test-{label}", label, subject=99) for label in range(2)]
        manifest = Manifest([*training, *testing], expected_classes=2)
        split = build_protocol_split(
            manifest,
            protocol="cs",
            validation_fraction=0.2,
            explicit_train_test={
                "train": [record.sample_id for record in training],
                "test": [record.sample_id for record in testing],
            },
        )
        train_subjects = {manifest.by_id(sample_id).subject_id for sample_id in split.train}
        validation_subjects = {
            manifest.by_id(sample_id).subject_id for sample_id in split.validation
        }
        self.assertFalse(train_subjects & validation_subjects)
        self.assertEqual(split.validation_strategy, "group_aware:subject_id")


if __name__ == "__main__":
    unittest.main()
