from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcformer.auxiliary.cache import (
    CacheError,
    ObservationBundle,
    ObservationCache,
    configuration_digest,
)
from mcformer.auxiliary.pipeline import TargetSettings, build_sample_target
from mcformer.auxiliary.types import (
    ObjectFrame,
    PersonPose,
    PoseFrame,
    TrackedObject,
    Wrist,
)
from mcformer.data.transforms import make_spatial_transform
from mcformer.reproducibility import sha256_file


def bundle() -> ObservationBundle:
    indices = tuple(range(6))
    return ObservationBundle(
        sample_id="sample/unsafe-name",
        width=100,
        height=100,
        frame_indices=indices,
        poses=tuple(
            PoseFrame(
                frame_index=index,
                people=(
                    PersonPose(
                        "actor",
                        Wrist((10 + index, 10), 1.0),
                        Wrist((50, 50), 1.0),
                    ),
                ),
            )
            for index in indices
        ),
        objects=tuple(
            ObjectFrame(
                frame_index=index,
                objects=(TrackedObject(7, 39, "bottle", 0.9, (9 + index, 9, 13 + index, 13)),),
            )
            for index in indices
        ),
        pose_backend="fixture",
        object_backend="fixture",
        mode="native_frames",
    )


class CachePipelineTests(unittest.TestCase):
    def test_cache_round_trip_and_deterministic_bytes(self) -> None:
        key = configuration_digest({"settings": 1})
        with tempfile.TemporaryDirectory() as directory:
            first = ObservationCache(Path(directory) / "a", cache_key=key)
            second = ObservationCache(Path(directory) / "b", cache_key=key)
            first.initialize({"settings": 1})
            second.initialize({"settings": 1})
            first_path = first.write(bundle())
            second_path = second.write(bundle())
            self.assertEqual(sha256_file(first_path), sha256_file(second_path))
            self.assertEqual(first.read(bundle().sample_id), bundle())
            first.write_target(bundle().sample_id, {"gate": [False, True]})
            self.assertEqual(first.read_target(bundle().sample_id), {"gate": [False, True]})
            with self.assertRaisesRegex(CacheError, "overwrite"):
                first.write(bundle())
            incompatible = ObservationCache(first.root, cache_key="different")
            with self.assertRaisesRegex(CacheError, "incompatible"):
                incompatible.initialize({"settings": 1})

    def test_build_sample_target_after_spatial_transform(self) -> None:
        spatial = make_spatial_transform(
            width=100,
            height=100,
            training=False,
            resize_short_side=100,
            output_size=100,
        )
        target = build_sample_target(
            bundle(),
            frame_indices=tuple(range(6)),
            spatial_transform=spatial,
            settings=TargetSettings(gaussian_sigma_frames=0),
        )
        self.assertEqual(target.object_trajectory.track_id, 7)
        self.assertTrue(all(target.coupling.gate[1:]))
        self.assertTrue(all(value > 0.97 for value in target.coupling.raw[1:]))


if __name__ == "__main__":
    unittest.main()
