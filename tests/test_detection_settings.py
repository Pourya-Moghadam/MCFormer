from __future__ import annotations

import unittest

from mcformer.auxiliary.detection import ByteTrackSettings, DetectionError


class DetectionSettingsTests(unittest.TestCase):
    def test_manuscript_defaults_are_valid(self) -> None:
        ByteTrackSettings().validate()

    def test_rejects_reversed_thresholds(self) -> None:
        with self.assertRaises(DetectionError):
            ByteTrackSettings(high_threshold=0.1, low_threshold=0.5).validate()


if __name__ == "__main__":
    unittest.main()
