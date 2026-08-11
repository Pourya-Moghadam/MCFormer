from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcformer.data.manifest import Manifest, ManifestError, SampleRecord
from mcformer.data.subsets import load_subset_names, resolve_label_ids


class SubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = Manifest(
            [
                SampleRecord("a", "toyota", "a.mp4", 0, "Drink.Frombottle"),
                SampleRecord("b", "toyota", "b.mp4", 1, "Drink From Can"),
            ]
        )

    def test_resolves_punctuation_insensitively(self) -> None:
        self.assertEqual(
            resolve_label_ids(self.manifest, ["drink from bottle", "Drink.From.Can"]),
            (0, 1),
        )

    def test_rejects_unknown_label(self) -> None:
        with self.assertRaisesRegex(ManifestError, "resolved to 0 labels"):
            resolve_label_ids(self.manifest, ["unknown"])

    def test_loads_named_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subsets.json"
            path.write_text('{"pair":["A","B"]}', encoding="utf-8")
            self.assertEqual(load_subset_names(path, "pair"), ("A", "B"))


if __name__ == "__main__":
    unittest.main()
