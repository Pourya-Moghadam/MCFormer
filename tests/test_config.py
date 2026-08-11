from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcformer.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_resolves_defaults_and_override(self) -> None:
        config = load_config(
            ROOT / "configs/experiment/e01_toyota_cs_mcformer.yaml",
            ['data.protocol="cv1"', "training.batch_size=8"],
        )
        self.assertEqual(config.get("data.dataset"), "toyota_smarthome")
        self.assertEqual(config.get("data.protocol"), "cv1")
        self.assertEqual(config.get("training.batch_size"), 8)
        self.assertTrue(config.get("model.mcim.enabled"))
        self.assertEqual(config.get("reproducibility.seeds"), [17, 29, 43])

    def test_returns_defensive_copy(self) -> None:
        config = load_config(ROOT / "configs/experiment/e01_toyota_cs_baseline.yaml")
        copied = config.as_dict()
        copied["training"]["epochs"] = 1
        self.assertEqual(config.get("training.epochs"), 50)

    def test_rejects_inheritance_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle.yaml"
            path.write_text(json.dumps({"defaults": ["cycle.yaml"]}), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "inheritance cycle"):
                load_config(path)

    def test_rejects_invalid_section_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            values = load_config(ROOT / "configs/experiment/e01_toyota_cs_baseline.yaml").as_dict()
            values["training"] = "not-a-mapping"
            path.write_text(json.dumps(values), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "training must be a mapping"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
