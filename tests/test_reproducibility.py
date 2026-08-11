from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from mcformer.device import DeviceError, resolve_device
from mcformer.reproducibility import initialize_run, seed_everything, sha256_file


class ReproducibilityTests(unittest.TestCase):
    def test_python_seed_is_repeatable(self) -> None:
        seed_everything(17)
        first = [random.random() for _ in range(3)]
        seed_everything(17)
        self.assertEqual(first, [random.random() for _ in range(3)])

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.txt"
            path.write_text("mc-former\n", encoding="utf-8")
            self.assertEqual(
                sha256_file(path),
                "9b298f52c178d737de5107dfdcb74931f3da4eed063d0d37066972642301b3b4",
            )

    def test_initialize_run_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            files = initialize_run(
                output_dir=output,
                resolved_config={"seed": 17},
                repository=directory,
            )
            self.assertEqual(set(files), {"config", "environment", "git"})
            self.assertEqual(json.loads(files["config"].read_text())["seed"], 17)
            with self.assertRaises(FileExistsError):
                initialize_run(output_dir=output, resolved_config={}, repository=directory)

    def test_device_cpu_and_unavailable_accelerator(self) -> None:
        self.assertEqual(resolve_device("cpu"), "cpu")
        try:
            resolved = resolve_device("cuda")
        except DeviceError:
            return
        self.assertTrue(resolved.startswith("cuda"))


if __name__ == "__main__":
    unittest.main()
