from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcformer.logging_utils import configure_logging


class LoggingTests(unittest.TestCase):
    def test_json_line_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            logger = configure_logging(log_file=path)
            logger.info(
                "metric recorded",
                extra={"event": "metric", "metric": "loss", "value": 1.25},
            )
            for handler in logger.handlers:
                handler.flush()
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["event"], "metric")
            self.assertEqual(payload["metric"], "loss")
            self.assertEqual(payload["value"], 1.25)


if __name__ == "__main__":
    unittest.main()
