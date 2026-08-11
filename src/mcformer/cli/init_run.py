"""Initialize a run directory with resolved configuration and provenance."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mcformer.config import load_config
from mcformer.logging_utils import configure_logging
from mcformer.reproducibility import initialize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.set)
    output = Path(args.output)
    files = initialize_run(
        output_dir=output,
        resolved_config=config.as_dict(),
        repository=args.repository,
    )
    logger = configure_logging(log_file=output / "run.jsonl")
    logger.info("Initialized run metadata", extra={"event": "run_initialized"})
    for name, path in files.items():
        logger.info("Wrote %s metadata to %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
