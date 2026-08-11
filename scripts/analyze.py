#!/usr/bin/env python3
"""Repository-local wrapper for :mod:`mcformer.cli.analyze`."""

from __future__ import annotations

from collections.abc import Sequence

from mcformer.cli.analyze import main as analyze_main


def main(argv: Sequence[str] | None = None) -> int:
    return analyze_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
