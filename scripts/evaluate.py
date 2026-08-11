#!/usr/bin/env python3
"""Repository-local wrapper for :mod:`mcformer.cli.evaluate`."""

from __future__ import annotations

from collections.abc import Sequence

from mcformer.cli.evaluate import main as evaluate_main


def main(argv: Sequence[str] | None = None) -> int:
    return evaluate_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
