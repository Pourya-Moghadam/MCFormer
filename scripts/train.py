#!/usr/bin/env python3
"""Repository-local wrapper for :mod:`mcformer.cli.train`."""

from __future__ import annotations

from collections.abc import Sequence

from mcformer.cli.train import main as train_main


def main(argv: Sequence[str] | None = None) -> int:
    return train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
