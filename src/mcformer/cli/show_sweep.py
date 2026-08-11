"""Validate and display the frozen E06--E12 experiment matrix."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from mcformer.experiments import load_experiment_matrix, matrix_as_mapping


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(matrix_as_mapping(load_experiment_matrix(args.matrix)), indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
