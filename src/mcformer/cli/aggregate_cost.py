"""Aggregate E14 raw preprocessing trials and optional training histories."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from mcformer.benchmarks.training_cost import (
    aggregate_cache_inventories,
    aggregate_epoch_histories,
    aggregate_preprocessing_trials,
    compare_training_profiles,
)
from mcformer.reproducibility import write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-trial", action="append", required=True)
    parser.add_argument("--cache-inventory", action="append", required=True)
    parser.add_argument("--baseline-history", action="append", default=[])
    parser.add_argument("--mcformer-history", action="append", default=[])
    parser.add_argument("--baseline-training-profile")
    parser.add_argument("--mcformer-training-profile")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: dict[str, object] = {
        "preprocessing": aggregate_preprocessing_trials(args.preprocessing_trial),
        "cache": aggregate_cache_inventories(args.cache_inventory),
    }
    if args.baseline_history:
        result["baseline_training"] = aggregate_epoch_histories(args.baseline_history)
    if args.mcformer_history:
        result["mcformer_training"] = aggregate_epoch_histories(args.mcformer_history)
    if bool(args.baseline_training_profile) != bool(args.mcformer_training_profile):
        build_parser().error("Training graph profiles must be provided as a baseline/method pair")
    if args.baseline_training_profile:
        result["training_graph"] = compare_training_profiles(
            args.baseline_training_profile, args.mcformer_training_profile
        )
    write_json_atomic(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
