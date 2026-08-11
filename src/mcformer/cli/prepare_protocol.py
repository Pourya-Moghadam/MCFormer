"""Generate a validated train/validation/test protocol manifest."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from mcformer.data.manifest import Manifest
from mcformer.data.protocols import (
    build_protocol_split,
    read_explicit_train_test,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--official-split", help="Required train/test JSON for Toyota protocols")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--validation-seed", type=int, default=2026)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = Manifest.read_jsonl(args.manifest)
    explicit = read_explicit_train_test(args.official_split) if args.official_split else None
    if manifest[0].dataset == "toyota_smarthome" and explicit is None:
        build_parser().error("Toyota protocols require --official-split")
    split = build_protocol_split(
        manifest,
        protocol=args.protocol,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        explicit_train_test=explicit,
    )
    split.write_json(args.output)
    print(
        f"Wrote {args.protocol}: train={len(split.train)}, validation={len(split.validation)}, "
        f"test={len(split.test)} ({split.validation_strategy})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
