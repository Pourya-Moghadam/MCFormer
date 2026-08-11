"""Create one deterministic training-only auxiliary corruption cache for E12."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mcformer.auxiliary.cache import ObservationCache
from mcformer.auxiliary.corruption_cache import CorruptionSpec, build_corrupted_cache
from mcformer.data.manifest import Manifest
from mcformer.data.protocols import ProtocolSplit
from mcformer.reproducibility import write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol-split", required=True)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--corruption",
        required=True,
        choices=("wrist_noise", "missed_detections", "object_occlusion", "track_swap"),
    )
    parser.add_argument("--value", required=True, type=float)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = Manifest.read_jsonl(args.manifest)
    split = ProtocolSplit.read_json(args.protocol_split, manifest)
    source = ObservationCache.open(args.source_cache)
    _, report = build_corrupted_cache(
        source,
        str(Path(args.output).expanduser().resolve()),
        sample_ids=split.train,
        spec=CorruptionSpec(args.corruption, args.value, args.seed),
    )
    write_json_atomic(Path(args.output) / "corruption_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
