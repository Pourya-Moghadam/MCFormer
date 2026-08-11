"""Audit manifest files, decoded metadata, and protocol integrity."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mcformer.data.manifest import Manifest, ManifestError, inspect_video_metadata
from mcformer.data.protocols import ProtocolSplit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol-split")
    parser.add_argument("--require-skeleton", action="store_true")
    parser.add_argument("--decode-metadata", action="store_true")
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--expected-train", type=int)
    parser.add_argument("--expected-validation", type=int)
    parser.add_argument("--expected-test", type=int)
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = Manifest.read_jsonl(args.manifest)
    if args.expected_samples is not None and len(manifest) != args.expected_samples:
        raise ManifestError(f"Expected {args.expected_samples} samples, found {len(manifest)}")
    root = Path(args.data_root).expanduser().resolve()
    manifest.validate_files(root, require_skeleton=args.require_skeleton)
    if args.protocol_split:
        split = ProtocolSplit.read_json(args.protocol_split, manifest)
        expected_counts = {
            "train": args.expected_train,
            "validation": args.expected_validation,
            "test": args.expected_test,
        }
        for partition, count in expected_counts.items():
            partition_actual = len(getattr(split, partition))
            if count is not None and partition_actual != count:
                raise ManifestError(
                    f"Expected {count} {partition} samples, found {partition_actual}"
                )
    elif any(
        value is not None
        for value in (args.expected_train, args.expected_validation, args.expected_test)
    ):
        build_parser().error("Partition count checks require --protocol-split")
    if args.decode_metadata:
        mismatches: list[str] = []
        for record in manifest:
            decoded = inspect_video_metadata(_resolve(root, record.rgb_path))
            recorded: dict[str, int | float | None] = {
                "num_frames": record.num_frames,
                "fps": record.fps,
                "width": record.width,
                "height": record.height,
            }
            decoded_values: dict[str, int | float] = {
                "num_frames": decoded["num_frames"],
                "fps": decoded["fps"],
                "width": decoded["width"],
                "height": decoded["height"],
            }
            for key, value in decoded_values.items():
                recorded_value = recorded[key]
                if recorded_value is None:
                    mismatches.append(f"{record.sample_id}:{key}:missing")
                elif key == "fps":
                    if abs(float(recorded_value) - float(value)) > 0.01:
                        mismatches.append(f"{record.sample_id}:{key}")
                elif int(recorded_value) != int(value):
                    mismatches.append(f"{record.sample_id}:{key}")
        if mismatches:
            raise ManifestError(f"Decoded metadata mismatches: {mismatches[:20]}")
    print(f"Validated {len(manifest)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
