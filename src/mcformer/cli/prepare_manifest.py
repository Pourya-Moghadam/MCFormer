"""Build a validated dataset manifest without redistributing source data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mcformer.data.manifest import (
    build_ntu_manifest,
    build_tabular_manifest,
    read_label_map,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("ntu_rgbd_60", "ntu_rgbd_120", "toyota_smarthome"),
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skeleton-root")
    parser.add_argument("--annotations", help="Required portable CSV for Toyota")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument(
        "--missing-samples",
        help="Required for NTU: one excluded sample ID per line (an explicit empty file is valid)",
    )
    parser.add_argument("--no-video-inspection", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = read_label_map(args.label_map)
    inspect = not args.no_video_inspection
    if args.dataset.startswith("ntu_"):
        if not args.missing_samples:
            build_parser().error("NTU datasets require --missing-samples")
        missing = [
            line.strip()
            for line in Path(args.missing_samples).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = build_ntu_manifest(
            rgb_root=args.data_root,
            dataset=args.dataset,
            label_map=labels,
            skeleton_root=args.skeleton_root,
            missing_sample_ids=missing,
            inspect_video=inspect,
        )
    else:
        if not args.annotations:
            build_parser().error("--annotations is required for Toyota Smarthome")
        manifest = build_tabular_manifest(
            annotation_path=args.annotations,
            data_root=args.data_root,
            dataset=args.dataset,
            label_map=labels,
            delimiter=args.delimiter,
            inspect_video=inspect,
        )
    digest = manifest.write_jsonl(args.output)
    print(f"Wrote {len(manifest)} records to {args.output} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
