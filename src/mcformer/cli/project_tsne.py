"""Generate the frozen E16 paired joint PCA-50/t-SNE projection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from mcformer.data.manifest import Manifest
from mcformer.data.subsets import load_subset_names, resolve_label_ids
from mcformer.evaluation.features import read_feature_archive
from mcformer.visualization.tsne import (
    TSNESettings,
    joint_tsne,
    select_paired_features,
    write_tsne_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-features", required=True)
    parser.add_argument("--mcformer-features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subsets", required=True)
    parser.add_argument("--subset-name", default="manipulation_actions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--standardize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = Manifest.read_jsonl(args.manifest)
    names = load_subset_names(args.subsets, args.subset_name)
    class_ids = resolve_label_ids(manifest, names)
    labels = {record.label_id: record.label_name for record in manifest}
    baseline, method = select_paired_features(
        read_feature_archive(args.baseline_features),
        read_feature_archive(args.mcformer_features),
        class_ids=class_ids,
        cap_per_class=200,
    )
    settings = TSNESettings(standardize=args.standardize)
    coordinates = joint_tsne(baseline, method, settings)
    write_tsne_artifacts(
        baseline,
        method,
        coordinates,
        class_names={class_id: labels[class_id] for class_id in class_ids},
        settings=settings,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
