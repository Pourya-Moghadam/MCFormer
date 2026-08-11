"""Extract E16 pre-classifier representations from one verified test checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mcformer.auxiliary.cache import configuration_digest
from mcformer.cli.runtime import load_data_contract, make_dataset, section, selected_seed
from mcformer.config import load_config
from mcformer.engine.checkpointing import load_inference_state, read_checkpoint
from mcformer.engine.data import build_data_loader
from mcformer.engine.distributed import barrier, initialize_distributed, shutdown_distributed
from mcformer.evaluation.features import (
    extract_features,
    write_feature_archive,
    write_feature_index,
)
from mcformer.models.registry import build_model
from mcformer.reproducibility import sha256_file, write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol-split", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.set)
    project, training = section(config, "project"), section(config, "training")
    context = initialize_distributed(str(project.get("device", "auto")))
    try:
        seed = selected_seed(config, args.seed)
        manifest, split = load_data_contract(
            config, manifest_path=args.manifest, protocol_path=args.protocol_split
        )
        dataset = make_dataset(
            config, manifest, split.test, root=args.data_root, training=False, seed=seed
        )
        loader, _ = build_data_loader(
            dataset,
            batch_size=int(training.get("per_device_batch_size", 4)),
            training=False,
            seed=seed,
            num_workers=int(training.get("num_workers", 4)),
            pin_memory=bool(training.get("pin_memory", True)) and context.device.type == "cuda",
            distributed=context.enabled,
            rank=context.rank,
            world_size=context.world_size,
        )
        model, _ = build_model(config, allow_random_initialization=True)
        checkpoint, checkpoint_sha = read_checkpoint(
            args.checkpoint, expected_sha256=args.checkpoint_sha256
        )
        config_sha = configuration_digest(config.as_dict())
        model = load_inference_state(model, checkpoint, config_sha256=config_sha, seed=seed).to(
            context.device
        )
        archive = extract_features(model, loader, context=context)
        if context.is_primary:
            assert archive is not None
            output = Path(args.output).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            write_feature_archive(archive, output / "features.npz")
            write_feature_index(archive, output / "samples.jsonl")
            write_json_atomic(
                output / "provenance.json",
                {
                    "seed": seed,
                    "partition": "test",
                    "feature": "final classifier-input pooled representation",
                    "checkpoint_sha256": checkpoint_sha,
                    "config_sha256": config_sha,
                    "manifest_sha256": sha256_file(args.manifest),
                    "protocol_sha256": sha256_file(args.protocol_split),
                },
            )
        barrier(context)
        return 0
    finally:
        shutdown_distributed(context)


if __name__ == "__main__":
    raise SystemExit(main())
