"""Evaluate a verified checkpoint on one fixed validation or test partition."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from torch.nn.parallel import DistributedDataParallel

from mcformer.auxiliary.cache import configuration_digest
from mcformer.cli.runtime import load_data_contract, make_dataset, section, selected_seed
from mcformer.config import load_config
from mcformer.engine.checkpointing import load_inference_state, read_checkpoint
from mcformer.engine.data import build_data_loader
from mcformer.engine.distributed import barrier, initialize_distributed, shutdown_distributed
from mcformer.evaluation.evaluator import evaluate_model, write_evaluation_artifacts
from mcformer.logging_utils import configure_logging
from mcformer.models.registry import build_model
from mcformer.reproducibility import seed_everything, sha256_file, write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol-split", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="test")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.set)
    project = section(config, "project")
    training = section(config, "training")
    model_config = section(config, "model")
    reproducibility = section(config, "reproducibility")
    context = initialize_distributed(str(project.get("device", "auto")))
    try:
        seed = selected_seed(config, args.seed)
        seed_everything(seed, deterministic=bool(reproducibility.get("deterministic", True)))
        logger = configure_logging()
        manifest, split = load_data_contract(
            config, manifest_path=args.manifest, protocol_path=args.protocol_split
        )
        sample_ids = split.validation if args.partition == "validation" else split.test
        dataset = make_dataset(
            config,
            manifest,
            sample_ids,
            root=args.data_root,
            training=False,
            seed=seed,
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
        model = load_inference_state(
            model,
            checkpoint,
            config_sha256=configuration_digest(config.as_dict()),
            seed=seed,
        )
        model.to(context.device)
        if context.enabled:
            model = DistributedDataParallel(model, device_ids=[context.local_rank])
        result = evaluate_model(
            model,
            loader,
            context=context,
            num_classes=int(model_config["num_classes"]),
            mixed_precision=str(training["mixed_precision"]),
        )
        if context.is_primary:
            output = Path(args.output).expanduser().resolve()
            write_evaluation_artifacts(result, output)
            write_json_atomic(
                output / "provenance.json",
                {
                    "partition": args.partition,
                    "seed": seed,
                    "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                    "checkpoint_sha256": checkpoint_sha,
                    "manifest_sha256": sha256_file(args.manifest),
                    "protocol_sha256": sha256_file(args.protocol_split),
                },
            )
            logger.info(
                "Evaluated %d samples: top1=%.6f mCA=%.6f",
                result.metrics.samples,
                result.metrics.top1_accuracy,
                result.metrics.mean_class_accuracy,
            )
        barrier(context)
        return 0
    finally:
        shutdown_distributed(context)


if __name__ == "__main__":
    raise SystemExit(main())
