"""Train one configured seed with deterministic resume and complete artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from mcformer.auxiliary.cache import ObservationCache, configuration_digest
from mcformer.auxiliary.pipeline import target_builder_from_cache
from mcformer.cli.runtime import (
    load_data_contract,
    make_dataset,
    section,
    selected_seed,
    target_settings,
)
from mcformer.config import load_config
from mcformer.engine.checkpointing import resume_training_checkpoint
from mcformer.engine.data import build_data_loader
from mcformer.engine.distributed import barrier, initialize_distributed, shutdown_distributed
from mcformer.engine.optim import OptimizerSettings, WarmupCosineScheduler, build_adamw
from mcformer.engine.trainer import (
    Trainer,
    TrainerSettings,
    create_grad_scaler,
    optimizer_updates_per_epoch,
)
from mcformer.logging_utils import configure_logging
from mcformer.models.registry import build_model
from mcformer.reproducibility import initialize_run, seed_everything, sha256_file, write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol-split", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initialization-checkpoint")
    parser.add_argument("--initialization-sha256")
    parser.add_argument("--resume")
    parser.add_argument("--resume-sha256")
    parser.add_argument("--allow-random-initialization", action="store_true")
    parser.add_argument("--repository", default="..")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def _exact_accumulation(global_batch: int, local_batch: int, world_size: int) -> int:
    denominator = local_batch * world_size
    if global_batch % denominator:
        raise ValueError(
            f"Global batch {global_batch} must be divisible by per-device batch {local_batch} "
            f"times world size {world_size}"
        )
    return global_batch // denominator


def _prepare_run(
    output: Path, config_values: Mapping[str, object], repository: str, resume: bool
) -> None:
    if resume:
        existing = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
        if existing != config_values:
            raise ValueError("Resume configuration differs from the run's resolved configuration")
    else:
        initialize_run(output_dir=output, resolved_config=config_values, repository=repository)


def _parameter_audit(model: torch.nn.Module) -> dict[str, object]:
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    rgb_model = getattr(model, "rgb_model", model)
    deployed = sum(
        parameter.numel() for parameter in rgb_model.parameters() if parameter.requires_grad
    )
    auxiliary_modules = getattr(model, "heads", None)
    if auxiliary_modules is None and hasattr(model, "mcim"):
        auxiliary_modules = {"mcim": model.mcim}
    head_parameters = (
        {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in auxiliary_modules.items()
        }
        if auxiliary_modules is not None
        else {}
    )
    backbone = getattr(rgb_model, "backbone", None)
    return {
        "total_trainable_parameters": total,
        "deployed_trainable_parameters": deployed,
        "training_only_parameters": total - deployed,
        "auxiliary_head_parameters": head_parameters,
        "backbone_output_dim": getattr(backbone, "output_dim", None),
        "auxiliary_token_dim": getattr(backbone, "temporal_dim", None),
        "insertion_stage": getattr(backbone, "insertion_stage", None),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume is not None and (
        args.initialization_checkpoint is not None
        or args.initialization_sha256 is not None
        or args.allow_random_initialization
    ):
        raise ValueError("Resume cannot also specify initialization options")
    if (args.initialization_checkpoint is None) != (args.initialization_sha256 is None):
        raise ValueError("Initialization checkpoint and SHA-256 must be provided together")
    config = load_config(args.config, args.set)
    reproducibility = section(config, "reproducibility")
    project = section(config, "project")
    training = section(config, "training")
    model_config = section(config, "model")
    logging_config = section(config, "logging")
    context = initialize_distributed(str(project.get("device", "auto")))
    output = Path(args.output).expanduser().resolve()
    try:
        seed = selected_seed(config, args.seed)
        seed_everything(seed, deterministic=bool(reproducibility.get("deterministic", True)))
        if context.is_primary:
            _prepare_run(output, config.as_dict(), args.repository, args.resume is not None)
        barrier(context)
        logger = configure_logging(
            level=str(logging_config.get("level", "INFO")),
            log_file=output / "run.jsonl" if context.is_primary else None,
        )
        manifest, split = load_data_contract(
            config, manifest_path=args.manifest, protocol_path=args.protocol_split
        )
        target_builder = None
        cache_sha256 = None
        mcim = model_config.get("mcim")
        has_auxiliary_heads = model_config.get("auxiliary_heads") is not None
        if (isinstance(mcim, Mapping) and mcim.get("enabled") is True) or has_auxiliary_heads:
            if args.cache is None:
                raise ValueError("MC-Former training requires --cache")
            cache = ObservationCache.open(args.cache)
            for sample_id in split.train:
                cache.read(sample_id)
            cache_sha256 = cache.content_digest(split.train)
            target_builder = target_builder_from_cache(cache, target_settings(config))
        train_dataset = make_dataset(
            config,
            manifest,
            split.train,
            root=args.data_root,
            training=True,
            seed=seed,
            target_builder=target_builder,
        )
        validation_dataset = make_dataset(
            config,
            manifest,
            split.validation,
            root=args.data_root,
            training=False,
            seed=seed,
        )
        local_batch = int(training.get("per_device_batch_size", 4))
        workers = int(training.get("num_workers", 4))
        pin_memory = bool(training.get("pin_memory", True)) and context.device.type == "cuda"
        accumulation = _exact_accumulation(
            int(training["batch_size"]), local_batch, context.world_size
        )
        train_loader, train_sampler = build_data_loader(
            train_dataset,
            batch_size=local_batch,
            training=True,
            seed=seed,
            num_workers=workers,
            pin_memory=pin_memory,
            distributed=context.enabled,
            rank=context.rank,
            world_size=context.world_size,
        )
        validation_loader, _ = build_data_loader(
            validation_dataset,
            batch_size=local_batch,
            training=False,
            seed=seed,
            num_workers=workers,
            pin_memory=pin_memory,
            distributed=context.enabled,
            rank=context.rank,
            world_size=context.world_size,
        )
        model, initialization = build_model(
            config,
            initialization_checkpoint=args.initialization_checkpoint,
            initialization_sha256=args.initialization_sha256,
            allow_random_initialization=args.allow_random_initialization or args.resume is not None,
        )
        model.to(context.device)
        parameter_audit = _parameter_audit(model)
        beta_values = training["adam_betas"]
        if not isinstance(beta_values, list) or len(beta_values) != 2:
            raise ValueError("training.adam_betas must contain exactly two values")
        optimizer_settings = OptimizerSettings(
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            betas=(float(beta_values[0]), float(beta_values[1])),
            epsilon=float(training["adam_epsilon"]),
            warmup_epochs=int(training["warmup_epochs"]),
            minimum_learning_rate=float(training["minimum_learning_rate"]),
        )
        optimizer = build_adamw(model.parameters(), optimizer_settings)
        updates = optimizer_updates_per_epoch(len(train_loader), accumulation)
        epochs = int(training["epochs"])
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_steps=updates * epochs,
            warmup_steps=updates * optimizer_settings.warmup_epochs,
            base_learning_rate=optimizer_settings.learning_rate,
            minimum_learning_rate=optimizer_settings.minimum_learning_rate,
        )
        precision = str(training["mixed_precision"])
        scaler = create_grad_scaler(context.device, precision)
        config_sha = configuration_digest(config.as_dict())
        start_epoch = 0
        best_metric = None
        best_epoch = None
        if args.resume is not None:
            if args.resume_sha256 is None:
                raise ValueError("--resume requires --resume-sha256")
            resumed = resume_training_checkpoint(
                args.resume,
                expected_sha256=args.resume_sha256,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config_sha256=config_sha,
                seed=seed,
                rank=context.rank,
            )
            start_epoch, best_metric, best_epoch = (
                resumed.next_epoch,
                resumed.best_metric,
                resumed.best_epoch,
            )
        if context.enabled:
            model = DistributedDataParallel(model, device_ids=[context.local_rank])
        if context.is_primary:
            setup = {
                "seed": seed,
                "world_size": context.world_size,
                "per_device_batch_size": local_batch,
                "gradient_accumulation_steps": accumulation,
                "effective_global_batch_size": local_batch * context.world_size * accumulation,
                "manifest_sha256": sha256_file(args.manifest),
                "protocol_sha256": sha256_file(args.protocol_split),
                "cache_sha256": cache_sha256,
                "initialization": asdict(initialization) if initialization else None,
                "model_audit": parameter_audit,
            }
            setup_path = output / "run_setup.json"
            if args.resume is not None:
                previous_setup = json.loads(setup_path.read_text(encoding="utf-8"))
                identity_keys = set(setup) - {"initialization"}
                if any(previous_setup.get(key) != setup[key] for key in identity_keys):
                    raise ValueError("Resume data, seed, or distributed batch identity differs")
            else:
                write_json_atomic(setup_path, setup)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            settings=TrainerSettings(
                epochs=epochs,
                accumulation_steps=accumulation,
                gradient_clip_norm=float(training["gradient_clip_norm"]),
                mixed_precision=precision,
                coupling_weight=float(training["coupling_weight"]),
                log_every_steps=int(logging_config["log_every_steps"]),
                primary_metric=str(config.get("evaluation.primary_metric", "top1_accuracy")),
            ),
            context=context,
            num_classes=int(model_config["num_classes"]),
            logger=logger,
            scaler=scaler,
        )
        trainer.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            train_dataset=train_dataset,
            train_sampler=train_sampler if hasattr(train_sampler, "set_epoch") else None,
            output_dir=output,
            config_sha256=config_sha,
            seed=seed,
            start_epoch=start_epoch,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
        return 0
    finally:
        shutdown_distributed(context)


if __name__ == "__main__":
    raise SystemExit(main())
