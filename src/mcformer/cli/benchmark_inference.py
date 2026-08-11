"""Benchmark an RGB-only deployed checkpoint under the frozen E13 protocol."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from mcformer.auxiliary.cache import configuration_digest
from mcformer.benchmarks.inference import (
    InferenceBenchmarkSettings,
    run_inference_benchmark,
)
from mcformer.cli.runtime import selected_seed
from mcformer.config import load_config
from mcformer.engine.checkpointing import load_inference_state, read_checkpoint
from mcformer.models.classifier import AuxiliaryFormer, MCFormer, VideoClassifier
from mcformer.models.registry import build_model
from mcformer.reproducibility import write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-non-reference-gpu", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("E13 requires CUDA; no benchmark result was written")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    if "V100-SXM2-32GB" not in name.replace(" ", "-") and not args.allow_non_reference_gpu:
        raise RuntimeError(
            f"Reference E13 hardware is V100-SXM2 32GB, found {name!r}; "
            "pass --allow-non-reference-gpu to record a clearly identified comparison"
        )
    config = load_config(args.config, args.set)
    seed = selected_seed(config, args.seed)
    model, _ = build_model(config, allow_random_initialization=True)
    checkpoint, checkpoint_sha = read_checkpoint(
        args.checkpoint, expected_sha256=args.checkpoint_sha256
    )
    if isinstance(model, MCFormer | AuxiliaryFormer) and checkpoint.get("model_type") != "rgb_only":
        raise RuntimeError(
            "E13 MC-Former measurements require the exported RGB-only checkpoint; "
            "the training graph is not a deployed model"
        )
    model = load_inference_state(
        model,
        checkpoint,
        config_sha256=configuration_digest(config.as_dict()),
        seed=seed,
    )
    if not isinstance(model, VideoClassifier):
        raise RuntimeError("E13 only benchmarks a deployed RGB VideoClassifier graph")
    result = run_inference_benchmark(
        model, device=device, settings=InferenceBenchmarkSettings(seed=2026)
    )
    result["checkpoint_sha256"] = checkpoint_sha
    result["config_sha256"] = configuration_digest(config.as_dict())
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "inference_benchmark.json", result)
    (output / "modules.txt").write_text(
        "\n".join(f"{item['name']}\t{item['type']}" for item in result["modules"]) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
