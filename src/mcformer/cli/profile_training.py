"""Profile the E14 FP16 forward/backward training graph without dataset I/O."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from mcformer.auxiliary.cache import configuration_digest
from mcformer.benchmarks.training_profile import profile_training_graph
from mcformer.config import load_config
from mcformer.models.registry import build_model
from mcformer.reproducibility import write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("E14 training graph profiling requires CUDA")
    config = load_config(args.config, args.set)
    model, _ = build_model(config, allow_random_initialization=True)
    result = profile_training_graph(model, device=torch.device("cuda:0"))
    result["config_sha256"] = configuration_digest(config.as_dict())
    write_json_atomic(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
