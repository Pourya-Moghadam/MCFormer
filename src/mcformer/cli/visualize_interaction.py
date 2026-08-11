"""Generate the deterministic E15 Toyota interaction/attention figure."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import torch

from mcformer.auxiliary.cache import ObservationCache, configuration_digest
from mcformer.cli.runtime import load_data_contract, make_dataset
from mcformer.config import load_config
from mcformer.data.subsets import resolve_label_ids
from mcformer.engine.checkpointing import load_inference_state, read_checkpoint
from mcformer.evaluation.predictions import PredictionSet, load_predictions
from mcformer.models.classifier import AuxiliaryFormer, MCFormer, VideoClassifier
from mcformer.models.registry import build_model
from mcformer.models.torchvision_backbones import VideoSwinTinyBackbone
from mcformer.reproducibility import sha256_file
from mcformer.visualization.qualitative import (
    render_qualitative_figure,
    rgb_from_normalized,
    select_qualitative_frames,
)
from mcformer.visualization.swin_attention import final_stage_attention_rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol-split", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def _sample_id(
    *,
    class_id: int,
    test_ids: tuple[str, ...],
    predictions: PredictionSet,
    cache: ObservationCache,
) -> str:
    by_id = predictions.by_id
    for sample_id in sorted(test_ids):
        prediction = by_id.get(sample_id)
        if prediction is None or prediction.label != class_id or prediction.prediction != class_id:
            continue
        target = cache.read_target(sample_id)["sample_target"]
        if float(target["coupling"]["coverage"]) >= 0.5:
            return sample_id
    raise RuntimeError("No correctly classified eligible Drink-From-bottle test sample exists")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.set)
    manifest, split = load_data_contract(
        config, manifest_path=args.manifest, protocol_path=args.protocol_split
    )
    drink_id = resolve_label_ids(manifest, ("Drink \u2013 From bottle",))[0]
    predictions = load_predictions(args.predictions)
    if tuple(record.sample_id for record in predictions.records) != tuple(sorted(split.test)):
        raise RuntimeError("E15 predictions must cover exactly the official Toyota-CS test split")
    if any(
        manifest.by_id(record.sample_id).label_id != record.label for record in predictions.records
    ):
        raise RuntimeError("E15 prediction labels disagree with the manifest")
    cache = ObservationCache.open(args.cache)
    sample_id = _sample_id(
        class_id=drink_id, test_ids=split.test, predictions=predictions, cache=cache
    )
    dataset = make_dataset(
        config, manifest, (sample_id,), root=args.data_root, training=False, seed=17
    )
    sample = dataset[0]
    cached = cache.read_target(sample_id)
    if tuple(int(value) for value in cached["frame_indices"]) != sample.frame_indices:
        raise RuntimeError("E15 cached target frame indices disagree with the evaluated clip")
    if cached["spatial_transform"] != asdict(sample.spatial_transform):
        raise RuntimeError("E15 cached target transform disagrees with the evaluated clip")
    model, _ = build_model(config, allow_random_initialization=True)
    checkpoint, checkpoint_sha = read_checkpoint(
        args.checkpoint, expected_sha256=args.checkpoint_sha256
    )
    config_sha = configuration_digest(config.as_dict())
    loaded = load_inference_state(model, checkpoint, config_sha256=config_sha, seed=17)
    if isinstance(loaded, MCFormer | AuxiliaryFormer):
        rgb_model = loaded.rgb_model
    elif isinstance(loaded, VideoClassifier):
        rgb_model = loaded
    else:
        raise RuntimeError("Qualitative checkpoint did not produce an RGB classifier")
    if not isinstance(rgb_model.backbone, VideoSwinTinyBackbone):
        raise RuntimeError("E15 requires the Video Swin-T MC-Former backbone")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rgb_model.eval().to(device)
    video = torch.as_tensor(sample.video, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        features, attention = final_stage_attention_rollout(rgb_model.backbone, video)
        logits = rgb_model.classifier(features.pooled)
        probabilities = logits.softmax(dim=1)[0]
    target = cached["sample_target"]
    selected = select_qualitative_frames(target)
    metadata = {
        "schema_version": 1,
        "sample_id": sample_id,
        "class_id": drink_id,
        "model_seed": 17,
        "selection": (
            "lexicographically first correctly classified Toyota-CS Drink-From-bottle "
            "sample with gate coverage >= 0.5"
        ),
        "selected_positions": list(selected),
        "selected_source_frame_indices": [sample.frame_indices[index] for index in selected],
        "attention": (
            "mean attention received after head-averaged residual rollout across the two "
            "final-stage Video Swin attention blocks; trilinear input-grid interpolation"
        ),
        "normalization": "independent min-max normalization per frame",
        "colormap": "viridis",
        "opacity": 0.45,
        "interpretation": "diagnostic association map; not causal evidence",
        "predicted_class": int(probabilities.argmax().item()),
        "predicted_confidence": float(probabilities.max().item()),
        "coupling_at_selected_positions": [
            float(target["coupling"]["target"][index]) for index in selected
        ],
        "trajectories": {
            "hand": target["hand"],
            "object": target["object_trajectory"],
        },
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "manifest_sha256": sha256_file(args.manifest),
        "protocol_sha256": sha256_file(args.protocol_split),
        "predictions_sha256": sha256_file(args.predictions),
        "cache_key": cache.cache_key,
    }
    render_qualitative_figure(
        rgb_from_normalized(sample.video),
        np.asarray(attention.float().cpu()),
        selected,
        metadata=metadata,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
