"""Precompute training-only pose/object observations and validate coupling targets."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from mcformer.auxiliary.cache import (
    ObservationCache,
    configuration_digest,
)
from mcformer.auxiliary.detection import ByteTrackSettings, YOLOv8ByteTrack
from mcformer.auxiliary.pipeline import TargetSettings, build_sample_target
from mcformer.auxiliary.pose import HRNetWholeBodyEstimator
from mcformer.auxiliary.preprocess import extract_observation_bundle
from mcformer.config import ResolvedConfig, load_config
from mcformer.data.manifest import Manifest
from mcformer.data.protocols import ProtocolSplit
from mcformer.data.sampling import sample_frame_indices
from mcformer.data.transforms import make_spatial_transform
from mcformer.device import resolve_device
from mcformer.logging_utils import configure_logging
from mcformer.reproducibility import sha256_file, write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol-split", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pose-source",
        choices=("hrnet", "ntu_projected_3d", "projected_3d_json"),
        default="hrnet",
    )
    parser.add_argument("--pose-config")
    parser.add_argument("--pose-checkpoint")
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument(
        "--mode",
        choices=("native_frames", "paper_cost_mode"),
        default="native_frames",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--cost-trial",
        type=int,
        help="Record raw E14 stage timings for this 1-based independent trial",
    )
    parser.add_argument(
        "--source-fingerprint",
        choices=("sha256", "manifest-only"),
        default="sha256",
        help="Hash training media for cache identity or trust only the manifest hash",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def _target_settings(config: ResolvedConfig) -> TargetSettings:
    get = config.get
    return TargetSettings(
        pose_confidence=float(get("auxiliary.pose_confidence")),
        pose_max_gap=int(get("auxiliary.pose_max_gap")),
        gaussian_sigma_frames=float(get("auxiliary.gaussian_sigma_frames")),
        object_max_gap=int(get("auxiliary.object_max_gap")),
        minimum_track_coverage=float(get("auxiliary.minimum_track_coverage")),
        minimum_track_mean_confidence=float(get("auxiliary.minimum_track_mean_confidence")),
        distance_threshold=float(get("auxiliary.distance_threshold")),
        epsilon=float(get("auxiliary.epsilon")),
    )


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.set)
    manifest = Manifest.read_jsonl(args.manifest)
    split = ProtocolSplit.read_json(args.protocol_split, manifest)
    device = resolve_device(args.device)
    logger = configure_logging(log_file=Path(args.output) / "preprocess.jsonl")

    pose_backend = None
    pose_identity: dict[str, str | None] = {"source": args.pose_source}
    if args.pose_source == "hrnet":
        if not args.pose_config or not args.pose_checkpoint:
            build_parser().error("HRNet requires --pose-config and --pose-checkpoint")
        pose_backend = HRNetWholeBodyEstimator(
            config_path=args.pose_config,
            checkpoint_path=args.pose_checkpoint,
            device=device,
            confidence_threshold=float(config.get("auxiliary.pose_confidence")),
        )
        pose_identity.update(
            {
                "config_sha256": sha256_file(args.pose_config),
                "checkpoint_sha256": sha256_file(args.pose_checkpoint),
            }
        )
    detector_identity = {
        "checkpoint_sha256": sha256_file(args.detector_checkpoint),
        "ultralytics_version": _version("ultralytics"),
        "confidence": config.get("auxiliary.detector_confidence"),
        "nms_iou": config.get("auxiliary.detector_nms_iou"),
        "tracker_high": config.get("auxiliary.tracker_high_threshold"),
        "tracker_low": config.get("auxiliary.tracker_low_threshold"),
        "tracker_match": config.get("auxiliary.tracker_match_threshold"),
        "tracker_buffer": config.get("auxiliary.tracker_buffer_frames"),
    }
    if args.pose_source == "hrnet":
        pose_identity["mmpose_version"] = _version("mmpose")
        pose_identity["mmcv_version"] = _version("mmcv")
    root = Path(args.data_root).expanduser().resolve()
    source_hashes: dict[str, dict[str, str]] | None = None
    if args.source_fingerprint == "sha256":
        source_hashes = {}
        for sample_id in split.train:
            record = manifest.by_id(sample_id)
            values = {"rgb": sha256_file(_resolve(root, record.rgb_path))}
            if args.pose_source != "hrnet" and record.skeleton_path:
                values["skeleton"] = sha256_file(_resolve(root, record.skeleton_path))
            if args.pose_source == "projected_3d_json" and record.calibration_path:
                values["calibration"] = sha256_file(_resolve(root, record.calibration_path))
            source_hashes[sample_id] = values
    identity = {
        "schema_version": 1,
        "mcformer_version": _version("mc-former") or "0.1.0",
        "manifest_sha256": sha256_file(args.manifest),
        "protocol_sha256": sha256_file(args.protocol_split),
        "source_fingerprint_mode": args.source_fingerprint,
        "source_hashes": source_hashes,
        "mode": args.mode,
        "pose": pose_identity,
        "detector": detector_identity,
        "target": config.get("auxiliary"),
        "sampling": {
            "num_frames": config.get("data.num_frames"),
            "stride": config.get("data.temporal_stride"),
            "input_size": config.get("data.input_size"),
            "resize_short_side": config.get("data.resize_short_side"),
            "evaluation_views": config.get("data.evaluation_views"),
        },
    }
    cache_key = configuration_digest(identity)
    cache = ObservationCache(args.output, cache_key=cache_key)
    cache.initialize(identity)
    detector = YOLOv8ByteTrack(
        checkpoint_path=args.detector_checkpoint,
        device=device,
        detector_confidence=float(config.get("auxiliary.detector_confidence")),
        nms_iou=float(config.get("auxiliary.detector_nms_iou")),
        settings=ByteTrackSettings(
            high_threshold=float(config.get("auxiliary.tracker_high_threshold")),
            low_threshold=float(config.get("auxiliary.tracker_low_threshold")),
            match_threshold=float(config.get("auxiliary.tracker_match_threshold")),
            buffer_frames=int(config.get("auxiliary.tracker_buffer_frames")),
        ),
    )
    target_settings = _target_settings(config)
    processed = skipped = failed = 0
    eligible_track_samples = 0
    coverage: list[float] = []
    errors: list[dict[str, str]] = []
    cost_samples: list[dict[str, object]] = []
    cost_frame_count = 0
    if args.cost_trial is not None and args.cost_trial <= 0:
        build_parser().error("--cost-trial must be a positive integer")
    for sample_id in split.train:
        if cache.contains(sample_id) and cache.contains_target(sample_id) and not args.overwrite:
            cached_target = cache.read_target(sample_id)["sample_target"]
            cached_coverage = float(cached_target["coupling"]["coverage"])
            coverage.append(cached_coverage)
            eligible_track_samples += cached_target["object_trajectory"] is not None
            skipped += 1
            continue
        record = manifest.by_id(sample_id)
        try:
            stage_timings: dict[str, float] = {}
            raw_cached = cache.contains(sample_id) and not args.overwrite
            if raw_cached:
                bundle = cache.read(sample_id)
            else:
                bundle = extract_observation_bundle(
                    record,
                    root=args.data_root,
                    object_backend=detector,
                    pose_backend=pose_backend,
                    pose_source=args.pose_source,
                    mode=args.mode,
                    clip_length=int(config.get("data.num_frames")),
                    stride=int(config.get("data.temporal_stride")),
                    stage_timings=stage_timings if args.cost_trial is not None else None,
                )
            temporal = sample_frame_indices(
                record.num_frames or 0,
                clip_length=int(config.get("data.num_frames")),
                stride=int(config.get("data.temporal_stride")),
                training=False,
            )
            spatial = make_spatial_transform(
                width=record.width or 0,
                height=record.height or 0,
                training=False,
                resize_short_side=int(config.get("data.resize_short_side")),
                output_size=int(config.get("data.input_size")),
            )
            target_start = time.perf_counter()
            target = build_sample_target(
                bundle,
                frame_indices=temporal.indices,
                spatial_transform=spatial,
                settings=target_settings,
            )
            if args.cost_trial is not None:
                stage_timings["target_generation_seconds"] = time.perf_counter() - target_start
            if not raw_cached:
                cache.write(bundle, overwrite=args.overwrite)
            cache.write_target(
                sample_id,
                {
                    "frame_indices": list(temporal.indices),
                    "spatial_transform": asdict(spatial),
                    "sample_target": target.as_mapping(),
                },
                overwrite=args.overwrite,
            )
            coverage.append(target.coupling.coverage)
            eligible_track_samples += target.object_trajectory is not None
            processed += 1
            if args.cost_trial is not None and not raw_cached:
                cost_frame_count += len(bundle.frame_indices)
                cost_samples.append(
                    {
                        "trial": args.cost_trial,
                        "sample_id": sample_id,
                        "frames": len(bundle.frame_indices),
                        **stage_timings,
                    }
                )
            logger.info(
                "Preprocessed %s",
                sample_id,
                extra={"event": "sample_complete", "value": target.coupling.coverage},
            )
        except Exception as error:
            failed += 1
            errors.append({"sample_id": sample_id, "error": repr(error)})
            logger.exception("Failed preprocessing %s", sample_id)
            if not args.continue_on_error:
                raise
    summary = {
        "cache_key": cache_key,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "mean_gate_coverage": sum(coverage) / len(coverage) if coverage else None,
        "eligible_track_samples": eligible_track_samples,
        "eligible_track_rate": (eligible_track_samples / len(coverage) if coverage else None),
        "zero_gate_samples": sum(value == 0 for value in coverage),
        "no_target_clip_rate": (
            sum(value == 0 for value in coverage) / len(coverage) if coverage else None
        ),
        "errors": errors,
    }
    write_json_atomic(Path(args.output) / "summary.json", summary)
    if args.cost_trial is not None:
        timing_path = Path(args.output) / f"cost_trial_{args.cost_trial:02d}.jsonl"
        with timing_path.open("w", encoding="utf-8") as handle:
            for item in cost_samples:
                handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
        inventory = {
            "trial": args.cost_trial,
            "samples": len(cost_samples),
            "frames": cost_frame_count,
            "cache_bytes": sum(
                path.stat().st_size for path in Path(args.output).rglob("*.json.gz")
            ),
        }
        write_json_atomic(
            Path(args.output) / f"cost_trial_{args.cost_trial:02d}_inventory.json", inventory
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
