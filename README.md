# MC-Former

[![CI](https://github.com/Pourya-Moghadam/MC-Former/actions/workflows/ci.yml/badge.svg)](https://github.com/Pourya-Moghadam/MC-Former/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-31110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reproducible PyTorch implementation accompanying **“MC-Former: Interaction-Aware Action
Recognition via Hand-Object Motion Coupling.”**

The data, model, training, and evaluation phases are complete: manifests, official protocols, RGB
loading/transforms, auxiliary preprocessing, three video backbones, MCIM, optimization,
distributed training, exact resume, metrics, verified checkpoint loading, and RGB-only export are
implemented and tested. Paper-level statistical analyses, hardware benchmarks, and figure
generation are provided as artifact-producing commands; dataset/GPU-dependent values are never
fabricated in the source tree. Their behavior is fixed in
[`REPRODUCIBILITY_SPEC.md`](REPRODUCIBILITY_SPEC.md). The manuscript-to-experiment map is
[`PAPER_EXPERIMENTS.md`](PAPER_EXPERIMENTS.md).

This is a code-only release. Dataset media, restricted annotations, third-party initialization
weights, trained checkpoints, and manuscript result artifacts are not redistributed. The commands
fail closed when a required local artifact or SHA-256 identity is missing.

## Method

MC-Former trains an RGB video transformer with a temporary Motion Coupling Induction Module
(MCIM). Training-only hand and object trajectories define a proximity-gated directional
co-motion target. MCIM regresses that target from temporal video features. MCIM and all auxiliary
pose/detection components are removed for deployment, so inference accepts RGB only.

## Installation

The reference GPU environment is Python 3.11.10, PyTorch 2.4.1, torchvision 0.19.1, and CUDA 12.1:

```bash
conda env create -f environment.yml
conda activate mcformer
python -m pip install -e .
```

For a platform-specific CPU or CUDA installation, create Python 3.11 and install the exact direct
versions from `requirements/base.txt`, then install the desired optional files:

```bash
python -m pip install -r requirements/dev.txt
python -m pip install -r requirements/preprocessing.txt
python -m pip install -r requirements/pose.txt
python -m pip install -e . --no-deps
```

All auxiliary checkpoints must be existing local files. The software never downloads weights.
Runtime SHA-256 hashes and package/configuration versions identify them in cache metadata.

## Configuration and run provenance

Configurations are hierarchical YAML mappings. `defaults` lists relative parent files; dotted
overrides accept JSON values:

```bash
mcformer-show-config \
  --config configs/experiment/e01_toyota_cs_mcformer.yaml \
  --set data.protocol="cv1"
```

You can initialize an immutable run directory independently for provenance inspection:

```bash
mcformer-init-run \
  --config configs/experiment/e01_toyota_cs_mcformer.yaml \
  --output outputs/e01-toyota-cs-seed17 \
  --repository . \
  --set reproducibility.seeds='[17]'
```

This records the resolved configuration, environment, command, dependency and hardware versions,
Git state, and structured log. Existing run directories are not overwritten.

## Dataset preparation

Original datasets and restricted annotations are not redistributed. Input contracts are:

- NTU RGB+D 60/120: official RGB files; official `.skeleton` files for the 3D target. File names
  supply subject, camera, setup, repetition, and action metadata.
- Toyota Smarthome: RGB files plus a portable CSV with `sample_id`, `rgb_path`, `label_id`, and the
  optional manifest fields documented by `SampleRecord`. Each protocol also requires an official
  JSON object with complete `train` and `test` sample-ID arrays.
- Portable projected 3D pose: a pose JSON and 3x4 calibration JSON using the exact contract in
  `REPRODUCIBILITY_SPEC.md`; their paths occupy `skeleton_path` and `calibration_path`.

NTU preparation requires an explicit missing-sample file, including when it is intentionally empty:

```bash
mcformer-prepare-manifest \
  --dataset ntu_rgbd_60 \
  --data-root /data/ntu60/rgb \
  --skeleton-root /data/ntu60/skeleton \
  --label-map metadata/labels/ntu60.json \
  --missing-samples /data/ntu60/missing_samples.txt \
  --output metadata/manifests/ntu60.jsonl

mcformer-prepare-protocol \
  --manifest metadata/manifests/ntu60.jsonl \
  --protocol cs \
  --output metadata/protocols/ntu60_cs.json
```

Toyota manifest and protocol construction:

```bash
mcformer-prepare-manifest \
  --dataset toyota_smarthome \
  --data-root /data/toyota \
  --annotations /data/toyota/annotations.csv \
  --label-map metadata/labels/toyota.json \
  --output metadata/manifests/toyota.jsonl

mcformer-prepare-protocol \
  --manifest metadata/manifests/toyota.jsonl \
  --protocol cs \
  --official-split /data/toyota/cs.json \
  --output metadata/protocols/toyota_cs.json
```

The official test partition stays intact. Validation is the deterministic seed-2026, group-aware
10% split defined in the release specification. Audit paths, split integrity, and video metadata:

```bash
mcformer-validate-data \
  --manifest metadata/manifests/ntu60.jsonl \
  --protocol-split metadata/protocols/ntu60_cs.json \
  --data-root /data/ntu60/rgb \
  --decode-metadata
```

## Auxiliary preprocessing

The scientific `native_frames` mode extracts observations from every decoded frame so seeded
random training clips can be reconstructed. `paper_cost_mode` is restricted to the centered
32-frame cost benchmark. Both are part of the immutable cache identity.

```bash
mcformer-preprocess \
  --config configs/experiment/e01_toyota_cs_mcformer.yaml \
  --manifest metadata/manifests/toyota.jsonl \
  --protocol-split metadata/protocols/toyota_cs.json \
  --data-root /data/toyota \
  --output outputs/cache/toyota_cs_2d \
  --pose-source hrnet \
  --pose-config /checkpoints/hrnet_w48_wholebody_384x288.py \
  --pose-checkpoint /checkpoints/hrnet_w48_wholebody.pth \
  --detector-checkpoint /checkpoints/yolov8x.pt \
  --mode native_frames
```

Use `--pose-source ntu_projected_3d` for official NTU `.skeleton` color-plane wrist coordinates,
or `--pose-source projected_3d_json` for the portable 3D/calibration contract. Source media,
skeletons, calibration, checkpoints, protocols, configuration, backend versions, and target
settings are hashed into the cache identity.

`VideoClipDataset` deterministically samples clips from seed, epoch, and sample ID. It returns
normalized `T,C,H,W` arrays and applies identical spatial geometry to RGB and cached observations.

## Models

The model package provides:

- pinned torchvision 0.19.1 adapters for Video Swin-T and MViTv2-S;
- a self-contained divided-space-time TimeSformer-Base;
- final- or intermediate-stage temporal tokens;
- the exact `768 -> 384 -> 1` MCIM (295,681 parameters);
- globally reduced masked MSE and combined classification/coupling loss;
- local checkpoint loading that requires a matching SHA-256; and
- physical RGB-only export with no MCIM state or dependency.

All public models accept `B,T,C,H,W` RGB tensors. `build_model` refuses silent random
initialization for paper configurations: pass a local checkpoint and its SHA-256. Unit tests may
explicitly set `allow_random_initialization=True`. The backbone's pooled representation alone feeds
the classifier; MCIM receives temporal tokens through a separate training-only branch.

## Training and evaluation

Run one configured seed with an explicit manifest, split, data root, verified ImageNet-1K
initialization checkpoint, and precomputed auxiliary cache:

```bash
python scripts/train.py \
  --config configs/experiment/e01_toyota_cs_mcformer.yaml \
  --manifest metadata/manifests/toyota.jsonl \
  --protocol-split metadata/protocols/toyota_cs.json \
  --data-root /data/toyota \
  --cache outputs/cache/toyota_cs_2d \
  --initialization-checkpoint /checkpoints/swin_t_imagenet1k.pth \
  --initialization-sha256 SHA256 \
  --seed 17 \
  --output outputs/e01-toyota-cs-seed17
```

The configured global batch is preserved exactly with per-device batches and gradient
accumulation. Launch multi-GPU runs with `torchrun`; non-padding evaluation samplers ensure every
validation/test sample is counted once. Checkpoints include model, optimizer, schedule, AMP, and
per-rank RNG states. Resume requires the recorded digest:

```bash
python scripts/train.py [THE SAME ARGUMENTS] \
  --resume outputs/e01-toyota-cs-seed17/checkpoints/last.pt \
  --resume-sha256 SHA256_FROM_LAST_JSON
```

Evaluate either the complete training checkpoint or the exported RGB-only checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/experiment/e01_toyota_cs_mcformer.yaml \
  --manifest metadata/manifests/toyota.jsonl \
  --protocol-split metadata/protocols/toyota_cs.json \
  --data-root /data/toyota \
  --checkpoint outputs/e01-toyota-cs-seed17/checkpoints/rgb_only.pt \
  --checkpoint-sha256 SHA256_FROM_RGB_ONLY_JSON \
  --partition test \
  --seed 17 \
  --output outputs/e01-toyota-cs-seed17/test
```

Evaluation writes aggregate and per-class metrics, a confusion-matrix CSV, and sample-sorted JSONL
containing labels, predictions, logits, probabilities, and padding-validity counts. Training writes
epoch history, target coverage, best-validation/last/final checkpoints, and an RGB-only deployment
checkpoint. No paper result is embedded in these generators.

## Derived paper analyses (E02–E05)

The analysis command consumes the sample-sorted `predictions.jsonl` files produced above. It
requires all three paper seeds, verifies exact official-test membership and labels against the
manifest/split, then verifies pairing across every method and seed.

Generate E02 repeated-run statistics and the paper-specified paired bootstrap. Use `accuracy` for
NTU and `mca` for Toyota; incompatible dataset/metric combinations are rejected:

```bash
python scripts/analyze.py statistics \
  --manifest metadata/manifests/toyota.jsonl \
  --protocol-split metadata/protocols/toyota_cs.json \
  --baseline 17=outputs/swin-seed17/test/predictions.jsonl \
  --baseline 29=outputs/swin-seed29/test/predictions.jsonl \
  --baseline 43=outputs/swin-seed43/test/predictions.jsonl \
  --method 17=outputs/mcformer-seed17/test/predictions.jsonl \
  --method 29=outputs/mcformer-seed29/test/predictions.jsonl \
  --method 43=outputs/mcformer-seed43/test/predictions.jsonl \
  --metric mca \
  --output outputs/paper/e02/toyota_cs
```

Generate E03 subset results for any Toyota protocol. Add only the tasks appropriate to the method
pair: E04 `pairs` compares pi-ViT with MC-Former on Toyota-CS, while E05 `confusion` compares Video
Swin with MC-Former on Toyota-CS.

```bash
python scripts/analyze.py diagnostics \
  --manifest metadata/manifests/toyota.jsonl \
  --protocol-split metadata/protocols/toyota_cs.json \
  --subsets metadata/subsets/toyota_diagnostics.json \
  --baseline 17=outputs/swin-seed17/test/predictions.jsonl \
  --baseline 29=outputs/swin-seed29/test/predictions.jsonl \
  --baseline 43=outputs/swin-seed43/test/predictions.jsonl \
  --method 17=outputs/mcformer-seed17/test/predictions.jsonl \
  --method 29=outputs/mcformer-seed29/test/predictions.jsonl \
  --method 43=outputs/mcformer-seed43/test/predictions.jsonl \
  --task subset --task confusion \
  --output outputs/paper/e03_e05/toyota_cs
```

E02 outputs seed metrics, mean/sample SD, all bootstrap draws, percentile CI metadata, classwise
gains, classwise median/IQR, LaTeX rows, and content-addressed provenance. E03–E05 output subset
metrics, pair sample lists and 31-way-error-preserving confusion tables, selected-class confusion
CSVs, PNG/PDF figures, LaTeX rows, and provenance. Manuscript numbers are never used as inputs.

Experiments E01–E16, expected artifacts, and manuscript table/figure mappings are cataloged in the
implementation plan. All formerly missing choices—including model shapes, optimization,
statistics, corruption, visualization, and benchmark protocols—are fixed by the release
specification. Manuscript result values are validation targets only and are never executable
constants.

## Controlled experiments (E06–E12)

The frozen 45-variant catalog is `configs/sweep/e06_e12.json`. Validate every inherited config,
override, seed, digest, and E12 corruption before launching jobs:

```bash
mcformer-show-sweep --matrix configs/sweep/e06_e12.json \
  > outputs/paper/e06_e12/resolved_matrix.json
```

Each displayed variant contains an absolute base `config` and exact `overrides`. Pass every
override as one `--set` argument to the documented training command. E06 supports spatial-only,
ungated temporal, gated temporal, and separate spatial-plus-temporal heads. E07 fixes the four
loss weights, E08 the three insertion stages, E09 the three matched backbone pairs, E10 the
spatial and fixed 256-D hallucination auxiliaries, and E11 all fifteen one-factor preprocessing
settings. Auxiliary heads never feed classifier features and are physically removed from the
RGB-only checkpoint, as fixed in `REPRODUCIBILITY_SPEC.md`.

E11 settings that affect pose, detector, tracker, or target construction require a separately
preprocessed cache using the same resolved override. Cache identity prevents accidental reuse.
For each E12 condition, materialize one corruption realization shared by model seeds:

```bash
python scripts/corrupt_cache.py \
  --manifest metadata/manifests/toyota.jsonl \
  --protocol-split metadata/protocols/toyota_cs.json \
  --source-cache outputs/cache/toyota_cs_2d \
  --corruption wrist_noise \
  --value 0.02 \
  --seed 2026 \
  --output outputs/cache/e12/wrist_noise_0.02
```

The corrupted cache records its source-content digest, exact changed frames/detections/tracks,
selected/replacement track IDs, fixed sample selection, and resulting cache key. Only training
partition bundles are written; validation and test remain RGB-only and uncorrupted.

After evaluating all three seeds, aggregate any E06–E12 table without embedded paper values:

```bash
python scripts/aggregate_sweep.py \
  --manifest metadata/manifests/toyota.jsonl \
  --protocol-split metadata/protocols/toyota_cs.json \
  --metric mca \
  --subsets metadata/subsets/toyota_diagnostics.json \
  --run spatial_only:17=outputs/e06/spatial/17/test/predictions.jsonl \
  --run spatial_only:29=outputs/e06/spatial/29/test/predictions.jsonl \
  --run spatial_only:43=outputs/e06/spatial/43/test/predictions.jsonl \
  --history spatial_only:17=outputs/e06/spatial/17/history.json \
  --history spatial_only:29=outputs/e06/spatial/29/history.json \
  --history spatial_only:43=outputs/e06/spatial/43/history.json \
  --output outputs/paper/e06
```

Repeat `--run` and `--history` for every variant in a table. Aggregation requires exactly seeds
17/29/43, exact official-test membership, paired labels, and matching history variants. It writes
per-seed values, mean/sample SD, manipulation-subset mCA, target coverage, LaTeX rows, hashes, and
provenance. Training run metadata also records deployed, total, and per-head parameter counts plus
the insertion-stage token dimension.

## Efficiency and figures (E13–E16)

E13 benchmarks the physically exported RGB graph using the fixed FP32, batch-one, 50-warm-up,
200-iteration, five-run protocol. It refuses CPU execution and non-reference hardware unless the
latter is explicitly acknowledged:

```bash
python scripts/benchmark_inference.py \
  --config configs/experiment/e01_toyota_cs_mcformer.yaml \
  --checkpoint outputs/toyota-cs/17/checkpoints/rgb_only.pt \
  --checkpoint-sha256 <sha256> \
  --output outputs/paper/e13/mcformer
```

E14 preprocessing runs accept `--mode paper_cost_mode --cost-trial N`; run independent trials
1–5 into separate cache directories. Training histories now contain CUDA-synchronized
`training_seconds` and `peak_training_memory_bytes` for the training epoch only. A cost run uses
four epochs: epoch zero is discarded as warm-up and epochs 1–3 are measured.

Run `scripts/profile_training.py --config <config> --output <profile.json>` once for the matched
baseline and MC-Former configs on the reference GPU; pass both profiles to the aggregator to
report full and training-only parameter/MAC deltas.

```bash
python scripts/aggregate_cost.py \
  --preprocessing-trial outputs/cost/trial1/cost_trial_01.jsonl \
  --preprocessing-trial outputs/cost/trial2/cost_trial_02.jsonl \
  --preprocessing-trial outputs/cost/trial3/cost_trial_03.jsonl \
  --preprocessing-trial outputs/cost/trial4/cost_trial_04.jsonl \
  --preprocessing-trial outputs/cost/trial5/cost_trial_05.jsonl \
  --cache-inventory outputs/cost/trial1/cost_trial_01_inventory.json \
  --cache-inventory outputs/cost/trial2/cost_trial_02_inventory.json \
  --cache-inventory outputs/cost/trial3/cost_trial_03_inventory.json \
  --cache-inventory outputs/cost/trial4/cost_trial_04_inventory.json \
  --cache-inventory outputs/cost/trial5/cost_trial_05_inventory.json \
  --baseline-history outputs/cost/baseline/history.json \
  --mcformer-history outputs/cost/mcformer/history.json \
  --output outputs/paper/e14/cost.json
```

E15 consumes seed-17 predictions, the matching auxiliary cache, and the verified checkpoint. It
selects the sample and four positions automatically, captures actual final-stage Swin attention,
and writes source panels, trajectories, normalized activation arrays, PNG/PDF, and provenance:

```bash
python scripts/visualize_interaction.py \
  --config configs/experiment/e01_toyota_cs_mcformer.yaml \
  --manifest metadata/manifests/toyota.jsonl \
  --protocol-split metadata/protocols/toyota_cs.json \
  --data-root /path/to/toyota --cache outputs/cache/toyota_cs_2d \
  --predictions outputs/toyota-cs/17/test/predictions.jsonl \
  --checkpoint outputs/toyota-cs/17/checkpoints/final.pt \
  --checkpoint-sha256 <sha256> --output outputs/paper/e15
```

E16 first extracts pooled classifier-input features from both seed-17 checkpoints, then validates
identical sample IDs and labels before the fixed joint PCA-50/t-SNE projection:

```bash
python scripts/extract_features.py --config <config> --manifest <manifest> \
  --protocol-split <split> --data-root <root> --checkpoint <checkpoint> \
  --checkpoint-sha256 <sha256> --seed 17 --output outputs/features/video_swin
python scripts/project_tsne.py \
  --baseline-features outputs/features/video_swin/features.npz \
  --mcformer-features outputs/features/mcformer/features.npz \
  --manifest metadata/manifests/toyota.jsonl \
  --subsets metadata/subsets/toyota_diagnostics.json \
  --output outputs/paper/e16
```

## Checks

```bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy src/mcformer
pytest
python -m build
```

A dependency-light suite works directly from checkout:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m mcformer.cli.show_config \
  --config configs/experiment/e01_toyota_cs_mcformer.yaml
```

## License and citation

The code is released under the [MIT License](LICENSE). Dataset and model-weight terms remain those
of their respective providers. Citation metadata is in [`CITATION.cff`](CITATION.cff). See the
intended uses and limitations in [`MODEL_CARD.md`](MODEL_CARD.md), contribution guidance in
[`CONTRIBUTING.md`](CONTRIBUTING.md), and public-release procedure in
[`PUBLISHING.md`](PUBLISHING.md).
