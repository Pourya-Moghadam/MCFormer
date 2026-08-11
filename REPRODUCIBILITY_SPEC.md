# MC-Former release decisions

This file is the normative specification wherever the manuscript is silent or internally
inconsistent. These are release decisions, not claims about undocumented historical runs. A run
that follows this file is an MC-Former release reproduction; deviations must be recorded in its
resolved configuration and provenance.

## Environment, artifacts, and data

- The supported reference environment is Python 3.11.10, PyTorch 2.4.1, torchvision 0.19.1,
  CUDA 12.1, and the exact direct dependency versions in `requirements/`. `environment.yml` is
  the GPU environment definition. Every run records the fully resolved transitive environment.
- Source code is MIT licensed. Original datasets are never redistributed. Released checkpoints,
  predictions, and aggregate caches go in a versioned Zenodo record; raw media, skeletons,
  annotations, and per-frame derived pose/detection caches remain local because upstream dataset
  terms govern them.
- Checkpoints are local and are never downloaded implicitly. Their runtime SHA-256, local file
  name, backend package version, and configuration hash are the artifact identity. A changed byte
  creates a different cache/run identity; no mutable URL is treated as identity.
- Users provide the official NTU RGB+D 60/120 or Toyota Smarthome release. NTU manifest creation
  requires an explicit missing-sample text file; an intentionally empty file is allowed. Toyota
  requires an explicit official train/test JSON for each CS/CV1/CV2 protocol. Manifests and split
  files are content hashed, and the validator rejects missing IDs, overlaps, absent classes, and
  incomplete membership. Consequently there is no silent fallback to guessed release counts.
- Dataset label IDs are zero-based and contiguous. Diagnostic Toyota labels are stored by their
  canonical names in `metadata/subsets/toyota_diagnostics.json` and resolved exactly after
  case/punctuation normalization; missing or ambiguous labels are fatal.

## Splits and sampling

- Official test partitions are untouched. Ten percent of the official training partition becomes
  validation with seed 2026. Complete subjects are held out for CS and complete cameras/setups for
  CV/CSet when all classes remain represented; otherwise the deterministic class-stratified clip
  fallback is used and recorded. The final epoch is the reported checkpoint. Best-validation is
  saved only as a diagnostic. Test is evaluated once after configuration freeze.
- Decode presentation-order native frames. A clip has indices `start + 2*i`, `i=0..31`. Training
  starts uniformly over valid starts using the run seed, epoch, and sample ID. Evaluation uses one
  centered temporal view. Short clips repeat their final frame and emit a padding mask.
- Resize the shorter side to 256 with bilinear interpolation. Training uses random resized crop
  scale `[0.8,1.0]`, aspect ratio `[0.75,1.3333333333333333]`, then horizontal flip probability
  0.5. Evaluation uses one 224 center crop. Normalize with ImageNet mean/std. No other RGB
  augmentation, repeated augmentation, label smoothing, or test ensemble is used.
- Apply the same temporal and spatial transforms to RGB, wrists, and boxes. Geometry outside the
  crop becomes invalid; boxes are clipped. Targets are constructed after geometry transforms.

## Pose, detection, and coupling

- Standard targets use local MMPose 1.3.2 HRNet-W48 COCO WholeBody 384x288 weights. Wrist
  confidence is 0.30. Framewise people are greedily associated by wrist centroid up to 0.25 of
  the image diagonal. The actor with greatest total valid wrist motion is selected; then its wrist
  with greatest motion is selected. Ties use lexical actor ID and then the left wrist.
- NTU 3D targets canonically use `colorX/colorY` in the official `.skeleton` files. Toyota and any
  other projected-3D input use `projected_3d_json`: `skeleton_path` contains frames, actor IDs, and
  nullable `left_wrist`/`right_wrist` entries of `{"xyz":[x,y,z],"confidence":c}`;
  `calibration_path` contains a finite 3x4 `projection_matrix`. Projection is homogeneous `P[X,Y,Z,1]`
  with division by its third coordinate. Coordinates use the source dataset's calibration units.
- Pose gaps of at most five frames are linearly interpolated, then valid segments are smoothed by
  a Gaussian with sigma one frame. Longer and edge gaps remain invalid.
- YOLOv8x/COCO runs on native frames, excludes `person`, uses detector confidence 0.25 and NMS IoU
  0.70. ByteTrack receives detections down to 0.10 for association; only observations at least
  0.25 are cached. Settings are high 0.50, low 0.10, new-track 0.50, match 0.80, buffer 30,
  score fusion enabled, no minimum-area/aspect filter, and tracker state reset per clip.
- Scientific preprocessing uses every native frame. `paper_cost_mode` is used only for the fixed
  centered 32-frame cost benchmark. The mode is part of the cache key.
- Eligible object tracks cover at least 50% of sampled frames with mean confidence at least 0.25.
  Selection minimizes median wrist-to-nearest-box distance, then maximizes confidence, then uses
  the lowest track ID. Object gaps up to three frames are interpolated.
- Gate distance uses wrist to selected box center after transformation and normalization by the
  original frame diagonal. The threshold is 0.15. Cosine epsilon is `1e-6`; stationary motion has
  target zero. Invalid frames are masked. Clips without a valid gate contribute classification
  loss and exactly zero coupling loss.

## Model and optimization contract

- Video Swin-T uses patch/tubelet `(2,4,4)`, window `(8,7,7)`, embed dimension 96, depths
  `(2,2,6,2)`, heads `(3,6,12,24)`, MLP ratio 4, QKV bias, drop/attention-drop 0, stochastic depth
  0.2, layer norm, global average pooling, and truncated-normal classifier initialization with
  standard deviation 0.02. Initialization is the official ImageNet-1K Swin-T 224 checkpoint.
- TimeSformer is Base, 16x16 patch, 768 dimensions, 12 layers, 12 heads, divided space-time
  attention. MViTv2 is the Small variant. Each emits the pre-classifier global representation.
- MCIM is attached after stage four. Spatially averaged tokens pass through biased linear layers
  `768 -> 384 -> 1`, ReLU, and no normalization/dropout. Scalar temporal predictions are linearly
  interpolated to 32 positions with aligned endpoints. This module has 295,681 parameters. The
  manuscript's 0.10M entry is retained as a manuscript value but is not used to alter architecture.
- The total loss is cross-entropy plus `lambda=1` times masked MSE. MSE is summed over all gated
  positions in the global distributed batch and divided by gate count plus `1e-6`; zero gates
  produce an exact differentiable zero.
- Train 50 epochs with global batch 16, AdamW `(beta1,beta2)=(0.9,0.999)`, epsilon `1e-8`, learning
  rate `1e-4`, weight decay 0.05, norm-5 clipping, FP16 AMP with dynamic scaling, and no LR scaling
  or layer-wise decay. Use gradient accumulation to preserve global batch 16. Warm up linearly
  from zero for five epochs, then cosine-decay to zero. Seeds are 17, 29, and 43.

The model graph, MCIM, losses, checkpoint validation, and RGB-only export implement this contract.
Training and evaluation orchestration implement these frozen decisions. E02--E05 repeated-run,
bootstrap, subset, pair, and confusion analyses are implemented from frozen prediction artifacts.
E06--E12 configurable auxiliary heads, frozen experiment matrix, corruption caches, and sweep
aggregation are implemented. E13--E16 provide measured benchmark, preprocessing-cost,
final-stage-attention, feature-extraction, and joint-projection artifact generators. Reference
numbers remain absent until those commands run with the required datasets, checkpoints, and GPU.

## Analysis and benchmark contract

- Spatial-only auxiliary target is normalized wrist-to-selected-box-center distance with the
  mutual-validity mask. Combined spatial/temporal uses separate heads and unit loss weight each.
- The 256-D hallucination target concatenates 128 pose and 128 object values. Each half comprises
  x/y positions and first differences, independently linearly resampled to 32 values per scalar
  channel and normalized by frame diagonal. Missing values are zero after interpolation; MSE uses
  the associated validity masks. A two-layer 256-D ReLU head predicts it, and its output is not fed
  to the classifier.
- Statistics use three seed runs. Gains use seed-mean probabilities, 10,000 paired resamples,
  seed 2026, percentile 95% intervals, ordinary resampling for accuracy and class-stratified
  resampling for mCA. Standard deviations use `ddof=1`.
- Corruptions occur before interpolation and smoothing. Seed 2026 produces a fixed realization
  shared across model seeds. Noise is independent Gaussian coordinate noise; dropout is
  independent Bernoulli observation removal; track swap replaces the selected track with the
  eligible track having the next-best selection rank.
- Toyota confusion matrices use CS, seed-mean probabilities, fixed manifest label order, true-row
  normalization, and `confusion_matrix_actions` from the committed diagnostic metadata.
- The qualitative example is the lexicographically first correctly classified Toyota-CS
  `Drink.Frombottle` sample with gate coverage at least 0.5 under seed 17. Frames are first gated,
  first positive-coupling, maximum-coupling, and last gated frame. Use final-stage attention
  rollout, per-frame min-max normalization, viridis, and opacity 0.45.
- t-SNE uses Toyota-CS manipulation classes, at most 200 lexicographically selected samples per
  class, seed-17 final representations, joint baseline/MC PCA to 50 dimensions, then joint
  scikit-learn t-SNE: 2D, perplexity 30, learning rate `auto`, 1,000 iterations, PCA init,
  Euclidean metric, seed 2026.
- Efficiency uses one NVIDIA V100-SXM2 32GB, the reference environment, batch 1 FP32 for inference,
  50 warm-up and 200 timed iterations repeated five times, CUDA synchronization, and no disk I/O.
  Multiply-add counts as one FLOP. Training-cost runs use CS, global batch 16 FP16, one warm epoch
  then three timed epochs. Preprocessing uses five trials and the actual gzip-JSON cache format.
