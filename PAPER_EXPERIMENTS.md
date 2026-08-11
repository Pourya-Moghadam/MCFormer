# Paper experiment map

This file is the standalone-release index for the experiments specified in the manuscript and
implemented by this repository. Exact data, preprocessing, optimization, seed, estimator, and
artifact contracts are normative in `REPRODUCIBILITY_SPEC.md`; manuscript numbers are comparison
targets and never executable constants.

| ID | Experiment | Primary paper artifact | Release path |
|---|---|---|---|
| E01 | Main results over NTU60 CS/CV, NTU120 CS/CSet, and Toyota CS/CV1/CV2 | Main result tables | Training and evaluation CLIs |
| E02 | Three-seed statistics and paired bootstrap | Statistics table | `scripts/analyze.py statistics` |
| E03 | Toyota manipulation-heavy subset | Manipulation subset table | `scripts/analyze.py diagnostics` |
| E04 | Same-object action-pair diagnostic | Pair table | `scripts/analyze.py diagnostics` |
| E05 | Selected-class confusion | Confusion figure | `scripts/analyze.py diagnostics` |
| E06 | Spatial and temporal auxiliary variants | Ablation table | `configs/sweep/e06_e12.json` |
| E07 | Auxiliary-loss weight sweep | Weight table | `configs/sweep/e06_e12.json` |
| E08 | MCIM insertion-stage sweep | Stage table | `configs/sweep/e06_e12.json` |
| E09 | TimeSformer, Video Swin, and MViTv2 matched backbones | Backbone table | Experiment configs and sweep catalog |
| E10 | Spatial, temporal, and feature-hallucination auxiliaries | Privileged-input table | Configurable auxiliary heads |
| E11 | Preprocessing sensitivity | Sensitivity table | Sweep catalog and cache identities |
| E12 | Corrupted training-time signals | Robustness table | `scripts/corrupt_cache.py` |
| E13 | Deployed RGB-only inference cost | Complexity table | `scripts/benchmark_inference.py` |
| E14 | Preprocessing and training cost | Training-cost table | Cost/profile scripts |
| E15 | Interaction sequence and attention rollout | Qualitative figure | `scripts/visualize_interaction.py` |
| E16 | Paired classifier-input feature projection | t-SNE figure | Feature extraction and projection scripts |

The repository owns MC-Former, Video Swin, TimeSformer, and MViTv2 implementations described in
the release configuration. Literature-only or separately owned baselines require their authors'
code and checkpoints; their published numbers are not embedded or presented as reproduced runs.
