# Edge Video Anomaly Transformer (evat)

## Status: Phase 8 — Industrial Anomaly Detection (code complete, experiment not yet run)

This project is under active, phase-by-phase development. Repository
foundation, dataset ingestion (MVTec AD, YouTube-VOS), the temporal video
pipeline (segmentation baseline, mask-IoU tracker, visual feature
extraction, from-scratch Video Transformer, controlled baseline/
ablation experiment infrastructure), and an MVTec AD industrial anomaly-
detection baseline (per-category Mahalanobis-distance normality model
over pretrained CNN features, image-level scoring, coarse pixel-level
localization) are implemented. MVTec AD is used **separately** from the
YouTube-VOS video pipeline — see `docs/anomaly_task_definition.md` and
`docs/task_definition.md` for the two tasks' definitions. **No
segmentation training, tracking evaluation, feature-extraction
experiment, Transformer training, Phase 7 comparison/ablation, or Phase 8
anomaly-detection experiment has been executed yet** — see
`docs/experiments.md`.

## Purpose

A production-oriented research prototype demonstrating a full computer-vision
engineering workflow: video segmentation, object tracking, anomaly detection,
temporal representation learning, a Video Transformer implemented from
scratch, and inference optimization for constrained/edge-oriented
environments.

This project follows a strict honesty policy: no claim of performance,
deployment, or capability is made unless it has actually been implemented and
measured. See `CLAUDE.md` for the full engineering contract.

## Planned architecture

```
Video
  ↓
Frame Sampling
  ↓
Segmentation
  ↓
Object Tracking
  ↓
Visual Feature Extraction
  ↓
Temporal Sequence Construction
  ↓
Video Transformer (from scratch)
  ↓
Anomaly Detection
  ↓
Inference / Benchmarking
  ↓
Optimized Runtime
```

See `docs/architecture.md` for per-component implementation status.

## Planned phases

1. Repository and engineering foundation (this phase)
2. Dataset ingestion and validation
3. Video preprocessing
4. Segmentation
5. Object tracking
6. Visual feature extraction
7. Video Transformer from scratch
8. Baselines and ablation experiments
9. Anomaly detection
10. Inference optimization
11. End-to-end pipeline
12. Production hardening
13. CI/CD and quality gates
14. Final benchmarking and documentation

## Local vs. Google Colab

| | Local | Google Colab |
|---|---|---|
| Unit/integration tests | Yes | — |
| Linting / type checking | Yes | — |
| Small synthetic smoke tests | Yes | — |
| Model training | — | Yes |
| Large-scale inference / evaluation | — | Yes |
| GPU benchmarks | — | Yes |

Full policy in `CLAUDE.md` Sections 7–10.

## Development workflow

```bash
pip install -e ".[dev]"
make test
make lint
make format
make typecheck
make check
```

## Dataset policy

MVTec AD (CC BY-NC-SA 4.0, non-commercial) and YouTube-VOS (non-commercial
research use, official Terms of Use) are the verified, selected datasets.
DAVIS 2017 is **not used** — its licensing could not be verified from an
official source. No raw dataset is downloaded or committed to this
repository. See `docs/datasets.md`.

## Current project status

- Package (`src/evat`) scaffolded, importable, tested.
- Dataset ingestion: MVTec AD and YouTube-VOS adapters, manifest/validation
  layer.
- Video pipeline: frame sampling, temporal sequence construction, tensor
  loading with object-ID preservation.
- Segmentation: a from-scratch U-Net baseline, training/checkpointing/
  evaluation/visualization code — implemented and tested (CPU, synthetic
  fixtures), **not yet trained on real data**.
- Tracking: a baseline mask-IoU tracker (matching, NEW/ACTIVE/MISSED/
  TERMINATED lifecycle, identity metrics, visualization) — implemented
  and tested (CPU, synthetic fixtures), **not yet evaluated on real
  YouTube-VOS data**.
- Feature extraction: a handcrafted baseline encoder + a MobileNetV3-
  Small learned encoder (untrained locally, no weight download), crop/
  temporal-sequence/cache layers — implemented and tested (CPU, synthetic
  fixtures), **not yet run on real YouTube-VOS data**.
- Video Transformer: implemented from scratch (attention, positional
  encoding, encoder blocks, masked pooling, classification head) — no
  complete pretrained video Transformer architecture is used anywhere.
  Unit-tested on CPU (including a hand-computed attention correctness
  test and a tiny overfit test), **not yet trained on real data**.
- Baselines and task definition: object category classification task
  (genuine YouTube-VOS annotation, see `docs/task_definition.md`),
  leakage-safe video-level split, non-temporal MLP baseline, GRU temporal
  baseline, shared controlled train/eval loop across MLP/GRU/Transformer
  — implemented and tested (CPU, tiny fixture), **no comparison or
  ablation experiment has been run yet**.
- Anomaly detection (MVTec AD, separate from the video pipeline): per-
  category Mahalanobis-distance normality model over pretrained CNN
  features, image-level ROC-AUC/PR-AUC, coarse pixel-level anomaly maps
  — implemented and tested (CPU, tiny fixture, no pretrained-weight
  download), **no real MVTec experiment has been run yet**.
- Ruff (lint + format), mypy, pytest, pre-commit, and CI configured.
- Dependencies: numpy, pillow, torch, torchvision (CPU locally; GPU work
  happens in Colab), pyyaml. No datasets or trained model weights are
  committed.
