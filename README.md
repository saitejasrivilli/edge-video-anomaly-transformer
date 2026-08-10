# Edge Video Anomaly Transformer (evat)

## Status: Phase 5 — Visual Feature Extraction (code complete, experiment not yet run)

This project is under active, phase-by-phase development. Repository
foundation, dataset ingestion (MVTec AD, YouTube-VOS), the temporal video
pipeline, a segmentation baseline, a baseline mask-IoU object tracker, and
a visual feature extraction layer (handcrafted baseline + MobileNetV3-
Small learned encoder, crop/temporal-sequence/cache logic, all
unit-tested on CPU with synthetic fixtures) are implemented. **No
segmentation training, no tracking evaluation, and no feature-extraction
experiment have been executed yet** — see `docs/experiments.md`. **The
Video Transformer is not implemented yet**, nor is anomaly detection.

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
  fixtures), **not yet run on real YouTube-VOS data**. The Video
  Transformer that will consume these sequences is not implemented yet.
- Ruff (lint + format), mypy, pytest, pre-commit, and CI configured.
- Dependencies: numpy, pillow, torch, torchvision (CPU locally; GPU work
  happens in Colab), pyyaml. No datasets or trained model weights are
  committed.
