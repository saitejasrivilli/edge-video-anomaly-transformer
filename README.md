# Edge Video Anomaly Transformer (evat)

## Status: Phase 0 — Repository and Engineering Foundation

This project is under active, phase-by-phase development. No segmentation,
tracking, Transformer, or anomaly-detection functionality exists yet — only
the repository/package foundation described below.

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

No dataset has been downloaded or integrated at this stage. Dataset
selection and license verification happen in Phase 1. See
`docs/datasets.md`.

## Current project status

- Package (`src/evat`) scaffolded, importable, tested.
- Ruff (lint + format), mypy, pytest, pre-commit, and CI configured.
- No ML dependencies, no datasets, no trained models.
