# Architecture

## Status

This document describes the **planned** system architecture. Nothing beyond
Phase 0 (repository and engineering foundation) has been implemented.

## Planned pipeline

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
Video Transformer (implemented from scratch)
  ↓
Anomaly Detection
  ↓
Inference / Benchmarking
  ↓
Optimized Runtime
```

## Component status

| Component | Status |
|---|---|
| Repository/package foundation | Implemented (Phase 0) |
| Dataset ingestion and validation | Not started (Phase 1) |
| Video preprocessing | Not started (Phase 2) |
| Segmentation | Not started (Phase 3) |
| Object tracking | Not started (Phase 4) |
| Visual feature extraction | Not started (Phase 5) |
| Video Transformer (from scratch) | Not started (Phase 6) |
| Baselines / ablations | Not started (Phase 7) |
| Anomaly detection | Not started (Phase 8) |
| Inference optimization | Not started (Phase 9) |
| End-to-end pipeline | Not started (Phase 10) |
| Production hardening | Not started (Phase 11) |
| CI/CD and quality gates | Partial — lightweight CI configured in Phase 0, expanded in Phase 12 |
| Final benchmarking and documentation | Not started (Phase 13) |

## Compute environment split

Heavy computation (training, large-scale inference, GPU benchmarks) runs in
Google Colab. The repository stays lightweight and locally runnable for
tests, linting, and type checking. See `CLAUDE.md` Section 7 for the full
policy.
