# Architecture

## Status

This document describes the **planned** system architecture and tracks
actual implementation status per component below. As of Phase 3, the
segmentation baseline's code, tests, and Colab training notebook exist,
but no training run has been executed yet — see `docs/experiments.md` for
the current (empty) results record.

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
| Dataset ingestion and validation | Implemented (Phase 1) — MVTec AD adapter; YouTube-VOS adapter added in Phase 2; DAVIS blocked |
| Video preprocessing | Implemented (Phase 2) — YouTube-VOS video index, frame sampling, temporal sequence construction, [T, C, H, W] tensor + mask loading. No resizing/normalization/augmentation yet. |
| Segmentation | Code complete (Phase 3) — U-Net baseline, dataset/transforms/loss/metrics/trainer/checkpointing/visualization implemented and unit-tested (CPU, synthetic fixtures). **Training not yet executed in Colab; no performance metrics exist yet.** |
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

## Segmentation baseline (Phase 3)

**Model:** `evat.models.unet.UNet` — a small, from-scratch U-Net (encoder/
decoder with skip connections via concatenation, `GroupNorm` + ReLU,
`MaxPool2d` downsampling, `ConvTranspose2d` upsampling). Configurable
`in_channels`, `out_channels`, `base_channels`, `depth`. Not a pretrained
or foundation model (no SAM/SAM2) — the goal is an understandable
project-owned baseline.

**Data representation and object-ID handling:** YouTube-VOS annotation
PNGs are palette-indexed (pixel value == object ID). The Phase 3 baseline
target is **binary foreground/background** — all nonzero object IDs are
collapsed into a single foreground class — because Phase 3's scope is a
single segmentation baseline, not per-instance classification. This is a
deliberate simplification, not a data-loss accident: `SegmentationDataset`
(`evat.training.dataset`) returns both the binary `mask` used for training
and the original, un-collapsed `object_id_mask` + `object_ids` tuple
unchanged, so Phase 4 (tracking) has the real per-object identities to
work with rather than reconstructing them from a binary mask.

**Preprocessing:** `evat.training.transforms` — bilinear resize for
images, **nearest-neighbor** resize for masks (bilinear would blend
label values at object boundaries and invent fractional classes that
never existed in the annotation), a simple `(x/255 - 0.5)/0.5` per-channel
normalization (not ImageNet stats — no ImageNet-pretrained weights are
used), and a horizontal flip applied identically to image and mask when
augmentation is enabled.

**Loss:** `evat.training.losses.BCEDiceLoss` — `BCEWithLogitsLoss` (stable,
well-understood) + soft Dice loss (targets mask overlap directly,
compensating for foreground/background class imbalance). A standard,
explainable combination for a first baseline, not a tuned/exotic loss.

**Metrics:** `evat.evaluation.metrics` — pixel-level IoU, Dice, precision,
recall over binarized (threshold 0.5) predictions vs. binary ground truth.
These are explicitly **pixel-level** metrics, not the object-level
metrics (e.g. J&F) used by the official DAVIS/YouTube-VOS benchmarks —
this project does not claim benchmark-comparable performance.

**Training/checkpointing:** `evat.training.trainer.Trainer` runs the
epoch loop (dataset/model/loss/optimizer are injected, not constructed
internally), evaluates on a schedule (`eval_every`), and checkpoints
model + optimizer + epoch + config + metrics via
`evat.training.checkpoint` after each evaluated epoch (Colab sessions can
terminate, so checkpointing is not optional).

**Qualitative evaluation:** `evat.visualization.overlay` renders a
frame / ground-truth / prediction / colored-overlay panel per sample, for
inspecting failure modes (missed objects, fragmented masks, boundary
errors) beyond aggregate metrics.

## Compute environment split

Heavy computation (training, large-scale inference, GPU benchmarks) runs in
Google Colab. The repository stays lightweight and locally runnable for
tests, linting, and type checking. See `CLAUDE.md` Section 7 for the full
policy.
