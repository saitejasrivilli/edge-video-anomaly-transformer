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
| Object tracking | Code complete (Phase 4) — baseline mask-IoU greedy tracker, lifecycle (NEW/ACTIVE/MISSED/TERMINATED), ground-truth/prediction separation, identity metrics, visualization implemented and unit-tested (CPU, synthetic fixtures). **Colab evaluation not yet executed; no tracking performance metrics exist yet.** |
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

## Object tracking baseline (Phase 4)

**This is a baseline tracker, not production-grade tracking.** It exists
to demonstrate understanding of the tracking problem — identity
maintenance across frames — not to compete with established trackers.

**Ground truth vs. prediction:** kept as two structurally distinct types
throughout `evat.tracking`. `evat.tracking.ground_truth.GroundTruthInstance`
carries YouTube-VOS's real object ID (`gt_object_id`); `evat.tracking.
schemas.ObjectCandidate` (tracker input) and `TrackedInstance` (tracker
output) carry no ground-truth identity at all — only a mask, bbox, and a
tracker-assigned `track_id`. `ground_truth.strip_identity()` is the only
path from one to the other, and it only removes identity, never adds it —
so ground truth cannot silently leak into what the tracker sees.

**Candidate source:** Phase 3's segmentation baseline is binary
foreground/background only (not instance-aware), so it cannot yet produce
per-object candidate masks. Until an instance-aware segmentation stage
exists, this baseline obtains per-frame object candidates by extracting
each ground-truth object ID's binary mask from YouTube-VOS annotations
(`ground_truth.extract_ground_truth_instances`) and stripping the ID
before handing candidates to the tracker. This is documented, not hidden:
evaluation never uses the stripped ID back — it independently re-matches
predicted tracks to ground truth by mask overlap.

**Matching:** `evat.tracking.matching` implements mask IoU, bbox IoU, and
centroid-distance similarity directly (no external tracking library).
`greedy_match` assigns highest-scoring track/candidate pairs first,
above a configurable threshold, with deterministic tie-breaking by index
— not globally optimal (unlike Hungarian assignment), but simple,
inspectable, and adequate for a first baseline.

**Track lifecycle** (`evat.tracking.schemas.TrackState`):

```
unmatched candidate                  --> NEW
NEW / ACTIVE / MISSED, matched       --> ACTIVE
ACTIVE / MISSED, unmatched,
  missed_frames <= max_missed_frames --> MISSED
MISSED, missed_frames >
  max_missed_frames                  --> TERMINATED (dropped)
```

Configurable via `evat.tracking.tracker.TrackerConfig` /
`configs/tracking.yaml`: `matching_method`, `iou_threshold`,
`max_missed_frames`, `min_track_length` — nothing hard-coded.

**Metrics** (`evat.tracking.metrics`) — deliberately not MOTA/HOTA/IDF1:

- **coverage**: fraction of ground-truth object-frames matched (IoU
  above threshold) by any predicted track.
- **id_consistency**: per ground-truth object, the fraction of its
  matched frames assigned to that object's single most common predicted
  track ID, averaged across objects.
- **identity_switches**: total frame-to-frame changes in the matched
  predicted track ID, summed across ground-truth objects.
- **track_fragmentation**: average number of distinct predicted track
  IDs matched to each ground-truth object.

**Visualization:** `evat.visualization.tracking_overlay.draw_tracks`
draws each tracked instance's bounding box and predicted `track_id` over
the frame, so identity switches are visible directly, not just inferred
from metrics.

**Temporal integration:** reuses the Phase 2 `evat.video.sequence.
TemporalSequence` / `evat.video.tensors.load_temporal_sequence` reader —
no separate video reader was built for tracking.

## Compute environment split

Heavy computation (training, large-scale inference, GPU benchmarks) runs in
Google Colab. The repository stays lightweight and locally runnable for
tests, linting, and type checking. See `CLAUDE.md` Section 7 for the full
policy.
