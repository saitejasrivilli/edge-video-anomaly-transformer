# Experiments

Every experiment recorded here must trace to a real run and a file under
`results/<component>/<experiment_name>/`. No number in this document may be
estimated, guessed, or "roughly" stated — see CLAUDE.md Section 1 and
Section 31.

## Segmentation baseline (Phase 3)

**Status: training not yet executed. No performance metrics are claimed.**

### Planned experiment strategy

1. **Smoke experiment** — a handful of YouTube-VOS training videos, 1
   epoch, verifies data loading, mask alignment, forward/backward pass,
   checkpointing, and evaluation run end-to-end without error. No
   performance conclusions are drawn from this run.
2. **Baseline experiment** — a Colab-appropriate subset of the YouTube-VOS
   training split, evaluated against the official validation split,
   producing the first real IoU/Dice/precision/recall numbers.
3. Only after the baseline is verified stable would a larger-scale
   experiment be considered (out of scope for Phase 3).

### How results will be recorded

`evat.evaluation.experiment.save_experiment_record` writes, per experiment:

```
results/segmentation/<experiment_name>/
    config.yaml     # exact SegmentationTrainingConfig used
    metrics.json     # metrics + git commit + dataset version + hardware + runtime
    summary.md       # human-readable summary of the above
```

`notebooks/03_segmentation_training.ipynb` is the only place these
experiments run (Google Colab, GPU). Local execution never trains this
model on real data — local tests use tiny synthetic/fixture data only
(`tests/unit/training/`, `tests/unit/models/`, `tests/unit/evaluation/`).

### Results

| Experiment | Dataset subset | GPU | Epochs | IoU | Dice | Precision | Recall | Runtime |
|---|---|---|---|---|---|---|---|---|
| _none yet_ | — | — | — | — | — | — | — | — |

This table will only be filled in with numbers copied directly from a
`results/segmentation/<experiment_name>/metrics.json` produced by an
actual Colab run.

## Object tracking baseline (Phase 4)

**Status: tracking evaluation not yet executed. No tracking performance
metrics are claimed.**

### Planned experiment strategy

1. **Smoke experiment** — a couple of YouTube-VOS training videos, verify
   frames/masks/object IDs load, the tracker produces tracks, metrics
   calculate, and visualization renders — no performance conclusions.
2. **Baseline evaluation** — the official YouTube-VOS validation split
   (never used for tuning), reporting coverage / id_consistency /
   identity_switches / track_fragmentation as actually measured, plus
   basic runtime/FPS (not claimed as "real-time" unless methodology
   supports it).

### How results will be recorded

`notebooks/04_tracking_evaluation.ipynb` prints metrics and runtime
directly from `evat.tracking.metrics.evaluate_tracking`; those printed
values — not estimates — are what would be copied into the table below
and into a `results/tracking/<experiment_name>/` record, following the
same `config.yaml`/`metrics.json`/`summary.md` convention as segmentation.

### Results

| Experiment | Split | Videos | Frames | Coverage | ID consistency | Identity switches | Fragmentation | Runtime | FPS |
|---|---|---|---|---|---|---|---|---|---|
| _none yet_ | — | — | — | — | — | — | — | — | — |

## Visual feature extraction (Phase 5)

**Status: feature extraction experiment not yet executed. No performance
or feature-quality metrics are claimed.**

### Planned experiment strategy

1. **Smoke experiment** — a couple of YouTube-VOS training videos: verify
   the MobileNetV3-Small encoder loads (with real pretrained weights, in
   Colab only), objects crop correctly, features are produced, temporal
   sequences build correctly, and the feature cache round-trips — no
   performance conclusions.
2. **Subset run** — a Colab-appropriate subset, measuring real extraction
   throughput (features/sec) and recording backbone, pretrained/frozen
   status, input size, feature dimension, frame/track/sequence counts,
   GPU, and runtime — all as actually printed, never estimated.
3. **Baseline vs. learned comparison** — optional; if performed, must use
   an actual measurement (e.g. nearest-neighbor consistency of same-track
   vs. different-track feature vectors) documented alongside the numbers.
   If not performed, this document states that explicitly rather than
   implying a comparison happened.

### How results will be recorded

`notebooks/05_feature_extraction.ipynb` prints shapes/counts/throughput
directly; those values — not estimates — would be copied into the table
below and into a `results/features/<experiment_name>/` record, following
the same `config.yaml`/`metrics.json`/`summary.md` convention used for
segmentation and tracking.

### Baseline vs. learned comparison

Not performed. No claim is made that either representation is superior.

### Results

| Experiment | Backbone | Pretrained | Frozen | Input size | Feature dim | Videos | Frames | Tracks | Sequences | GPU | Runtime | Throughput |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| _none yet_ | — | — | — | — | — | — | — | — | — | — | — | — |
