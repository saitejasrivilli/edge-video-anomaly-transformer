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
