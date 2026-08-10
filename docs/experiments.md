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

## Video Transformer (Phase 6)

**Status: training not yet executed. No model performance metrics are
claimed.**

### Planned experiment strategy

1. **Smoke run** — synthetic tensors only: verify the model builds, moves
   to GPU, forward/backward pass work, and a checkpoint saves/loads — no
   performance conclusions, no real data required.
2. **Small training run** — on real Phase 5 feature sequences, once a
   downstream label source exists (this phase implements the model and
   the interface, not a labeled task — see limitations below). Would
   record: GPU, batch size, sequence length, feature dimension, `d_model`,
   layers, heads, learning rate, epochs, runtime, and validation metrics,
   all as actually measured.
3. Ablations (sequence length, layers, heads, temporal baseline vs.
   Transformer) are explicitly deferred to Phase 7.

### How results will be recorded

`notebooks/06_video_transformer.ipynb` prints configuration, parameter
count, and (once a real run is possible) loss/metrics/runtime directly;
those values would be copied into the table below and into a
`results/transformer/<experiment_name>/` record, following the same
`config.yaml`/`metrics.json`/`summary.md` convention as prior phases.

### Results

| Experiment | d_model | Layers | Heads | Seq length | Params | GPU | Batch size | Epochs | Runtime | Metric |
|---|---|---|---|---|---|---|---|---|---|---|
| _none yet_ | — | — | — | — | — | — | — | — | — | — |

## Baselines and ablations (Phase 7)

**Status: comparison and ablation experiments not yet executed. No
model performance metrics are claimed.**

Task: see `docs/task_definition.md` (object category classification from
YouTube-VOS's official per-object category annotation). "NO VALID
DOWNSTREAM TASK ESTABLISHED" does NOT apply — a genuine, dataset-provided
label was identified and a defensible task was defined.

### Planned experiment strategy

1. **Main comparison** — `TemporalMeanPoolBaseline` vs. `TemporalGRUBaseline`
   vs. `VideoTransformer`, identical split/features/sequence length/batch
   size/optimizer/epochs (`configs/phase7.yaml`), evaluated with macro F1
   (primary) and accuracy.
2. **Sequence-length ablation** — `configs/phase7.yaml`'s
   `ablations.sequence_length` list, tested only after inspecting how
   many usable frames real sequences actually have.
3. **Positional-encoding ablation** — Transformer with vs. without
   sinusoidal positional encoding, same data/training.
4. **Masking ablation** — correct validity masking vs. forcing the mask
   to all-valid, to check whether padded positions contaminate the
   pooled representation. Only meaningful if built sequences actually
   contain padding.
5. **Model-size ablation** — small vs. medium Transformer configuration,
   same training budget, if compute permits.

### Seed strategy

Single fixed seed (`configs/phase7.yaml: seed: 42`) planned for the
initial run, given free-Colab compute constraints. **This is a
single-run limitation** — no statistical significance (mean/std across
seeds) can be claimed from one seed. If compute permits, the main
comparison will be repeated across a small number of seeds and mean ±
std reported; until then, all numbers here would represent one run only.

### How results will be recorded

`notebooks/07_baselines_and_ablations.ipynb` calls
`evat.experiments.record.save_experiment_result` for every model and
ablation, writing `results/phase7/<experiment_name>/{config.json,
metrics.json,summary.md}`. Numbers below would be copied directly from
those files, never hand-typed.

### Main comparison results

| Model | Macro F1 | Accuracy | Macro Precision | Macro Recall | Params | Runtime |
|---|---|---|---|---|---|---|
| _none yet_ | — | — | — | — | — | — |

### Ablation results

| Ablation | Variant | Macro F1 | Accuracy | Runtime |
|---|---|---|---|---|
| _none yet_ | — | — | — | — |

### Analysis

Not yet possible — no experiments have been run. Once run, this section
must explicitly answer (measured result vs. interpretation vs.
hypothesis, kept distinct): does temporal modeling help; does the
Transformer beat the non-temporal baseline; does it beat the GRU; does
positional encoding matter; does sequence length matter; does masking
matter; what is the compute/performance tradeoff. Negative results
(GRU or MLP winning) are valid and will be reported as such.
