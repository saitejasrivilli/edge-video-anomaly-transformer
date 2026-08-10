# Phase 7 Downstream Task Definition

## Task

**Object category classification from a temporal visual feature
sequence.** Given the sequence of visual features extracted for one
tracked object across a video (Phase 5's `TemporalFeatureSequence`),
predict which YouTube-VOS object category that track belongs to (e.g.
"dog", "bear", "airplane", ...).

This is downstream-task direction **A** from the phase brief: object/
category classification using object-associated sequences, using a label
the dataset genuinely provides — not an invented or self-fulfilling
target.

## Input

For one tracked object: `features [T, D]` + `validity [T]`, exactly the
`TemporalFeatureSequence` produced by the Phase 5 pipeline (Phase 4
tracker output -> Phase 5 crop + encoder -> Phase 5 temporal builder).
No information beyond the object's own visual appearance across frames is
given to the model.

## Target

`gt_object_id`'s associated **category string**, taken directly from
YouTube-VOS's official `meta.json`:

```
{"videos": {"<video_id>": {"objects": {"<object_id>": {"category": "<name>", "frames": [...]}}}}}
```

Parsed today by `evat.data.datasets.youtube_vos.ObjectMeta.category`
(Phase 2 code, unmodified). This is an official annotation field, not
derived, inferred, or synthesized — see `docs/datasets.md` for
YouTube-VOS's verified license/source.

Category strings are mapped to integer class indices by
`evat.experiments.task_category.build_category_label_map`, built by
sorting the **observed** category names for determinism (not a
hard-coded list, since the exact category set an experiment sees depends
on which videos are used).

## Dataset Source

YouTube-VOS `train` split's `meta.json`, official per-object category
annotations. See `docs/datasets.md` ("Selected Video Dataset") for the
verified license (non-commercial research use) and source.

## Split

**Split unit: video**, not object or frame. All objects/frames belonging
to a given `video_id` go entirely into one split — never divided across
train/validation. This matters because a video's frames are highly
correlated (same lighting, same object instance, near-duplicate crops
frame-to-frame); splitting below video level would leak near-identical
visual evidence between train and validation.

`evat.experiments.task_category.split_videos_by_video_id(videos,
val_fraction, seed)` performs this split deterministically from a fixed
seed, shuffling the list of `video_id`s (not the objects/frames within
them) before partitioning.

**Deviation from YouTube-VOS's official train/val split — documented,
not accidental:** the official YouTube-VOS `valid` split is designed for
the *segmentation generalization* benchmark, and deliberately contains
categories NOT present in `train` (to test whether a segmentation model
generalizes to unseen object classes). That design is incompatible with
closed-set category **classification** — a classifier cannot correctly
predict a category label it never saw during training. Using the
official `valid` split here would produce a metric that measures nothing
meaningful about the model, only about the label-set mismatch. Phase 7
therefore constructs its own held-out validation split **from the
`train` split only**, via the video-level split above, so every category
appearing in validation also appears in training.

## Leakage Analysis

- **Video-level split** (above) prevents near-duplicate-frame leakage
  between train and validation.
- **Category label is never part of the model's input.** The model only
  ever receives visual features (`[T, D]`) and the validity mask — the
  category string is used exclusively to build the integer target and
  is discarded before constructing model input tensors.
- **No temporal future leakage:** the target is a single per-object
  label that is constant across the entire object's lifetime in the
  video (an object's category does not change frame to frame), so there
  is no "predicting the future from the past" concern the way there
  would be for a next-frame-prediction task. The full feature sequence
  is legitimately available at "inference time" for this task, since the
  task is defined as "classify this observed track," not "predict what
  happens next."
- **Feature-cache leakage:** Phase 5's `FeatureCache` keys on
  `(dataset, video_id, frame_id, track_id, extractor_name, config_hash)`
  — a cached feature is never shared across videos, so no cross-split
  contamination can occur through the cache.
- **Object-candidate leakage (inherited, documented limitation):** per
  Phase 4/5, object candidates for this pipeline are currently derived
  from YouTube-VOS's own ground-truth object-ID masks (via
  `evat.tracking.ground_truth.strip_identity`), because Phase 3's
  segmentation baseline is not yet instance-aware. This means the
  *location/extent* of each object crop is ground-truth-derived, which
  is a known limitation of this pipeline stage (documented since Phase
  4), not something newly introduced by this task. It does not leak the
  **category label** itself into the input — only the mask, the same
  mask a real instance segmenter would also need to produce.

## Evaluation

Categories are naturally imbalanced (some objects like "person" or
"dog" appear far more often in YouTube-VOS than rare categories), so
**accuracy alone would be misleading** — a model that always predicts
the majority category could score deceptively well. Metrics used:

- **macro F1** (primary) — averages F1 per class unweighted, so rare
  categories count as much as common ones.
- **accuracy** (secondary, for interpretability).
- **macro precision / macro recall** (diagnostic, to see whether errors
  skew toward false positives or false negatives for a given class).

Implemented in `evat.experiments.metrics` without adding a new
dependency (no scikit-learn) — computed directly from per-class
confusion counts.

## Limitations

- This task demonstrates **temporal feature aggregation for a
  whole-object property**, not per-frame or next-frame temporal
  reasoning — it does not test the Transformer's ability to model
  *changing* state over time, only its ability to aggregate a sequence
  into a stable representation.
- The category set and its distribution are whatever is present in the
  videos actually used; with only a small Colab-feasible subset, some
  categories may have very few examples, limiting statistical confidence
  (see `docs/experiments.md` for the seed/limitation discussion once an
  experiment is run).
- Object crops depend on ground-truth masks (see Leakage Analysis above)
  rather than a trained instance segmenter, so this task's results say
  nothing about performance in a fully automated pipeline yet.
- A single category per object is a coarse signal; it does not evaluate
  localization, tracking, or segmentation quality — only whether the
  aggregated visual-temporal representation carries enough information
  to name the object.
