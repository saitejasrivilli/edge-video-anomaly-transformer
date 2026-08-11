# Phase 8 Anomaly Detection Task Definition

MVTec AD is used separately from the YouTube-VOS video pipeline. It is a
still-image dataset — this phase does not treat it as video, does not use
the temporal Transformer, tracker, or temporal-sequence machinery from
Phases 2/4/5/6/7 at all.

## Image-level anomaly detection

**Input:** one MVTec test image (any category).
**Output:** a scalar anomaly score (Mahalanobis distance from the
category's learned normal feature distribution). Higher = more
anomalous. Threshold-free evaluation (ROC-AUC/PR-AUC) is primary; an
operating threshold, if reported, is derived from training data only
(see "Evaluation" below).

## Pixel-level anomaly localization

**Input:** one MVTec test image.
**Output:** an anomaly map at the CNN backbone's downsampled feature-grid
resolution (coarse — NOT pixel-precise semantic segmentation), upsampled
via nearest-neighbor for visualization/pixel-metric comparison against
the ground-truth defect mask.

## Training data

Only `train/good/*.png` images for a given category — normal (defect-
free) images exclusively, per MVTec's own unsupervised-anomaly-detection
protocol (see `docs/datasets.md`). No anomalous image, train or test,
ever contributes to fitting the normal distribution.

## Test data

`test/good/*.png` (normal) and `test/<defect_type>/*.png` (anomalous)
for the same category. Ground-truth pixel masks
(`ground_truth/<defect_type>/*_mask.png`) exist only for anomalous test
images (MVTec provides no mask for normal images, since there is no
defect to localize).

## Labels

- **Image-level label:** derived from the Phase 1 `SampleRecord.label`
  field — `"good"` = normal (0), any other value = anomalous (1). This
  is MVTec's own directory-naming convention, not invented.
- **Pixel-level label:** the corresponding `ground_truth` mask, binarized
  (any nonzero pixel = defect).

## Dataset Source

MVTec AD, official source and verified CC BY-NC-SA 4.0 (non-commercial)
license documented in `docs/datasets.md`. Parsed by the unmodified Phase
1 adapter (`evat.data.datasets.mvtec`).

## Category strategy

**One normality model per category**, never a single global model. MVTec
categories (e.g. "bottle" vs. "screw") have entirely different normal
appearances; a global distribution would flag ordinary cross-category
variation as "anomalous," which is not a valid experiment. This matches
MVTec's own per-category train/test structure and the standard MVTec AD
evaluation protocol. See `evat.anomaly.model` and the category-isolation
test in `tests/unit/anomaly/test_model.py`.

## Evaluation protocol

- **Image-level:** ROC-AUC and PR-AUC over `(anomaly_score,
  is_anomalous)` pairs for a category's full test set (`good` +
  all defect types together, as MVTec's standard evaluation does).
  Threshold-free — no accuracy/F1 number is reported at an arbitrarily
  chosen threshold.
- **Pixel-level:** pixel-wise ROC-AUC between the upsampled anomaly map
  and the binarized ground-truth mask, computed only over anomalous test
  images (normal images have no ground-truth mask to compare against).
- **Threshold (if reported at all):** the given percentile (default
  95th) of the anomaly scores computed on the TRAINING normal images
  themselves — never derived from test labels (CLAUDE.md Phase 8 Section
  14).

## Leakage controls

- Normal distributions are fit exclusively from `train/good` images.
  Anomalous images (test-time, by construction) never appear in fitting.
- The threshold, when used, comes from training-set scores only.
- Category models are isolated (see Category strategy above) — a
  category's fitted mean/precision matrices are never combined with or
  substituted for another category's.

## Limitations

- Pixel-level localization is coarse (feature-grid resolution, one
  shared normal distribution across all spatial positions — not a
  per-location model like PaDiM). It demonstrates the localization
  *mechanism*, not competitive pixel-precision.
- The Mahalanobis-distance-over-pretrained-features approach is a strong,
  simple, well-understood baseline — not a claim of state-of-the-art
  MVTec performance.
- MobileNetV3-Small was chosen for Colab/CPU feasibility (consistent
  with Phase 5), not because it is the optimal backbone for anomaly
  detection; a deeper/wider backbone might do better at the cost of
  compute.
- Regularized covariance (shrinkage) is a documented compromise for
  small per-category training sets and moderate feature dimensionality,
  not a tuned, cross-validated choice.
