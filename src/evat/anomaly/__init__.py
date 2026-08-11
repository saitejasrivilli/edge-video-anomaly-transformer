"""Industrial anomaly detection on MVTec AD — separate from the YouTube-VOS
video pipeline. MVTec AD is a still-image dataset; this package does not
treat it as video, does not use the temporal Transformer, and does not
reuse the tracking/temporal-sequence machinery.

Pipeline: pretrained CNN feature extractor (Phase 5, unmodified) ->
normal-only training feature distribution (per-category Gaussian) ->
Mahalanobis distance -> image-level anomaly score / pixel-level anomaly
map -> ROC-AUC / PR-AUC evaluation.
"""
