"""Object tracking: schemas, matching, tracker, and evaluation.

Ground truth (YouTube-VOS object IDs) and tracker predictions (predicted
track IDs) are kept in explicitly separate types throughout this package —
see ``schemas.ObjectCandidate`` (anonymous, tracker input) vs.
``ground_truth.GroundTruthInstance`` (labeled, evaluation-only input). The
tracker never receives ground-truth identity as part of its input.
"""
