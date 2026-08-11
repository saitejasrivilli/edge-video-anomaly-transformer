"""End-to-end Phase 8 pipeline test against a tiny local MVTec-shaped fixture.

MVTec manifest (Phase 1, unmodified) -> normal-only training images ->
CNN features (Phase 5, unmodified) -> fitted normality model -> image-
level anomaly scores + ROC-AUC -> pixel-level anomaly map.

No real MVTec data or pretrained weights are downloaded/required.
"""

from pathlib import Path

import torch

from evat.anomaly.dataset import filter_records, load_mvtec_image, load_mvtec_mask
from evat.anomaly.localization import upsample_anomaly_map
from evat.anomaly.metrics import roc_auc_score
from evat.anomaly.model import anomaly_map, fit_category_anomaly_model, image_anomaly_score
from evat.data.datasets.mvtec import build_manifest
from evat.features.encoders import CNNEncoderConfig, CNNFeatureEncoder

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "mvtec_anomaly_mini"


def test_full_anomaly_pipeline_on_bottle_category():
    records = build_manifest(FIXTURE_ROOT, dataset_version="fixture-v0")
    train_normal = filter_records(records, category="bottle", split="train", label="good")
    test_records = filter_records(records, category="bottle", split="test")

    size = (16, 16)
    encoder = CNNFeatureEncoder(CNNEncoderConfig(pretrained=False))

    train_images = torch.stack(
        [load_mvtec_image(FIXTURE_ROOT, r.image_path, size) for r in train_normal]
    )
    model = fit_category_anomaly_model("bottle", encoder, train_images)

    test_images = torch.stack(
        [load_mvtec_image(FIXTURE_ROOT, r.image_path, size) for r in test_records]
    )
    scores = image_anomaly_score(model, encoder, test_images)
    labels = [0 if r.label == "good" else 1 for r in test_records]

    assert scores.shape == (len(test_records),)
    # With only 2 classes present in this tiny fixture, ROC-AUC is well-defined.
    if 0 in labels and 1 in labels:
        auc = roc_auc_score(scores, __import__("numpy").array(labels))
        assert 0.0 <= auc <= 1.0

    defect_record = next(r for r in test_records if r.label != "good")
    defect_image = load_mvtec_image(FIXTURE_ROOT, defect_record.image_path, size).unsqueeze(0)
    amap = anomaly_map(model, encoder, defect_image)
    upsampled = upsample_anomaly_map(amap, size=size)

    assert upsampled.shape == size

    gt_mask = load_mvtec_mask(FIXTURE_ROOT, defect_record.annotation_path, size=size)
    assert gt_mask.shape == size
    assert gt_mask.sum() > 0
