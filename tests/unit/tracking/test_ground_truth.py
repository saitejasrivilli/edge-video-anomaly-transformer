import numpy as np

from evat.tracking.ground_truth import extract_ground_truth_instances, strip_identity
from evat.tracking.schemas import ObjectCandidate


def test_extract_ground_truth_instances_splits_by_object_id():
    id_mask = np.array([[0, 1, 1], [2, 2, 0], [0, 0, 0]], dtype=np.uint8)

    instances = extract_ground_truth_instances(id_mask, frame_id="0")

    assert {i.gt_object_id for i in instances} == {"1", "2"}
    obj1 = next(i for i in instances if i.gt_object_id == "1")
    assert obj1.mask.tolist() == [[0, 1, 1], [0, 0, 0], [0, 0, 0]]


def test_extract_ground_truth_instances_ignores_background():
    id_mask = np.zeros((3, 3), dtype=np.uint8)

    instances = extract_ground_truth_instances(id_mask, frame_id="0")

    assert instances == []


def test_strip_identity_drops_gt_id_but_keeps_mask_and_bbox():
    id_mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    instances = extract_ground_truth_instances(id_mask, frame_id="0")

    candidates = strip_identity(instances)

    assert all(isinstance(c, ObjectCandidate) for c in candidates)
    assert not hasattr(candidates[0], "gt_object_id")
    assert candidates[0].mask.tolist() == instances[0].mask.tolist()
