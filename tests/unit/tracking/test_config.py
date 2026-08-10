from evat.tracking.tracker import TrackerConfig


def test_tracker_config_roundtrips_through_yaml(tmp_path):
    config = TrackerConfig(iou_threshold=0.4, max_missed_frames=3)
    path = tmp_path / "tracking.yaml"

    config.to_yaml(path)
    loaded = TrackerConfig.from_yaml(path)

    assert loaded == config


def test_tracker_config_from_yaml_ignores_extra_sections(tmp_path):
    path = tmp_path / "tracking.yaml"
    path.write_text(
        "matching_method: bbox_iou\n"
        "iou_threshold: 0.5\n"
        "max_missed_frames: 2\n"
        "min_track_length: 1\n"
        "evaluation:\n"
        "  iou_threshold: 0.6\n"
    )

    config = TrackerConfig.from_yaml(path)

    assert config.matching_method == "bbox_iou"
    assert config.iou_threshold == 0.5
