from evat.experiments.config import Phase7ExperimentConfig


def test_config_roundtrips_through_yaml(tmp_path):
    config = Phase7ExperimentConfig(seed=1, batch_size=16, epochs=5)
    path = tmp_path / "phase7.yaml"

    config.to_yaml(path)
    loaded = Phase7ExperimentConfig.from_yaml(path)

    assert loaded == config


def test_config_from_yaml_ignores_ablations_section(tmp_path):
    path = tmp_path / "phase7.yaml"
    path.write_text(
        "seed: 1\nval_fraction: 0.2\nnum_frames_per_video: 16\nsequence_length: 16\n"
        "stride: 1\nbatch_size: 8\nlearning_rate: 0.001\nepochs: 20\noptimizer: adam\n"
        "ablations:\n  sequence_length: [4, 16]\n"
    )

    config = Phase7ExperimentConfig.from_yaml(path)

    assert config.seed == 1
