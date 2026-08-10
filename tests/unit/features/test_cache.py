import numpy as np

from evat.features.cache import FeatureCache, compute_cache_key, hash_config


def test_hash_config_is_deterministic_for_same_config():
    config = {"backbone": "mobilenet_v3_small", "pretrained": False}

    assert hash_config(config) == hash_config(dict(config))


def test_hash_config_differs_for_different_config():
    a = hash_config({"pretrained": True})
    b = hash_config({"pretrained": False})

    assert a != b


def test_compute_cache_key_is_deterministic():
    key_a = compute_cache_key("youtube_vos", "video1", "0", 1, "baseline_stats_v1", "abc123")
    key_b = compute_cache_key("youtube_vos", "video1", "0", 1, "baseline_stats_v1", "abc123")

    assert key_a == key_b


def test_compute_cache_key_distinguishes_global_from_track():
    global_key = compute_cache_key("youtube_vos", "video1", "0", None, "baseline_stats_v1", "abc")
    track_key = compute_cache_key("youtube_vos", "video1", "0", 1, "baseline_stats_v1", "abc")

    assert global_key != track_key


def test_cache_put_then_get_round_trips(tmp_path):
    cache = FeatureCache(tmp_path)
    feature = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    config_hash = hash_config({"backbone": "x"})

    cache.put("key1", feature, config_hash=config_hash, git_commit="abc123")
    loaded = cache.get("key1", expected_config_hash=config_hash)

    assert np.array_equal(loaded, feature)


def test_cache_get_returns_none_for_missing_key(tmp_path):
    cache = FeatureCache(tmp_path)

    assert cache.get("missing") is None


def test_cache_detects_stale_entry_on_config_change(tmp_path):
    cache = FeatureCache(tmp_path)
    feature = np.zeros(3, dtype=np.float32)
    old_hash = hash_config({"v": 1})
    new_hash = hash_config({"v": 2})

    cache.put("key1", feature, config_hash=old_hash)

    assert cache.is_stale("key1", expected_config_hash=new_hash) is True
    assert cache.is_stale("key1", expected_config_hash=old_hash) is False
    # get() with the new (mismatched) hash refuses to silently return stale data.
    assert cache.get("key1", expected_config_hash=new_hash) is None
