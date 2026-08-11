import numpy as np

from evat.anomaly.localization import upsample_anomaly_map


def test_upsample_preserves_block_structure():
    small_map = np.array([[0.0, 1.0], [1.0, 0.0]])

    upsampled = upsample_anomaly_map(small_map, size=(4, 4))

    assert upsampled.shape == (4, 4)
    # Nearest-neighbor: top-left 2x2 block should all equal the original [0,0] value.
    assert (upsampled[:2, :2] == 0.0).all()
    assert (upsampled[:2, 2:] == 1.0).all()
