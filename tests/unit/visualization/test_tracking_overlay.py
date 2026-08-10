import numpy as np
from PIL import Image

from evat.tracking.schemas import TrackedInstance, TrackState
from evat.visualization.tracking_overlay import draw_tracks


def test_draw_tracks_returns_rgb_image_same_size():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    instance = TrackedInstance(
        frame_id="0",
        track_id=1,
        state=TrackState.ACTIVE,
        mask=np.zeros((10, 10)),
        bbox=(2, 2, 5, 5),
    )

    image = draw_tracks(frame, [instance])

    assert isinstance(image, Image.Image)
    assert image.size == (10, 10)
    assert image.mode == "RGB"


def test_draw_tracks_handles_no_instances():
    frame = np.zeros((5, 5, 3), dtype=np.uint8)

    image = draw_tracks(frame, [])

    assert image.size == (5, 5)
