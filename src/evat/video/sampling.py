"""Frame sampling strategies.

Pure index arithmetic — no file I/O, no image decoding. Given a total frame
count, return which frame indices to include in a temporal sequence.
"""

from __future__ import annotations


def uniform_frame_indices(num_frames_total: int, num_samples: int) -> list[int]:
    """Pick ``num_samples`` indices spread evenly across ``[0, num_frames_total)``.

    Always includes index 0. If ``num_samples >= num_frames_total``, returns
    every available index (no upsampling/duplication).

    Raises:
        ValueError: if ``num_frames_total`` or ``num_samples`` is not positive.
    """
    if num_frames_total <= 0:
        raise ValueError("num_frames_total must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if num_samples >= num_frames_total:
        return list(range(num_frames_total))

    step = num_frames_total / num_samples
    return [int(i * step) for i in range(num_samples)]


def strided_frame_indices(num_frames_total: int, stride: int) -> list[int]:
    """Pick every ``stride``-th index starting at 0.

    Raises:
        ValueError: if ``num_frames_total`` is not positive or ``stride`` < 1.
    """
    if num_frames_total <= 0:
        raise ValueError("num_frames_total must be positive")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    return list(range(0, num_frames_total, stride))
