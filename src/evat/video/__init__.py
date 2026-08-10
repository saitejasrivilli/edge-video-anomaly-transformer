"""Dataset-agnostic temporal video pipeline.

- ``sampling``: choose which frame indices to use from a video
- ``sequence``: assemble a ``TemporalSequence`` from a video record + indices
- ``tensors``: load a ``TemporalSequence`` into image/annotation arrays

These modules operate on the generic ``VideoRecord``/``FrameRecord`` shape
produced by dataset adapters (e.g. ``evat.data.datasets.youtube_vos``) and
do not know about any single dataset's on-disk layout.
"""
