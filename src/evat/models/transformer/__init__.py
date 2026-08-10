"""Video Transformer, implemented from scratch with PyTorch primitives.

No complete pretrained Video Transformer (TimeSformer, VideoMAE, Video
Swin, etc.) is used anywhere in this package. Positional encoding, Q/K/V
projections, scaled dot-product attention, multi-head attention,
residual+normalization, the feed-forward network, and the encoder block
are all implemented directly here — only ``torch.nn.Linear``,
``LayerNorm``, ``Dropout``, and tensor ops are reused from PyTorch. See
docs/architecture.md ("Video Transformer (Phase 6)") for the full
rationale and tensor-shape reference.
"""
