"""A small, configurable U-Net for the Phase 3 segmentation baseline.

This is a from-scratch, understandable baseline — not a pretrained or
large foundation segmentation model (no SAM/SAM2). Encoder/decoder depth,
channel width, input channels, and output channels are all configurable so
the same architecture can be reused for different mask representations
(e.g. binary foreground/background now, multi-class later) without code
changes.
"""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two 3x3 conv + norm + ReLU blocks, the U-Net's basic unit."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Configurable U-Net: encoder with downsampling, decoder with skip connections.

    Args:
        in_channels: number of input image channels (e.g. 3 for RGB).
        out_channels: number of output segmentation channels/classes (e.g.
            1 for binary foreground/background logits).
        base_channels: channel width of the first encoder stage; doubles
            at each subsequent stage.
        depth: number of downsampling stages (encoder blocks below the
            bottleneck). Input spatial dimensions must be divisible by
            ``2 ** depth``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.depth = depth
        channels = [base_channels * (2**i) for i in range(depth + 1)]

        self.downs = nn.ModuleList()
        prev_channels = in_channels
        for c in channels[:-1]:
            self.downs.append(DoubleConv(prev_channels, c))
            prev_channels = c
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(channels[-2], channels[-1])

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        prev_channels = channels[-1]
        for c in reversed(channels[:-1]):
            self.up_convs.append(nn.ConvTranspose2d(prev_channels, c, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(c * 2, c))
            prev_channels = c

        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % (2**self.depth) != 0 or x.shape[-2] % (2**self.depth) != 0:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[-2:])} must be divisible by "
                f"2**depth ({2**self.depth}) for this U-Net configuration."
            )

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up_conv, up_block, skip in zip(self.up_convs, self.ups, reversed(skips), strict=True):
            x = up_conv(x)
            x = torch.cat([x, skip], dim=1)
            x = up_block(x)

        return self.out_conv(x)
