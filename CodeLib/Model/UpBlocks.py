from typing import Any
import torch.nn as nn
from .KeepSizeConvLayers import (
    KeepSizeConv2dLayerSS,
    KeepSizeConv2dLayerx2,
    KeepSizeConv2dLayerx2S,
    KeepSizeConv2dLayerx2SS,
)


class UpBlock(nn.Module):
    """
    Upsample x2 and conv(use KeepSizeConv2dLayerx2)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            KeepSizeConv2dLayerx2(in_channels, out_channels),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class UpBlockS(nn.Module):
    """
    Upsample x2 and conv(use KeepSizeConv2dLayerx2S)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            KeepSizeConv2dLayerx2S(in_channels, out_channels),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class UpBlockSS(nn.Module):
    """
    Upsample x2 and conv(use KeepSizeConv2dLayerx2SS)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            KeepSizeConv2dLayerx2SS(in_channels, out_channels),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class UpBlock1SS(nn.Module):
    """
    Upsample x2 and conv(use KeepSizeConv2dLayerSS)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            KeepSizeConv2dLayerSS(in_channels, out_channels),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)
