from typing import Any
import torch.nn as nn
from .KeepSizeConvLayers import (
    KeepSizeConv2dLayerSS,
    KeepSizeConv2dLayerx2,
    KeepSizeConv2dLayerx2S,
    KeepSizeConv2dLayerx2SS,
)


class DownBlock(nn.Module):
    """
    Conv and downsample to line size 1/2 (area 1/4)
    use KeepSizeConv2dLayerx2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            KeepSizeConv2dLayerx2(in_channels, out_channels), nn.MaxPool2d(2, 2)
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class DownBlockS(nn.Module):
    """
    Conv and downsample to line size 1/2 (area 1/4)
    use KeepSizeConv2dLayerx2S
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            KeepSizeConv2dLayerx2S(in_channels, out_channels), nn.MaxPool2d(2, 2)
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class DownBlockSS(nn.Module):
    """
    Conv and downsample to line size 1/2 (area 1/4)
    use KeepSizeConv2dLayerx2SS
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            KeepSizeConv2dLayerx2SS(in_channels, out_channels), nn.MaxPool2d(2, 2)
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class DownBlock1SS(nn.Module):
    """
    Conv and downsample to line size 1/2 (area 1/4)
    use KeepSizeConv2dLayerSS
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            KeepSizeConv2dLayerSS(in_channels, out_channels), nn.MaxPool2d(2, 2)
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)
