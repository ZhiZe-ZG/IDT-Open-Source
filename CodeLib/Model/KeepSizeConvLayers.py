from typing import Any
import torch.nn as nn
from math import sqrt, ceil


class KeepSizeConv2dLayer(nn.Module):
    """
    Conv2d layer but keep output size equal to input size.
    With dropout2d, batchnorm2d, ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class KeepSizeConv2dLayerSS(nn.Module):
    """
    Conv2d layer but keep output size equal to input size.
    With ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.Dropout2d(inplace=True),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class KeepSizeConv2dLayerx2(nn.Module):
    """
    Conv2d layer but keep output size equal to input size.
    Has 2 Conv2d layers.
    With dropout2d, batchnorm2d, ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = ceil(sqrt(in_channels * out_channels))
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class KeepSizeConv2dLayerx2S(nn.Module):
    """
    Conv2d layer but keep output size equal to input size.
    Has 2 Conv2d layers.
    With dropout2d, ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = ceil(sqrt(in_channels * out_channels))
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(inplace=True),
            nn.ReLU(inplace=True),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class KeepSizeConv2dLayerx2SS(nn.Module):
    """
    Conv2d layer but keep output size equal to input size.
    Has 2 Conv2d layers.
    With 1 ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = ceil(sqrt(in_channels * out_channels))
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)
