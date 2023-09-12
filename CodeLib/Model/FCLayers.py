from typing import Any
import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    """
    Linear fully connected layer with dropout, batchnorm1d and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class FullyConnectedLayerS(nn.Module):
    """
    Linear fully connected layer with dropout and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)


class FullyConnectedLayerSS(nn.Module):
    """
    Linear fully connected layer with ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(*args) -> Any:
        self, x = args
        return self.layer(x)
