from typing import Any, List
import torch
import torch.nn as nn
from .FCLayers import FullyConnectedLayerSS
from .DownBlocks import DownBlock1SS
from .UpBlocks import UpBlock1SS


class GenerateEncoder(nn.Module):
    """
    Experiment Generator
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            FullyConnectedLayerSS(1, 8),
            FullyConnectedLayerSS(8, 256),
        )
        self.up = nn.Sequential(
            UpBlock1SS(4, 8),
            UpBlock1SS(8, 3),
        )
        # to deep will lead failure, if you want deeper add jump links

    def forward(*args) -> Any:
        self, x = args
        x = self.fc(x).view(-1, 4, 8, 8)  # N, C, H, W
        x = self.up(x)
        return x


class GenerateDecoder(nn.Module):
    """
    Experimental
    """

    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            DownBlock1SS(3, 8),
            DownBlock1SS(8, 4),
        )
        self.fc = nn.Sequential(
            FullyConnectedLayerSS(256, 8), FullyConnectedLayerSS(8, 2)
        )

    def forward(*args) -> Any:
        self, x = args
        x = self.down(x).view(-1, 256)
        x = self.fc(x)
        return x
