from typing import Any
import torch
import torch.nn as nn
from ..Model import DownBlock1SS, FullyConnectedLayerSS


class SeperateModel(nn.Module):
    """
    Structure like GenerateDecoder, but different.
    3 layer
    """

    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.down = nn.Sequential(
            DownBlock1SS(3, 8),
            DownBlock1SS(8, 16),
            DownBlock1SS(16, 16),
        )
        self.fc = nn.Sequential(
            FullyConnectedLayerSS(256, 16), FullyConnectedLayerSS(16, 2)
        )
        self.out = nn.Softmax(dim=1)

    def forward(*args) -> Any:
        self, x = args
        x = self.down(x).view(-1, 256)
        x = self.fc(x)
        x = self.out(x)
        return x

class SeperateModelS(nn.Module):
    """
    Structure like GenerateDecoder, but different.
    """

    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.down = nn.Sequential(
            DownBlock1SS(3, 4),
            DownBlock1SS(4, 4),
        )
        self.fc = nn.Sequential(
            FullyConnectedLayerSS(256, 16), FullyConnectedLayerSS(16, 2)
        )
        self.out = nn.Softmax(dim=1)

    def forward(*args) -> Any:
        self, x = args
        x = self.down(x).view(-1, 256)
        x = self.fc(x)
        x = self.out(x)
        return x
