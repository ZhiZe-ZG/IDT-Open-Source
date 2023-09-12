from typing import Any
import torch
import torch.nn as nn
from .DownBlocks import DownBlock1SS
from .FCLayers import FullyConnectedLayerSS


class OverAllModel(nn.Module):
    """
    Structure like GenerateDecoder, but different.
    """

    def __init__(self, class_num: int):
        super().__init__()
        self.class_num = class_num
        self.down = nn.Sequential(
            DownBlock1SS(3, 8),
            DownBlock1SS(8, 16),
            DownBlock1SS(16, 32),
            DownBlock1SS(32, 64),
        )
        self.fc = nn.Sequential(
            FullyConnectedLayerSS(256, 128), FullyConnectedLayerSS(128, class_num)
        )
        self.out = nn.Softmax(dim=1)

    def forward(*args) -> Any:
        self, x = args
        x = self.down(x).view(-1, 256)
        x = self.fc(x)
        x = self.out(x)
        return x

