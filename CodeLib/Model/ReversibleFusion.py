from typing import Any, List
import torch
import torch.nn as nn
from .DownBlocks import DownBlock1SS
from .UpBlocks import UpBlock1SS

class FusionEncoder(nn.Module):
    def __init__(self, input_images: int = 2, each_image_channel: int = 3):
        """
        assume input 2 image, each of them has 3 channel, and be concatenated.
        """
        super().__init__()
        total_input_channel = input_images * each_image_channel
        self.down = nn.Sequential(
            DownBlock1SS(total_input_channel, 16),
            DownBlock1SS(16, 32),
        )
        self.up = nn.Sequential(
            UpBlock1SS(32, 16),
            UpBlock1SS(16, 3),
        )

    def forward(*args) -> Any:
        self, x = args
        x = self.down(x)
        x = self.up(x)
        return x


class FusionDecoder(nn.Module):
    def __init__(self, output_images: int = 2, each_image_channel: int = 3):
        super().__init__()
        self.down = nn.Sequential(
            DownBlock1SS(3, 16),
            DownBlock1SS(16, 32),
        )
        self.up = nn.Sequential(
            UpBlock1SS(32, 16),
            UpBlock1SS(16, output_images * each_image_channel),
        )

    def forward(*args) -> Any:
        self, x = args
        x = self.down(x)
        x = self.up(x)
        return x
