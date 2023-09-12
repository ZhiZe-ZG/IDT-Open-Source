from typing import Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from ..Model import FusionEncoder, FusionDecoder

class ReversibleFusion(pl.LightningModule):
    """
    uesd to generate reversible fusion.
    """

    def __init__(self, fusion_num: int = 2):
        super().__init__()
        self.fe = FusionEncoder(input_images=fusion_num)
        self.fd = FusionDecoder(output_images=fusion_num)
        self.criterion = nn.SmoothL1Loss()

    def forward(*args) -> Any:
        self, x = args
        out = self.fe(x)
        return out

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(*args: Any) -> Any:
        self, train_batch, _ = args  # the last parameter is batch_idx
        x = train_batch  # just input
        z = self.fe(x)
        y = self.fd(z)
        class_loss = self.criterion(
            y, x
        )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
        # flatten_loss = 1/torch.abs(z.view(z.shape[0],z.shape[1],-1) - z.mean(dim=[2,3])[:,:,None]).sum(dim=[1,2]).mean()
        loss = class_loss  # +0.1*flatten_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(*args: Any) -> Any:
        self, val_batch, _ = args  # the last parameter is batch_idx
        x = val_batch  # just input
        z = self.fe(x)
        y = self.fd(z)
        class_loss = self.criterion(
            y, x
        )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
        # flatten_loss = 1/torch.abs(z.view(z.shape[0],z.shape[1],-1) - z.mean(dim=[2,3])[:,:,None]).sum(dim=[1,2]).mean()
        loss = class_loss  # +0.1*flatten_loss
        self.log("val_loss", loss)
        return loss
