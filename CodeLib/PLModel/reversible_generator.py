from typing import Any, List
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from ..Model import GenerateDecoder, GenerateEncoder


class GeneratorPLModule(pl.LightningModule):
    """
    classification
    uesd to generate reversible generator.
    """

    def __init__(self):
        super().__init__()
        self.ge = GenerateEncoder()
        self.gd = GenerateDecoder()
        self.criterion = nn.CrossEntropyLoss()

    def forward(*args) -> Any:
        self, x = args
        out = self.ge(x)
        return out

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(*args: Any) -> Any:
        self, train_batch, _ = args  # the last parameter is batch_idx
        x = train_batch  # just input
        z = self.ge(x)
        y = self.gd(z)
        target = (x > 0.5).to(torch.long).squeeze()
        class_loss = self.criterion(
            y, target
        )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
        loss = class_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(*args: Any) -> Any:
        self, val_batch, _ = args  # the last parameter is batch_idx
        x = val_batch  # just input
        z = self.ge(x)
        y = self.gd(z)
        target = (x > 0.5).to(torch.long).squeeze()
        class_loss = self.criterion(
            y, target
        )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
        loss = class_loss
        self.log("val_loss", loss)
        return loss
