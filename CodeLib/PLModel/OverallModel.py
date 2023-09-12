from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from ..Model import OverAllModel
from torchvision.models import vgg16, resnet152

class OverAllModelPL(pl.LightningModule):

    def __init__(self,class_num:int):
        super().__init__()
        self.net = OverAllModel(class_num=class_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(*args) -> Any:
        self, x = args
        out = self.net(x)
        pre_label = torch.argmax(out, dim=1) # type:ignore
        return pre_label # type:ignore

    def configure_optimizers(self)->Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(*args: Any) -> Any:
        self, train_batch, _ = args  # the last parameter is batch_idx
        x, y = train_batch
        z = self.net(x)
        loss = self.criterion(z, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(*args: Any) -> Any:
        self, val_batch, _ = args  # the last parameter is batch_idx
        x, y = val_batch
        z = self.net(x)
        loss = self.criterion(z, y)
        self.log('val_loss', loss)
        return loss

class OverAllRESPL(pl.LightningModule):

    def __init__(self,class_num:int):
        super().__init__()
        self.net = resnet152(num_classes=class_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(*args) -> Any:
        self, x = args
        out = self.net(x)
        pre_label = torch.argmax(out, dim=1) # type:ignore
        return pre_label # type:ignore

    def configure_optimizers(self)->Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(*args: Any) -> Any:
        self, train_batch, _ = args  # the last parameter is batch_idx
        x, y = train_batch
        z = self.net(x)
        loss = self.criterion(z, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(*args: Any) -> Any:
        self, val_batch, _ = args  # the last parameter is batch_idx
        x, y = val_batch
        z = self.net(x)
        loss = self.criterion(z, y)
        self.log('val_loss', loss)
        return loss
    
class OverAllVGGPL(pl.LightningModule):

    def __init__(self,class_num:int):
        super().__init__()
        self.net = vgg16(num_classes=class_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(*args) -> Any:
        self, x = args
        out = self.net(x)
        pre_label = torch.argmax(out, dim=1) # type:ignore
        return pre_label # type:ignore

    def configure_optimizers(self)->Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(*args: Any) -> Any:
        self, train_batch, _ = args  # the last parameter is batch_idx
        x, y = train_batch
        z = self.net(x)
        loss = self.criterion(z, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(*args: Any) -> Any:
        self, val_batch, _ = args  # the last parameter is batch_idx
        x, y = val_batch
        z = self.net(x)
        loss = self.criterion(z, y)
        self.log('val_loss', loss)
        return loss