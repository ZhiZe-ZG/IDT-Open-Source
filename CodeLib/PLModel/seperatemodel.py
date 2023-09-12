from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from ..Model import SeperateModelS, SeperateModel


class SeperateModelPL(pl.LightningModule):
    def __init__(self,feature_num:int):
        super().__init__()
        self.feature_num  = feature_num
        self.decoder_list = self.feature_generator_list = nn.ModuleList([SeperateModel() for _ in range(feature_num)]) # this just like list, but can trace parameters 
        self.criterion = nn.CrossEntropyLoss()
        self.label_mask = torch.bitwise_left_shift(torch.ones(feature_num).to(torch.long),torch.arange(feature_num))
        self.loss_list = []

    def forward(*args) -> Any:
        self, x = args
        out = [n(x) for n in self.decoder_list]
        pre = torch.stack(out).argmax(dim=2)
        pre = self.label_mask[:,None]*pre # type:ignore
        pre = pre.sum(dim=0) # type:ignore
        # out = torch.concat(out, dim=0)
        return pre

    def configure_optimizers(self)->Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(*args: Any) -> Any:
        self, train_batch, _ = args  # the last parameter is batch_idx
        x, y = train_batch
        out = [n(x) for n in self.decoder_list]
        out = torch.concat(out, dim=0)
        # prepare labels
        label_mask = self.label_mask.clone()
        label_mask = label_mask.to(y.get_device())
        y = torch.bitwise_and(y[:,None], label_mask).permute(1,0).reshape(-1)
        y = (y>0).to(torch.long)
        # print(out,y)
        loss = self.criterion(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(*args: Any) -> Any:
        self, val_batch, _ = args  # the last parameter is batch_idx
        x, y = val_batch
        out = [n(x) for n in self.decoder_list]
        outc = torch.concat(out, dim=0)
        # prepare labels
        label_mask = self.label_mask.clone()
        label_mask = label_mask.to(y.get_device())
        y = torch.bitwise_and(y[:,None], label_mask).permute(1,0)
        y = (y>0).to(torch.long)
        # loss list
        L = []
        for idx in range(len(out)):
            L.append(self.criterion(out[idx],y[idx]))
        self.loss_list = L
        yc = y.reshape(-1)
        # print(out,y)
        loss = self.criterion(outc, yc)
        self.log('val_loss', loss)
        return loss

class SeperateModelSPL(pl.LightningModule):
    def __init__(self,feature_num:int):
        super().__init__()
        self.feature_num  = feature_num
        self.decoder_list = self.feature_generator_list = nn.ModuleList([SeperateModelS() for _ in range(feature_num)]) # this just like list, but can trace parameters 
        self.criterion = nn.CrossEntropyLoss()
        self.label_mask = torch.bitwise_left_shift(torch.ones(feature_num).to(torch.long),torch.arange(feature_num))

    def forward(*args) -> Any:
        self, x = args
        out = [n(x) for n in self.decoder_list]
        pre = torch.stack(out).argmax(dim=2)
        pre = self.label_mask[:,None]*pre # type:ignore
        pre = pre.sum(dim=0) # type:ignore
        # out = torch.concat(out, dim=0)
        return pre

    def configure_optimizers(self)->Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(*args: Any) -> Any:
        self, train_batch, _ = args  # the last parameter is batch_idx
        x, y = train_batch
        out = [n(x) for n in self.decoder_list]
        out = torch.concat(out, dim=0)
        # prepare labels
        label_mask = self.label_mask.clone()
        label_mask = label_mask.to(y.get_device())
        y = torch.bitwise_and(y[:,None], label_mask).permute(1,0).reshape(-1)
        y = (y>0).to(torch.long)
        # print(out,y)
        loss = self.criterion(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(*args: Any) -> Any:
        self, val_batch, _ = args  # the last parameter is batch_idx
        x, y = val_batch
        out = [n(x) for n in self.decoder_list]
        out = torch.concat(out, dim=0)
        # prepare labels
        label_mask = self.label_mask.clone()
        label_mask = label_mask.to(y.get_device())
        y = torch.bitwise_and(y[:,None], label_mask).permute(1,0).reshape(-1)
        y = (y>0).to(torch.long)
        # print(out,y)
        loss = self.criterion(out, y)
        self.log('val_loss', loss)
        return loss