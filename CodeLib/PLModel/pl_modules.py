# """
# Wrap pytorch models into pl modules.
# """

# import pytorch_lightning as pl
# from ..Model import UpBlockSS, DownBlockSS
# # from ..Model.models import (
# #     FusionDecoder,
# #     FusionEncoder,
# #     GenerateDecoder,
# #     GenerateEncoder,
# #     OverAllModel,
# #     GenerateDecode2Class,
# #     OverAllModel2,
# #     GenerateDecoderSS,
# #     GenerateEncoderSS,
# #     GenerateDecoderS,
# #     GenerateEncoderS,
# #     FullyConnectedLayerS,
# #     FullyConnectedLayer,
# #     GenerateEncoderE,
# #     GenerateDecoderE,
# # )
# # from ..Model import OverAllModelMNIST, GenerateDecode2ClassMNIST
# import torch.nn as nn
# import torch.optim as optim
# import torch


# class OverAllModelPL(pl.LightningModule):
#     def __init__(self, class_num: int):
#         super().__init__()
#         self.net = OverAllModel(class_num=class_num)
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         out = self.net(x)
#         return out

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         z = self.net(x)
#         loss = self.criterion(z, y)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         z = self.net(x)
#         loss = self.criterion(z, y)
#         self.log("val_loss", loss)
#         return loss


# class OverAllModelMNISTPL(pl.LightningModule):
#     def __init__(self, class_num: int):
#         super().__init__()
#         self.net = OverAllModelMNIST(class_num=class_num)
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         out = self.net(x)
#         return out

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         z = self.net(x)
#         loss = self.criterion(z, y)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         z = self.net(x)
#         loss = self.criterion(z, y)
#         self.log("val_loss", loss)
#         return loss


# class OverAllModel2PL(pl.LightningModule):
#     def __init__(self, class_num: int):
#         super().__init__()
#         self.net = OverAllModel2(class_num=class_num)
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         out = self.net(x)
#         return out

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         z = self.net(x)
#         loss = self.criterion(z, y)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         z = self.net(x)
#         loss = self.criterion(z, y)
#         self.log("val_loss", loss)
#         return loss


# class SeperateModelPL(pl.LightningModule):
#     def __init__(self, feature_num: int):
#         super().__init__()
#         self.feature_num = feature_num
#         self.decoder_list = self.feature_generator_list = nn.ModuleList(
#             [GenerateDecode2Class() for _ in range(feature_num)]
#         )  # this just like list, but can trace parameters
#         self.criterion = nn.CrossEntropyLoss()
#         self.label_mask = torch.bitwise_left_shift(
#             torch.ones(feature_num).to(torch.long), torch.arange(feature_num)
#         )

#     def forward(self, x):
#         out = [n(x) for n in self.decoder_list]
#         # out = torch.concat(out, dim=0)
#         return out

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         out = [n(x) for n in self.decoder_list]
#         out = torch.concat(out, dim=0)
#         # prepare labels
#         label_mask = self.label_mask.clone()
#         label_mask = label_mask.to(y.get_device())
#         y = torch.bitwise_and(y[:, None], label_mask).permute(1, 0).reshape(-1)
#         y = (y > 0).to(torch.long)
#         # print(out,y)
#         loss = self.criterion(out, y)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         out = [n(x) for n in self.decoder_list]
#         out = torch.concat(out, dim=0)
#         # prepare labels
#         label_mask = self.label_mask.clone()
#         label_mask = label_mask.to(y.get_device())
#         y = torch.bitwise_and(y[:, None], label_mask).permute(1, 0).reshape(-1)
#         y = (y > 0).to(torch.long)
#         # print(out,y)
#         loss = self.criterion(out, y)
#         self.log("val_loss", loss)
#         return loss


# from .reversible_generator import GenerateDecoder


# class SeperateModelMNISTPL(pl.LightningModule):
#     def __init__(self, feature_num: int):
#         super().__init__()
#         self.feature_num = feature_num
#         self.decoder_list = self.feature_generator_list = nn.ModuleList(
#             [GenerateDecoder() for _ in range(feature_num)]
#         )  # this just like list, but can trace parameters
#         self.criterion = nn.CrossEntropyLoss()
#         self.label_mask = torch.bitwise_left_shift(
#             torch.ones(feature_num).to(torch.long), torch.arange(feature_num)
#         )

#     def forward(self, x):
#         out = [n(x) for n in self.decoder_list]
#         # out = torch.concat(out, dim=0)
#         return out

#     # def parameters(self,recurse:bool=True):
#     #     '''
#     #     rewrite parameters.
#     #     '''
#     #     parameter_list = [[p for p in n.parameters()] for n in self.decoder_list]
#     #     parameters = sum(parameter_list,[])
#     #     return iter(parameters)

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         out = [n(x) for n in self.decoder_list]
#         out = torch.concat(out, dim=0)
#         # prepare labels
#         label_mask = self.label_mask.clone()
#         label_mask = label_mask.to(y.get_device())
#         y = torch.bitwise_and(y[:, None], label_mask).permute(1, 0).reshape(-1)
#         y = (y > 0).to(torch.long)
#         # print(out,y)
#         loss = self.criterion(out, y)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         out = [n(x) for n in self.decoder_list]
#         out = torch.concat(out, dim=0)
#         # prepare labels
#         label_mask = self.label_mask.clone()
#         label_mask = label_mask.to(y.get_device())
#         y = torch.bitwise_and(y[:, None], label_mask).permute(1, 0).reshape(-1)
#         y = (y > 0).to(torch.long)
#         # print(out,y)
#         loss = self.criterion(out, y)
#         self.log("val_loss", loss)
#         return loss


# class ReversibleGenerator(pl.LightningModule):
#     def __init__(self):
#         """
#         without BN
#         """
#         super().__init__()
#         self.ge = GenerateEncoder()
#         self.gd = GenerateDecoder()
#         # self.criterion = nn.MSELoss()
#         self.criterion = nn.L1Loss()

#     def forward(self, x):
#         out = self.ge(x)
#         return out

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx):
#         x = train_batch  # just input
#         z = self.ge(x)
#         y = self.gd(z)
#         loss = self.criterion(y, x)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x = val_batch  # just input
#         z = self.ge(x)
#         y = self.gd(z)
#         loss = self.criterion(y, x)
#         self.log("val_loss", loss)
#         return loss


# from ..Model import FullyConnectedLayerSS


# class ReversibleFC(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.ge = nn.Sequential(
#             FullyConnectedLayerSS(1, 8), FullyConnectedLayerSS(8, 64)
#         )
#         self.gd = nn.Sequential(
#             FullyConnectedLayerSS(64, 8), FullyConnectedLayerSS(8, 1)
#         )

#         # self.criterion = nn.MSELoss()
#         self.criterion = nn.L1Loss()

#     def forward(self, x):
#         out = self.ge(x)
#         return out

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx):
#         x = train_batch  # just input
#         z = self.ge(x)
#         y = self.gd(z)
#         loss = self.criterion(
#             y, x
#         )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x = val_batch  # just input
#         z = self.ge(x)
#         y = self.gd(z)
#         loss = self.criterion(
#             y, x
#         )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
#         self.log("val_loss", loss)
#         return loss


# class ReversibleFC2(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.ge = nn.Sequential(
#             FullyConnectedLayerSS(1, 8), FullyConnectedLayerSS(8, 64)
#         )
#         self.ge2 = nn.Sequential(UpBlockSS(1, 1))
#         self.gd2 = nn.Sequential(DownBlockSS(1, 1))
#         self.gd = nn.Sequential(
#             FullyConnectedLayerSS(64, 8), FullyConnectedLayerSS(8, 1)
#         )

#         # self.criterion = nn.MSELoss()
#         self.criterion = nn.L1Loss()

#     def forward(self, x):
#         out = self.ge(x)
#         return out

#     def configure_optimizers(self):
#         # optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         optimizer = optim.AdamW(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx):
#         x = train_batch  # just input
#         z = self.ge(x)
#         z = z.view(-1, 1, 8, 8)  # N, C, H, W
#         z = self.ge2(z)
#         z = self.gd2(z)
#         z = z.view(-1, 64)
#         y = self.gd(z)
#         loss = self.criterion(
#             y, x
#         )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x = val_batch  # just input
#         z = self.ge(x)
#         z = z.view(-1, 1, 8, 8)  # N, C, H, W
#         z = self.ge2(z)
#         z = self.gd2(z)
#         z = z.view(-1, 64)
#         y = self.gd(z)
#         loss = self.criterion(
#             y, x
#         )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
#         self.log("val_loss", loss)
#         return loss


# class ReversibleFC3(pl.LightningModule):
#     """
#     classification
#     uesd to generate reversible generator.
#     """

#     def __init__(self):
#         super().__init__()
#         self.ge = GenerateEncoderE()
#         self.gd = GenerateDecoderE()
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         out = self.ge(x)
#         return out

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx):
#         x = train_batch  # just input
#         z = self.ge(x)
#         y = self.gd(z)
#         class_loss = self.criterion(
#             y, (x > 0.5).to(torch.long).squeeze()
#         )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
#         # flatten_loss = 1/torch.abs(z.view(z.shape[0],z.shape[1],-1) - z.mean(dim=[2,3])[:,:,None]).sum(dim=[1,2]).mean()
#         loss = class_loss  # +0.1*flatten_loss
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x = val_batch  # just input
#         z = self.ge(x)
#         # print(z.shape)
#         y = self.gd(z)
#         class_loss = self.criterion(
#             y, (x > 0.5).to(torch.long).squeeze()
#         )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
#         # flatten_loss = 1/torch.abs(z.view(z.shape[0],z.shape[1],-1) - z.mean(dim=[2,3])[:,:,None]).sum(dim=[1,2]).mean()
#         loss = class_loss  # +0.1*flatten_loss
#         self.log("val_loss", loss)
#         return loss


# class ReversibleFusion(pl.LightningModule):
#     """
#     uesd to generate reversible fusion.
#     """

#     def __init__(self):
#         super().__init__()
#         self.fe = FusionEncoder(input_images=2)
#         self.fd = FusionDecoder(output_images=2)
#         self.criterion = nn.SmoothL1Loss()

#     def forward(self, x):
#         out = self.fe(x)
#         return out

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, train_batch, batch_idx):
#         x = train_batch  # just input
#         z = self.fe(x)
#         y = self.fd(z)
#         class_loss = self.criterion(
#             y, x
#         )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
#         # flatten_loss = 1/torch.abs(z.view(z.shape[0],z.shape[1],-1) - z.mean(dim=[2,3])[:,:,None]).sum(dim=[1,2]).mean()
#         loss = class_loss  # +0.1*flatten_loss
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x = val_batch  # just input
#         z = self.fe(x)
#         y = self.fd(z)
#         class_loss = self.criterion(
#             y, x
#         )  # (input, target) but at start ZhiZe write (target, input) what a stupid error!
#         # flatten_loss = 1/torch.abs(z.view(z.shape[0],z.shape[1],-1) - z.mean(dim=[2,3])[:,:,None]).sum(dim=[1,2]).mean()
#         loss = class_loss  # +0.1*flatten_loss
#         self.log("val_loss", loss)
#         return loss


# # class ReversibleFCClassify(pl.LightningModule):

# #     def __init__(self):
# #         '''
# #         in regression problem Active function in generator encoder could lead information lost
# #         so just keep classify reversible
# #         '''
# #         super().__init__()
# #         self.ge = nn.Sequential(
# #             nn.Linear(1, 8),
# #             nn.Sigmoid(),
# #             nn.Linear(8, 128),
# #             nn.Sigmoid(),
# #         )
# #         # nn.Sequential(
# #         #     FullyConnectedLayerSS(1,8),
# #         #     FullyConnectedLayerSS(8,128)
# #         # )
# #         self.gd =  nn.Sequential(
# #             nn.Linear(128, 8),
# #             nn.Sigmoid(),
# #             nn.Linear(8, 2),
# #             nn.Sigmoid(),
# #         )
# #         # nn.Sequential(
# #         #     FullyConnectedLayerSS(128,8),
# #         #     FullyConnectedLayerSS(8,1)
# #         # )
# #         # self.criterion = nn.MSELoss()
# #         self.criterion = nn.CrossEntropyLoss()


# #     def forward(self, x):
# #         out = self.ge(x)
# #         return out

# #     def configure_optimizers(self):
# #         optimizer = optim.Adam(self.parameters(), lr=1e-3)
# #         return optimizer

# #     def training_step(self, train_batch, batch_idx):
# #         x = train_batch # just input
# #         z = self.ge(x)
# #         y = self.gd(z)
# #         loss = self.criterion(x, y)
# #         self.log('train_loss', loss)
# #         return loss

# #     def validation_step(self, val_batch, batch_idx):
# #         x = val_batch # just input
# #         z = self.ge(x)
# #         y = self.gd(z)
# #         loss = self.criterion(x, y)
# #         self.log('val_loss', loss)
# #         return loss
