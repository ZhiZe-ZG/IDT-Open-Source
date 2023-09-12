# """
# This file contains all the pytorch models.
# """

# import torch
# import torch.nn as nn
# from .FCLayers import (
#     FullyConnectedLayer,
#     FullyConnectedLayerS,
#     FullyConnectedLayerSS,
# )
# from .DownBlocks import (
#     DownBlock,
#     DownBlockS,
#     DownBlockSS,
#     DownBlock1SS,
#     KeepSizeConv2dLayerx2,
# )
# from .UpBlocks import (
#     UpBlock,
#     UpBlockS,
#     UpBlockSS,
#     UpBlock1SS,
# )
# from .KeepSizeConvLayers import (
#     KeepSizeConv2dLayerSS,
# )


# class GenerateEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Sequential(
#             FullyConnectedLayer(
#                 1, 128
#             ),  # to ensure the model is reversible, need a large net width to keep information
#             FullyConnectedLayer(128, 128),
#         )
#         self.up = nn.Sequential(UpBlock(2, 4), UpBlock(4, 16))
#         self.out = nn.Conv2d(16, 3, kernel_size=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         x = self.fc(x).view(-1, 2, 8, 8)  # N, C, H, W
#         x = self.up(x)
#         x = self.out(x)
#         return x


# class GenerateDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.inch = nn.Conv2d(3, 16, kernel_size=1)
#         self.down = nn.Sequential(DownBlock(16, 4), DownBlock(4, 2))
#         self.fc = nn.Sequential(FullyConnectedLayer(128, 8), FullyConnectedLayer(8, 1))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         x = self.inch(x)
#         x = self.down(x).view(-1, 128)
#         x = self.fc(x)
#         return x


# class GenerateEncoderS(nn.Module):
#     """
#     without BN
#     """

#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Sequential(
#             FullyConnectedLayerS(1, 128), FullyConnectedLayerS(128, 128)
#         )
#         self.up = nn.Sequential(UpBlockS(2, 4), UpBlockS(4, 16))
#         self.out = nn.Conv2d(16, 3, kernel_size=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         x = self.fc(x).view(-1, 2, 8, 8)  # N, C, H, W
#         x = self.up(x)
#         x = self.out(x)
#         return x


# class GenerateDecoderS(nn.Module):
#     """
#     without BN
#     """

#     def __init__(self):
#         super().__init__()
#         self.inch = nn.Conv2d(3, 16, kernel_size=1)
#         self.down = nn.Sequential(DownBlockS(16, 4), DownBlockS(4, 2))
#         self.fc = nn.Sequential(
#             FullyConnectedLayerS(128, 8), FullyConnectedLayerS(8, 1)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         x = self.inch(x)
#         x = self.down(x).view(-1, 128)
#         x = self.fc(x)
#         return x


# class GenerateEncoderSS(nn.Module):
#     """
#     without BN, dropout
#     """

#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Sequential(
#             FullyConnectedLayerSS(1, 8), FullyConnectedLayerSS(8, 128)
#         )
#         self.up = nn.Sequential(UpBlockSS(2, 4), UpBlockSS(4, 16))
#         self.out = nn.Conv2d(16, 3, kernel_size=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         x = self.fc(x).view(-1, 2, 8, 8)  # N, C, H, W
#         x = self.up(x)
#         x = self.out(x)
#         return x


# class GenerateDecoderSS(nn.Module):
#     """
#     without BN, dropout
#     """

#     def __init__(self):
#         super().__init__()
#         self.inch = nn.Conv2d(3, 16, kernel_size=1)
#         self.down = nn.Sequential(DownBlockSS(16, 4), DownBlockSS(4, 2))
#         self.fc = nn.Sequential(
#             FullyConnectedLayerSS(128, 8), FullyConnectedLayerSS(8, 1)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         x = self.inch(x)
#         x = self.down(x).view(-1, 128)
#         x = self.fc(x)
#         return x


# class GenerateDecode2Class(nn.Module):
#     """instead get the origin feature, this model get a 2 class feature."""

#     def __init__(self):
#         super().__init__()
#         self.inch = nn.Conv2d(3, 16, kernel_size=1)
#         self.down = nn.Sequential(DownBlock(16, 4), DownBlock(4, 2))
#         self.fc = nn.Sequential(FullyConnectedLayer(128, 8), FullyConnectedLayer(8, 2))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         x = self.inch(x)
#         x = self.down(x).view(-1, 128)
#         x = self.fc(x)
#         return x


# class GenerateDecode2ClassMNIST(nn.Module):
#     """instead get the origin feature, this model get a 2 class feature."""

#     def __init__(self):
#         super().__init__()
#         # self.inch = nn.Conv2d(3,16,kernel_size=1)
#         self.down = nn.Sequential(DownBlock1SS(3, 8), DownBlock1SS(8, 2))
#         self.fc = nn.Sequential(
#             FullyConnectedLayerSS(32, 4), FullyConnectedLayerSS(4, 2)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         # x = self.inch(x)
#         x = self.down(x).view(-1, 32)
#         x = self.fc(x)
#         return x


# class FusionEncoder(nn.Module):
#     def __init__(self, input_images: int = 2, each_image_channel: int = 3):
#         """
#         assume input 2 image, each of them has 3 channel, and be concatenated.
#         """
#         super().__init__()
#         total_input_channel = input_images * each_image_channel
#         # self.inch = nn.Conv2d(total_input_channel,total_input_channel*2,kernel_size=1)
#         self.down = nn.Sequential(
#             DownBlock(total_input_channel, 16),
#             # DownBlock(16,32),
#             DownBlock(16, 32),
#         )
#         self.up = nn.Sequential(
#             UpBlock(32, 16),
#             # UpBlock(32,16),
#             UpBlock(16, 3),
#         )
#         # self.out = nn.Conv2d(8,3,kernel_size=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         # x = self.inch(x)
#         x = self.down(x)
#         x = self.up(x)
#         # x = self.out(x)
#         return x


# class FusionDecoder(nn.Module):
#     def __init__(self, output_images: int = 2, each_image_channel: int = 3):
#         super().__init__()
#         # self.inch = nn.Conv2d(3,8,kernel_size=1)
#         self.down = nn.Sequential(
#             DownBlock(3, 16),
#             # DownBlock(16,32),
#             DownBlock(16, 32),
#         )
#         self.up = nn.Sequential(
#             UpBlock(32, 16),
#             # UpBlock(32,16),
#             UpBlock(16, output_images * each_image_channel),
#         )
#         # self.out = nn.Conv2d(8,output_images*each_image_channel,kernel_size=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         # x = self.inch(x)
#         x = self.down(x)
#         x = self.up(x)
#         # x = self.out(x)
#         return x


# class OverAllModel(nn.Module):
#     """
#     Structure like GenerateDecoder, but different.
#     """

#     def __init__(self, class_num: int):
#         super().__init__()
#         self.class_num = class_num
#         self.inch = nn.Conv2d(3, 16, kernel_size=1)
#         self.down = nn.Sequential(DownBlock1SS(16, 4), DownBlock1SS(4, 2))
#         self.fc = nn.Sequential(
#             FullyConnectedLayerSS(128, 8), FullyConnectedLayerSS(8, class_num)
#         )
#         self.out = nn.Softmax(dim=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         x = self.inch(x)
#         x = self.down(x).view(-1, 128)
#         x = self.fc(x)
#         x = self.out(x)
#         return x


# class OverAllModelMNIST(nn.Module):
#     """
#     Structure like GenerateDecoder, but different.
#     """

#     def __init__(self, class_num: int):
#         super().__init__()
#         self.class_num = class_num
#         # self.inch = nn.Conv2d(3,4,kernel_size=1)
#         self.down = nn.Sequential(
#             DownBlockSS(3, 4), DownBlockSS(4, 4), DownBlockSS(4, 2)
#         )
#         self.fc = nn.Sequential(
#             FullyConnectedLayerSS(32, 16), FullyConnectedLayerSS(16, class_num)
#         )
#         self.out = nn.Softmax(dim=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         # x = self.inch(x)
#         x = self.down(x).view(-1, 32)
#         x = self.fc(x)
#         x = self.out(x)
#         return x


# class OverAllModel2(nn.Module):
#     """
#     Structure like FusionDecoder plus GenerateDecoder, but different.
#     """

#     def __init__(self, class_num: int):
#         super().__init__()
#         self.class_num = class_num
#         self.inch = nn.Conv2d(3, 16, kernel_size=1)
#         self.body = nn.Sequential(
#             KeepSizeConv2dLayerSS(16, 64),
#             KeepSizeConv2dLayerSS(64, 64),
#             KeepSizeConv2dLayerSS(64, 16),
#         )
#         self.down = nn.Sequential(DownBlock1SS(16, 4), DownBlock1SS(4, 2))
#         self.fc = nn.Sequential(
#             FullyConnectedLayerSS(128, 8), FullyConnectedLayerSS(8, class_num)
#         )
#         self.out = nn.Softmax(dim=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         x = self.inch(x)
#         x = self.down(x).view(-1, 128)
#         x = self.fc(x)
#         x = self.out(x)
#         return x


# class GenerateEncoderE(nn.Module):
#     """
#     Experiment Generator
#     """

#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Sequential(
#             FullyConnectedLayerSS(1, 8),
#             FullyConnectedLayerSS(8, 256),
#         )
#         self.up = nn.Sequential(
#             UpBlockSS(4, 8),
#             # UpBlockSS(8,8),
#             UpBlockSS(8, 3),
#         )
#         # to deep will lead failure, if you want deeper add jump links

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         x = self.fc(x).view(-1, 4, 8, 8)  # N, C, H, W
#         x = self.up(x)
#         # x = self.out(x)
#         return x


# class GenerateDecoderE(nn.Module):
#     """
#     Experimental
#     """

#     def __init__(self):
#         super().__init__()
#         # self.inch = nn.Conv2d(3,16,kernel_size=1)
#         self.down = nn.Sequential(
#             DownBlockSS(3, 8),
#             # DownBlockSS(8,8),
#             DownBlockSS(8, 4),
#         )
#         self.fc = nn.Sequential(
#             FullyConnectedLayerSS(256, 8), FullyConnectedLayerSS(8, 2)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         To fit the prototype of forward in current pytorch, I need rewrite this function like this:
#         def forward(*args)->Any:
#         This is ridculous. So the error hint can not be fixed.
#         """
#         # x = self.inch(x)
#         x = self.down(x).view(-1, 256)
#         x = self.fc(x)
#         return x
