#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   unet_hsseg.py
@Time    :   2025/07/13 20:38:55
@Author  :   biabuluo
@Version :   1.0
@Desc    :   None
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(
        self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True
    ):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2
            )
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params["in_chns"]
        self.ft_chns = self.params["feature_chns"]
        self.n_class = self.params["class_num"]
        self.bilinear = self.params["bilinear"]
        self.dropout = self.params["dropout"]
        assert len(self.ft_chns) == 5
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class AuxHead(nn.Module):
    def __init__(self, in_chnl, mid_chnl, num_classes):
        super(AuxHead, self).__init__()
        self.conv1 = nn.Conv2d(in_chnl, mid_chnl, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_chnl, num_classes, kernel_size=3, padding=1)

    def forward(self, x, shape):
        x = self.conv1(x)
        x = torch.nn.functional.interpolate(x, shape)
        return self.conv2(x)


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params["in_chns"]
        self.ft_chns = self.params["feature_chns"]
        self.n_class = self.params["class_num"]
        self.bilinear = self.params["bilinear"]
        assert len(self.ft_chns) == 5

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0
        )
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0
        )
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0
        )
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0
        )

        self.out_conv = nn.Conv2d(
            self.ft_chns[0], self.n_class, kernel_size=3, padding=1
        )
        self.auxhead3 = AuxHead(self.ft_chns[3], self.ft_chns[0], self.n_class)
        self.auxhead2 = AuxHead(self.ft_chns[2], self.ft_chns[0], self.n_class)
        self.auxhead1 = AuxHead(self.ft_chns[1], self.ft_chns[0], self.n_class)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x5 = self.up1(x4, x3)
        dp3_out_seg = self.auxhead3(x5, shape)

        x6 = self.up2(x5, x2)
        dp2_out_seg = self.auxhead2(x6, shape)

        x7 = self.up3(x6, x1)
        dp1_out_seg = self.auxhead1(x7, shape)

        x8 = self.up4(x7, x0)
        dp0_out_seg = self.out_conv(x8)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class UNet_HSSEG(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_HSSEG, self).__init__()

        params = {
            "in_chns": in_chns,
            "feature_chns": [16, 32, 64, 128, 256],
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num": class_num,
            "bilinear": False,
            "acti_func": "relu",
        }
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x, mode="eval", is_aug=False):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape
        )
        if mode == "train":
            return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg
        else:
            return dp0_out_seg


if __name__ == "__main__":
    x = torch.randn((2, 1, 256, 256))
    model = UNet_HSSEG(
        1,
        4,
    )
    print([x.shape for x in model(x, mode="train")])
