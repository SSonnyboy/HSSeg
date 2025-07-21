#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   vnet_hsseg.py
@Time    :   2025/07/13 21:41:16
@Author  :   biabuluo
@Version :   1.0
@Desc    :   None
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.uniform import Uniform


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization="none"):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == "batchnorm":
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == "groupnorm":
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == "instancenorm":
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != "none":
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization="none"):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != "none":
            ops.append(
                nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride)
            )
            if normalization == "batchnorm":
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == "groupnorm":
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == "instancenorm":
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(
                nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride)
            )

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNetEncoder(nn.Module):
    def __init__(self, n_channels, n_filters, normalization="none", has_dropout=False):
        super(VNetEncoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(
            1, n_channels, n_filters, normalization=normalization
        )
        self.block_one_dw = DownsamplingConvBlock(
            n_filters, 2 * n_filters, normalization=normalization
        )

        self.block_two = ConvBlock(
            2, n_filters * 2, n_filters * 2, normalization=normalization
        )
        self.block_two_dw = DownsamplingConvBlock(
            n_filters * 2, n_filters * 4, normalization=normalization
        )

        self.block_three = ConvBlock(
            3, n_filters * 4, n_filters * 4, normalization=normalization
        )
        self.block_three_dw = DownsamplingConvBlock(
            n_filters * 4, n_filters * 8, normalization=normalization
        )

        self.block_four = ConvBlock(
            3, n_filters * 8, n_filters * 8, normalization=normalization
        )
        self.block_four_dw = DownsamplingConvBlock(
            n_filters * 8, n_filters * 16, normalization=normalization
        )

        self.block_five = ConvBlock(
            3, n_filters * 16, n_filters * 16, normalization=normalization
        )
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]


class UpBlock3D(nn.Module):
    """Upsampling followed by ConvBlock (3D), compatible with provided ConvBlock."""

    def __init__(
        self,
        in_channels1,
        in_channels2,
        out_channels,
        n_stages=2,
        normalization="none",
        dropout_p=0.0,
        trilinear=True,
    ):
        super(UpBlock3D, self).__init__()
        self.trilinear = trilinear
        self.dropout_p = dropout_p

        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels1, in_channels2, kernel_size=2, stride=2
            )

        self.conv = ConvBlock(
            n_stages=n_stages,
            n_filters_in=in_channels2 * 2,
            n_filters_out=out_channels,
            normalization=normalization,
        )

        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x1, x2):
        if self.trilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)

        # handle shape mismatch (padding)
        if x1.shape[-3:] != x2.shape[-3:]:
            diff = [s2 - s1 for s1, s2 in zip(x1.shape[-3:], x2.shape[-3:])]
            x1 = F.pad(x1, [d // 2 for d in reversed(diff * 2)])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return self.dropout(x)


class AuxHead(nn.Module):
    def __init__(self, in_chnl, mid_chnl, num_classes, upsample_mode):
        super(AuxHead, self).__init__()
        self.conv1 = nn.Conv3d(in_chnl, mid_chnl, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(mid_chnl, num_classes, kernel_size=1)
        self.upsample_mode = upsample_mode

    def forward(self, x, shape):
        x = self.conv1(x)
        x = F.interpolate(
            x,
            size=shape,
            mode=self.upsample_mode,
            align_corners=False,
        )
        return self.conv2(x)


class VNetDecoder(nn.Module):
    def __init__(self, params, has_dropout=False):
        super(VNetDecoder, self).__init__()
        self.has_dropout = has_dropout
        self.params = params
        self.ft_chns = self.params["feature_chns"]
        self.n_class = self.params["class_num"]
        self.normalization = self.params["normalization"]
        self.upsample_mode = self.params.get("upsample_mode", "trilinear")

        assert len(self.ft_chns) == 5

        self.up1 = UpBlock3D(
            self.ft_chns[4],
            self.ft_chns[3],
            self.ft_chns[3],
            normalization=self.normalization,
        )
        self.up2 = UpBlock3D(
            self.ft_chns[3],
            self.ft_chns[2],
            self.ft_chns[2],
            normalization=self.normalization,
        )
        self.up3 = UpBlock3D(
            self.ft_chns[2],
            self.ft_chns[1],
            self.ft_chns[1],
            normalization=self.normalization,
        )
        self.up4 = UpBlock3D(
            self.ft_chns[1],
            self.ft_chns[0],
            self.ft_chns[0],
            normalization=self.normalization,
        )

        self.out_conv = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size=1)
        self.auxhead3 = AuxHead(
            self.ft_chns[3], self.ft_chns[0], self.n_class, self.upsample_mode
        )
        self.auxhead2 = AuxHead(
            self.ft_chns[2], self.ft_chns[0], self.n_class, self.upsample_mode
        )
        self.auxhead1 = AuxHead(
            self.ft_chns[1], self.ft_chns[0], self.n_class, self.upsample_mode
        )
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.drop = nn.AlphaDropout()

    def forward(self, features, shape):
        x0, x1, x2, x3, x4 = features

        x = self.up1(x4, x3)
        dp3_out_seg = self.auxhead3(x, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.auxhead2(x, shape)
        x = self.up3(x, x1)
        dp1_out_seg = self.auxhead1(x, shape)
        x = self.up4(x, x0)
        if self.has_dropout:
            x = self.dropout(x)
        dp0_out_seg = self.out_conv(x)

        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class VNet_HSSEG(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=2,
        n_filters=16,
        normalization="none",
        has_dropout=False,
    ):
        super(VNet_HSSEG, self).__init__()

        self.encoder = VNetEncoder(n_channels, n_filters, normalization, has_dropout)
        params = {
            "in_chns": n_channels,
            "feature_chns": [n_filters * (2**i) for i in range(5)],  # 16,32,64,128,256
            "class_num": n_classes,
            "normalization": normalization,
            "upsample_mode": "trilinear",  # or "nearest"
        }
        self.decoder = VNetDecoder(params, has_dropout=has_dropout)
        self.dropout = nn.Dropout3d(p=0.5)

    def forward(self, x, turnoff_drop=False, mode="eval", is_aug=True):
        if turnoff_drop:
            orig_dropout = self.encoder.has_dropout
            self.encoder.has_dropout = False
            self.decoder.has_dropout = False

        features = self.encoder(x)
        # if mode == "train" and is_aug:
        #     features = [self.dropout(x) for x in features]
        shape = x.shape[2:]
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            features, shape
        )
        if turnoff_drop:
            self.encoder.has_dropout = orig_dropout
            self.decoder.has_dropout = orig_dropout
        if mode == "train":
            return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg
        else:
            return dp0_out_seg


if __name__ == "__main__":
    x = torch.randn((2, 1, 112, 112, 80))
    model = VNet_HSSEG(
        1,
        2,
        normalization="batchnorm",
        has_dropout=True,
    )
    print([x.shape for x in model(x, mode="train")])
