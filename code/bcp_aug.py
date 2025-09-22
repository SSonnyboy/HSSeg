#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   bcp_aug.py
@Time    :   2025/04/26 13:43:30
@Author  :   biabuluo
@Version :   1.0
@Desc    :   BCP aug: https://github.com/DeepMed-Lab-ECNU/BCP
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert (
            inputs.size() == target.size()
        ), "predict & target shape do not match: {} vs {}".format(
            inputs.size(), target.size()
        )
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            # bug found by @CamillerFerros at github issue#25
            mask = mask.repeat(1, self.n_classes, 1, 1).type(torch.float32)
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes


def generate_mask_2d(img):
    batch_size, channel, img_x, img_y = (
        img.shape[0],
        img.shape[1],
        img.shape[2],
        img.shape[3],
    )
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w : w + patch_x, h : h + patch_y] = 0
    loss_mask[:, w : w + patch_x, h : h + patch_y] = 0
    return mask.long(), loss_mask.long()


def generate_mask_la(img, mask_ratio=2 / 3):
    batch_size, channel, img_x, img_y, img_z = (
        img.shape[0],
        img.shape[1],
        img.shape[2],
        img.shape[3],
        img.shape[4],
    )
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = (
        int(img_x * mask_ratio),
        int(img_y * mask_ratio),
        int(img_z * mask_ratio),
    )
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w : w + patch_pixel_x, h : h + patch_pixel_y, z : z + patch_pixel_z] = 0
    loss_mask[
        :, w : w + patch_pixel_x, h : h + patch_pixel_y, z : z + patch_pixel_z
    ] = 0
    return mask.long(), loss_mask.long()


def generate_mask_pan(img, patch_size=64):
    batch_l = img.shape[0]
    # batch_unlab = unimg.shape[0]
    loss_mask = torch.ones(batch_l, 96, 96, 96).cuda()
    # loss_mask_unlab = torch.ones(batch_unlab, 96, 96, 96).cuda()
    mask = torch.ones(96, 96, 96).cuda()
    w = np.random.randint(0, 96 - patch_size)
    h = np.random.randint(0, 96 - patch_size)
    z = np.random.randint(0, 96 - patch_size)
    mask[w : w + patch_size, h : h + patch_size, z : z + patch_size] = 0
    loss_mask[:, w : w + patch_size, h : h + patch_size, z : z + patch_size] = 0
    # loss_mask_unlab[:, w:w+patch_size, h:h+patch_size, z:z+patch_size] = 0
    # cordi = [w, h, z]
    return mask.long(), loss_mask.long()


def get_bcp_aug_2d(img_label, img_unlabel):
    l_a, l_b = img_label.chunk(2)
    u_a, u_b = img_unlabel.chunk(2)
    bcp_mask, bcp_loss_mask = generate_mask_2d(l_a)
    bcp_u = l_a * bcp_mask + u_a * (1 - bcp_mask)  # ua-前景
    bcp_l = u_b * bcp_mask + l_b * (1 - bcp_mask)  # lb-前景
    return bcp_u, bcp_l, bcp_mask, bcp_loss_mask


def get_bcp_y_2d(img_y, img_p, bcp_mask):
    y_a, y_b = img_y.chunk(2)
    p_a, p_b = img_p.chunk(2)
    bcp_p = y_a * bcp_mask + p_a * (1 - bcp_mask)  # ya-前景
    bcp_y = p_b * bcp_mask + y_b * (1 - bcp_mask)  # pb-前景
    return bcp_p, bcp_y


def bcp_loss(
    class_um,
    output_soft,
    img_l,
    patch_l,
    mask,
    l_weight=2 / 3,
    u_weight=1 / 3,
    if_unlab=False,
):
    dice_loss = DiceLoss(class_um)
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = u_weight, l_weight
    if if_unlab:
        image_weight, patch_weight = l_weight, u_weight
    patch_mask = 1 - mask
    loss_dice = (
        dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    )
    loss_dice += (
        dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1))
        * patch_weight
    )
    return loss_dice / 2
