#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   sp.py
@Time    :   2025/04/19 11:04:06
@Author  :   biabuluo
@Version :   1.0
@Desc    :   None
"""
import torch


def sp_loss(g_s, g_t):
    return sum([similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])


def similarity_loss(f_s, f_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss


if __name__ == "__main__":
    x1 = torch.rand((10, 256, 16, 16))
    x2 = torch.rand((10, 256, 16, 16))
    print(100 * sp_loss(x1, x2))
