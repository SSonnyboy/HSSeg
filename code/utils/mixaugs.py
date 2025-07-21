import numpy as np
import random
import torch
import scipy.stats as stats


def generate_mask_acdc(img):
    batch_size, channel, img_x, img_y = (
        img.shape[0],
        img.shape[1],
        img.shape[2],
        img.shape[3],
    )
    # loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w : w + patch_x, h : h + patch_y] = 0
    return mask.long()


def generate_mask_la(img):
    batch_size, channel, img_x, img_y, img_z = (
        img.shape[0],
        img.shape[1],
        img.shape[2],
        img.shape[3],
        img.shape[4],
    )
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = (
        int(img_x * 2 / 3),
        int(img_y * 2 / 3),
        int(img_z * 2 / 3),
    )
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w : w + patch_pixel_x, h : h + patch_pixel_y, z : z + patch_pixel_z] = 0
    return mask.long()


def generate_mask_pan(img):
    mask = torch.ones(96, 96, 96).cuda()
    w = np.random.randint(0, 96 - 64)
    h = np.random.randint(0, 96 - 64)
    z = np.random.randint(0, 96 - 64)
    mask[w : w + 64, h : h + 64, z : z + 64] = 0
    return mask.long()


def generate_mask(img, mode="acdc"):
    if mode == "acdc":
        return generate_mask_acdc(img)
    elif mode == "la":
        return generate_mask_la(img)
    else:
        return generate_mask_pan(img)


# # # # # # # # # # # # # # # # # # # # #
# # 1. cutmix for single batch img
# # # # # # # # # # # # # # # # # # # # #


def cut_mix_single(image, mask):
    img_a, img_b = image.chunk(2)
    img_in = img_a * mask + img_b * (1 - mask)
    img_out = img_a * (1 - mask) + img_b * mask
    img_cm = torch.cat([img_in, img_out], dim=0)
    return img_cm


# # # # # # # # # # # # # # # # # # # # #
# # 1. cutmix
# # # # # # # # # # # # # # # # # # # # #


def cut_mix(image, target, mode="acdc"):
    img_a, img_b = image.chunk(2)
    target_a, target_b = target.chunk(2)
    mask = generate_mask(image, mode)
    img_in = img_a * mask + img_b * (1 - mask)
    img_out = img_a * (1 - mask) + img_b * mask
    target_in = target_a * mask + target_b * (1 - mask)
    target_out = target_a * (1 - mask) + target_b * mask
    img_cm = torch.cat([img_in, img_out], dim=0)
    target_cm = torch.cat([target_in, target_out], dim=0)
    return img_cm, target_cm
