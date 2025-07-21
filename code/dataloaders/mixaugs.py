import numpy as np
import random
import torch
import scipy.stats as stats


def generate_mask_2d(img):
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


def generate_mask_3d(img):
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


# # # # # # # # # # # # # # # # # # # # #
# # 1. cutmix for 2d
# # # # # # # # # # # # # # # # # # # # #


def cut_mix(image, target):
    img_a, img_b = image.chunk(2)
    target_a, target_b = target.chunk(2)
    mask = generate_mask_2d(img_a)
    img_in = img_a * mask + img_b * (1 - mask)
    img_out = img_a * (1 - mask) + img_b * mask
    target_in = target_a * mask + target_b * (1 - mask)
    target_out = target_a * (1 - mask) + target_b * mask
    img_cm = torch.cat([img_in, img_out], dim=0)
    target_cm = torch.cat([target_in, target_out], dim=0)
    return img_cm, target_cm


# # # # # # # # # # # # # # # # # # # # #
# # 2. cutmix for 3d
# # # # # # # # # # # # # # # # # # # # #
def cut_mix_3d(image, target):
    img_a, img_b = image.chunk(2)
    target_a, target_b = target.chunk(2)
    mask = generate_mask_3d(img_a)
    img_in = img_a * mask + img_b * (1 - mask)
    img_out = img_a * (1 - mask) + img_b * mask
    target_in = target_a * mask + target_b * (1 - mask)
    target_out = target_a * (1 - mask) + target_b * mask
    img_cm = torch.cat([img_in, img_out], dim=0)
    target_cm = torch.cat([target_in, target_out], dim=0)
    return img_cm, target_cm
