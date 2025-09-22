import argparse
import logging
import os
import os.path as osp
import random
import sys
import yaml

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.mixaugs import cut_mix_3d
from dataloaders.dataset_3d import (
    LAHeart,
    Pancreas,
    WeakStrongAugment,
    TwoStreamBatchSampler,
)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.util import update_values, time_str, AverageMeter
from val_3D import var_all_case_LA, var_all_case_Pancrease
from utils.mixaugs import *
from mainloss import RNKCLoss3D
from decay import *


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        I. helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args["consistency"] * ramps.sigmoid_rampup(epoch, args["consistency_rampup"])


def update_ema_variables(model, ema_model, alpha, global_step, args):
    # adjust the momentum param
    if global_step < args["consistency_rampup"]:
        alpha = 0.0
    else:
        alpha = min(1 - 1 / (global_step - args["consistency_rampup"] + 1), alpha)

    # update weights
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    # update buffers
    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.data = buffer_eval.data * alpha + buffer_train.data * (1 - alpha)


def kl_tmp(pred, target, ignore=None):
    if ignore is not None:
        pred = pred * (
            1 - ignore.unsqueeze(1)
        )  # Mask out regions of interest in prob_u
        target = target * (
            1 - ignore.unsqueeze(1)
        )  # Mask out regions of interest in prob_t
    kl_loss = F.kl_div(torch.log(pred + 1e-6), target, reduction="batchmean")
    return kl_loss


# Cross-Entropy Loss function
def ce_loss_tmp(pred, target, ignore=None):
    if ignore is not None:
        ignore = ignore.unsqueeze(1).float()  # Ensure ignore has the right shape
        pred = pred * (1 - ignore)  # Ignore predictions in the mask region
        target = target * (1 - ignore)  # Ignore the target in the mask region

    return F.cross_entropy(
        pred, target.squeeze(1).long()
    )  # Squeeze target if necessary


def mse_loss_tmp(pred, target, ignore=None):
    if ignore is not None:
        ignore = ignore.unsqueeze(1)
        pred = pred * (1 - ignore)
        target = target * (1 - ignore)  # Ignore the target in the mask region

    # MSE Loss calculation
    return F.mse_loss(pred, target, reduction="mean")  # We use mean reduction for MSE


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        II. trainer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def train(args, snapshot_path):
    base_lr = args["base_lr"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
    num_classes = args["num_classes"]

    loss_type = args["loss_type"]  # rnkc
    hist_weight = args["hist_weight"]  # 0.4
    gamma = args["gamma"]  # 0.05
    train_mode = args["train_mode"]  # 1-6
    stage_k = args["stage_k"]  # 1-3
    threshold = args["threshold"]

    cur_time = time_str()
    writer = SummaryWriter(snapshot_path + "/log")
    csv_train = os.path.join(
        snapshot_path, "log", "seg_{}_train_iter.csv".format(cur_time)
    )
    csv_test = os.path.join(
        snapshot_path, "log", "seg_{}_validate_ep.csv".format(cur_time)
    )

    def create_model(ema=False):
        # Network definition
        net = net_factory(net_type=args["model"], in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args["seed"] + worker_id)

    # + + + + + + + + + + + #
    # 1. create model
    # + + + + + + + + + + + #
    model = create_model()
    ema_model = create_model(ema=True)
    model.train()
    ema_model.train()

    # + + + + + + + + + + + #
    # 2. dataset
    # + + + + + + + + + + + #
    fdloader = LAHeart
    flag_pancreas = True if "pancreas" in args["root_path"].lower() else False
    if flag_pancreas:
        fdloader = Pancreas
    db_train = fdloader(
        base_dir=args["root_path"],
        split="train",
        num=None,
        transform=transforms.Compose(
            [WeakStrongAugment(args["patch_size"], flag_rot=not flag_pancreas)]
        ),
    )

    labeled_idxs = list(range(0, args["labeled_num"]))
    unlabeled_idxs = list(range(args["labeled_num"], args["max_samples"]))

    batch_sampler = TwoStreamBatchSampler(
        unlabeled_idxs, labeled_idxs, batch_size, args["labeled_bs"]
    )

    # + + + + + + + + + + + #
    # 3. dataloader
    # + + + + + + + + + + + #
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=args["workers"],
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    logging.info("{} iterations per epoch".format(len(trainloader)))

    # + + + + + + + + + + + #
    # 4. optim, scheduler
    # + + + + + + + + + + + #
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    )
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    dice_loss_ignore = losses.DiceLossNew(num_classes)
    rnkcloss = RNKCLoss3D(gamma_base=gamma)

    def compute_label_loss(segs, target_lb):
        target_lb_long = target_lb.long()
        target_lb_float = target_lb.unsqueeze(1).float()
        loss = 0.0
        weights = linear_decay_weights(len(segs))
        for seg, alpha in zip(segs, weights):
            ce = ce_loss(seg, target_lb_long)
            dice = dice_loss(torch.softmax(seg, dim=1), target_lb_float)
            loss += alpha * ((ce + dice) / 2.0)
        return loss

    def compute_ulb_main_loss(seg_u, seg_t, threshold=0.95, loss_type="rnkc"):
        prob = torch.softmax(seg_t, dim=1)
        pred = torch.argmax(prob, dim=1).unsqueeze(1).float()
        mask = (torch.max(prob.detach(), dim=1)[0] < threshold).float()
        loss = dice_loss_ignore(torch.softmax(seg_u, dim=1), pred, ignore=mask)
        if loss_type == "rnkc":
            loss += rnkcloss(seg_u, seg_t, mask)
        elif loss_type == "mse":
            loss += mse_loss_tmp(torch.softmax(seg_u, dim=1), prob, ignore=mask)
        elif loss_type == "ce":
            loss += ce_loss_tmp(torch.softmax(seg_u, dim=1), pred, ignore=mask)
        elif loss_type == "kl":
            loss += kl_tmp(torch.softmax(seg_u, dim=1), prob, ignore=mask)
        else:
            loss += loss
        return loss / 2.0

    def compute_ulb_his_loss(segs_u, his_segs_u, threshold=0.95, loss_type="rnkc"):
        loss = 0.0
        weights = linear_decay_weights(len(segs_u))
        for seg_u, his_u, alpha in zip(segs_u, his_segs_u, weights):
            prob = torch.softmax(his_u, dim=1)
            pred = torch.argmax(prob, dim=1).unsqueeze(1).float()
            mask = (torch.max(prob.detach(), dim=1)[0] < threshold).float()
            # print(torch.softmax(seg_u, dim=1).shape, pred.shape, mask.shape)
            loss_item = dice_loss_ignore(torch.softmax(seg_u, dim=1), pred, ignore=mask)
            if loss_type == "rnkc":
                loss_item += rnkcloss(seg_u, his_u, mask)
            elif loss_type == "mse":
                loss_item += mse_loss_tmp(
                    torch.softmax(seg_u, dim=1), prob, ignore=mask
                )
            elif loss_type == "ce":
                loss_item += ce_loss_tmp(torch.softmax(seg_u, dim=1), pred, ignore=mask)
            elif loss_type == "kl":
                loss += kl_tmp(torch.softmax(seg_u, dim=1), prob, ignore=mask)
            else:
                loss_item += loss_item
            loss += alpha * (loss_item / 2.0)
        return loss

    def compute_ulb_stru_loss(segs_u, ref_seg_u, threshold=0.95, loss_type="rnkc"):
        ref_prob = torch.softmax(ref_seg_u, dim=1).detach()
        ref_pred = torch.argmax(ref_prob, dim=1).unsqueeze(1).float()
        ref_mask = (torch.max(ref_prob, dim=1)[0] < threshold).float()
        weights = linear_decay_weights(len(segs_u))
        loss = 0.0
        for seg_u, alpha in zip(segs_u, weights):
            loss_item = dice_loss_ignore(
                torch.softmax(seg_u, dim=1), ref_pred, ignore=ref_mask
            )
            if loss_type == "rnkc":
                loss_item += rnkcloss(seg_u, ref_seg_u, ref_mask)
            elif loss_type == "mse":
                loss_item += mse_loss_tmp(
                    torch.softmax(seg_u, dim=1), ref_prob, ignore=ref_mask
                )
            elif loss_type == "ce":
                loss_item += ce_loss_tmp(
                    torch.softmax(seg_u, dim=1), ref_pred, ignore=ref_mask
                )
            elif loss_type == "kl":
                loss += kl_tmp(torch.softmax(seg_u, dim=1), ref_prob, ignore=ref_mask)
            else:
                loss_item += loss_item
            loss += alpha * (loss_item / 2.0)
        return loss

    # + + + + + + + + + + + #
    # 5. training loop
    # + + + + + + + + + + + #
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_dice = 0.0
    best_dice_stu = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        # metric indicators
        meter_sup_losses = AverageMeter()
        meter_uns_losses = AverageMeter(20)
        meter_train_losses = AverageMeter(20)
        meter_learning_rates = AverageMeter()
        for i_batch, sampled_batch in enumerate(trainloader):
            num_lb = args["labeled_bs"]
            num_ulb = batch_size - num_lb

            # 1) get augmented data
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            img_lb_w, target_lb = weak_batch[num_ulb:], label_batch[num_ulb:]
            img_ulb_w, img_ulb_s = weak_batch[:num_ulb], strong_batch[:num_ulb]
            # ss aug
            mode = "pan" if flag_pancreas else "la"
            if random.random() < 0.5:  # for img_lb_w
                img_lb_w, target_lb = cut_mix(img_lb_w, target_lb, mode=mode)
            # for label
            seg0, seg1, seg2, seg3 = model(img_lb_w, mode="train", is_aug=True)
            cm_flag = random.random() < 0.5
            if cm_flag:
                del img_ulb_s
                cm_mask = generate_mask(img_ulb_w, mode=mode)
                img_ulb_s = cut_mix_single(img_ulb_w, cm_mask)
            # for unlabel
            seg0_u, seg1_u, seg2_u, seg3_u = model(img_ulb_s, mode="train", is_aug=True)
            with torch.no_grad():
                if train_mode in (1, 2):
                    seg0_his_u, seg1_his_u, seg2_his_u, seg3_his_u = model(
                        img_ulb_w, mode="train"
                    )
                else:
                    if train_mode in (4, 5, 6):
                        seg0_stru_u = model(img_ulb_w, mode="eval")
                    seg0_his_u, seg1_his_u, seg2_his_u, seg3_his_u = ema_model(
                        img_ulb_w, mode="train"
                    )
            if cm_flag:
                if train_mode in (4, 5, 6):
                    seg0_stru_u = cut_mix_single(seg0_stru_u, cm_mask)
                seg3_his_u = cut_mix_single(seg3_his_u, cm_mask)
                seg2_his_u = cut_mix_single(seg2_his_u, cm_mask)
                seg1_his_u = cut_mix_single(seg1_his_u, cm_mask)
                seg0_his_u = cut_mix_single(seg0_his_u, cm_mask)

            segs = [seg0, seg1, seg2, seg3]
            segs_u = [seg0_u, seg1_u, seg2_u, seg3_u]
            his_segs_u = [seg0_his_u, seg1_his_u, seg2_his_u, seg3_his_u]

            # only test in mode 6
            if stage_k == 1:
                segs = segs[:1]
                segs_u = segs_u[:1]
                his_segs_u = his_segs_u[:1]
            elif stage_k == 2:
                segs = segs[:2]
                segs_u = segs_u[:2]
                his_segs_u = his_segs_u[:2]
            elif stage_k == 3:
                segs = segs[:3]
                segs_u = segs_u[:3]
                his_segs_u = his_segs_u[:3]
            elif stage_k == 4:
                segs = segs[:]
                segs_u = segs_u[:]
                his_segs_u = his_segs_u[:]

            if train_mode == 1:
                segs = [seg0]
                loss_type = "none"
            elif train_mode == 2:
                segs = [seg0]
                loss_type = "rnkc"
            elif train_mode in (3, 4, 5):
                loss_type = "none"
            elif train_mode == 6:
                loss_type = "rnkc"

            loss_lb = compute_label_loss(segs, target_lb)

            # 6) unsupervised loss
            if iter_num < 1000:
                loss_ulb = torch.tensor(0.0)
            else:
                if train_mode == 1:
                    segs = [seg0]
                    loss_type = "none"
                    loss_ulb = compute_ulb_main_loss(
                        seg0_u, seg0_his_u, loss_type=loss_type, threshold=threshold
                    )
                elif train_mode == 2:
                    segs = [seg0]
                    loss_type = "rnkc"
                    loss_ulb = compute_ulb_main_loss(
                        seg0_u, seg0_his_u, loss_type=loss_type, threshold=threshold
                    )
                elif train_mode == 3:
                    loss_type = "none"
                    loss_ulb = compute_ulb_his_loss(
                        segs_u, his_segs_u, loss_type=loss_type, threshold=threshold
                    )
                elif train_mode == 4:
                    loss_type = "none"
                    loss_ulb = compute_ulb_stru_loss(
                        segs_u,
                        seg0_stru_u.detach(),
                        loss_type=loss_type,
                        threshold=threshold,
                    )
                elif train_mode == 5:
                    loss_type = "none"
                    loss_ulb_his = compute_ulb_his_loss(
                        segs_u, his_segs_u, loss_type=loss_type, threshold=threshold
                    )

                    loss_ulb_stru = compute_ulb_stru_loss(
                        segs_u,
                        seg0_stru_u.detach(),
                        loss_type=loss_type,
                        threshold=threshold,
                    )
                    loss_ulb = (
                        hist_weight * (loss_ulb_his) + (1 - hist_weight) * loss_ulb_stru
                    )
                elif train_mode == 6:
                    loss_type = "rnkc"
                    loss_ulb_his = compute_ulb_his_loss(
                        segs_u, his_segs_u, loss_type=loss_type, threshold=threshold
                    )

                    loss_ulb_stru = compute_ulb_stru_loss(
                        segs_u,
                        seg0_stru_u.detach(),
                        loss_type=loss_type,
                        threshold=threshold,
                    )
                    loss_ulb = (
                        hist_weight * (loss_ulb_his) + (1 - hist_weight) * loss_ulb_stru
                    )
            # 7) total loss
            consistency_weight = get_current_consistency_weight(iter_num // 150, args)
            loss = loss_lb + consistency_weight * loss_ulb

            # 8) update student model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 9) update teacher model
            update_ema_variables(model, ema_model, args["ema_decay"], iter_num, args)
            # 10) udpate learing rate
            if args["poly"]:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_
            else:
                lr_ = base_lr

            # 11) record statistics
            iter_num = iter_num + 1
            # --- a) writer
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/loss_lb", loss_lb, iter_num)
            writer.add_scalar("info/loss_ulb", loss_ulb, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            # --- b) loggers
            logging.info(
                "iteration:{}  t-loss:{:.4f}, loss-lb:{:.4f}, loss-ulb:{:.4f}, weight:{:.2f}, lr:{:.4f}".format(
                    iter_num,
                    loss.item(),
                    loss_lb.item(),
                    loss_ulb.item(),
                    consistency_weight,
                    lr_,
                )
            )
            # --- c) avg meters
            meter_sup_losses.update(loss_lb.item())
            meter_uns_losses.update(loss_ulb.item())
            meter_train_losses.update(loss.item())
            meter_learning_rates.update(lr_)

            # --- d) csv
            tmp_results = {
                "loss_total": loss.item(),
                "loss_lb": loss_lb.item(),
                "loss_ulb": loss_ulb.item(),
                "lweight_ub": consistency_weight,
                "lr": lr_,
            }
            data_frame = pd.DataFrame(
                data=tmp_results, index=range(iter_num, iter_num + 1)
            )
            if iter_num > 1 and osp.exists(csv_train):
                data_frame.to_csv(csv_train, mode="a", header=None, index_label="iter")
            else:
                data_frame.to_csv(csv_train, index_label="iter")

            if iter_num >= max_iterations:
                break

        # 12) validating
        if (
            epoch_num % args.get("test_interval_ep", 1) == 0
            or iter_num >= max_iterations
        ):
            model.eval()
            ema_model.eval()

            if "pancreas" in args["root_path"].lower():
                dice_sample_stu = var_all_case_Pancrease(
                    model,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=16,
                    stride_z=16,
                    flag_nms=True,
                )
                dice_sample = var_all_case_Pancrease(
                    ema_model,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=16,
                    stride_z=16,
                    flag_nms=True,
                )
            else:
                dice_sample_stu = var_all_case_LA(
                    model,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=18,
                    stride_z=4,
                )
                dice_sample = var_all_case_LA(
                    ema_model,
                    args["root_path"],
                    num_classes=num_classes,
                    patch_size=args["patch_size"],
                    stride_xy=18,
                    stride_z=4,
                )

            if dice_sample_stu > best_dice_stu:
                best_dice_stu = dice_sample_stu
                tmp_stu_snapshot_path = os.path.join(snapshot_path, "student")
                if not os.path.exists(tmp_stu_snapshot_path):
                    os.makedirs(tmp_stu_snapshot_path, exist_ok=True)

                save_mode_path_stu = os.path.join(
                    tmp_stu_snapshot_path,
                    "ep_{:0>3}_dice_{}.pth".format(epoch_num, round(best_dice_stu, 4)),
                )
                torch.save(model.state_dict(), save_mode_path_stu)

                save_best_path_stu = os.path.join(
                    snapshot_path, "{}_best_stu_model.pth".format(args["model"])
                )
                torch.save(model.state_dict(), save_best_path_stu)

            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                tmp_tea_snapshot_path = os.path.join(snapshot_path, "teacher")
                if not os.path.exists(tmp_tea_snapshot_path):
                    os.makedirs(tmp_tea_snapshot_path, exist_ok=True)

                save_mode_path = os.path.join(
                    tmp_tea_snapshot_path,
                    "ep_{:0>3}_dice_{}.pth".format(epoch_num, best_dice),
                )
                torch.save(ema_model.state_dict(), save_mode_path)

                save_best_path = os.path.join(
                    snapshot_path, "{}_best_tea_model.pth".format(args["model"])
                )
                torch.save(ema_model.state_dict(), save_best_path)

            # writer
            writer.add_scalar("Var_dice/Dice_ema", dice_sample, epoch_num)
            writer.add_scalar("Var_dice/Best_dice_ema", best_dice, epoch_num)
            writer.add_scalar("Var_dice/Dice", dice_sample_stu, epoch_num)
            writer.add_scalar("Var_dice/Best_dice", best_dice_stu, epoch_num)

            # csv
            tmp_results_ts = {
                "loss_total": meter_train_losses.avg,
                "loss_sup": meter_sup_losses.avg,
                "loss_unsup": meter_uns_losses.avg,
                "learning_rate": meter_learning_rates.avg,
                "Dice_tea": dice_sample,
                "Dice_tea_best": best_dice,
                "Dice_stu": dice_sample_stu,
                "Dice_stu_best": best_dice_stu,
            }
            data_frame = pd.DataFrame(
                data=tmp_results_ts, index=range(epoch_num, epoch_num + 1)
            )
            if epoch_num > 0 and osp.exists(csv_test):
                data_frame.to_csv(csv_test, mode="a", header=None, index_label="epoch")
            else:
                data_frame.to_csv(csv_test, index_label="epoch")

            # logs
            logging.info(
                "iteration %d : dice_score: %f best_dice: %f"
                % (iter_num, dice_sample, best_dice)
            )
            logging.info(
                " <<Test>> - Ep:{}  - Dice-S/T:{:.2f}/{:.2f}, Best-S:{:.2f}, Best-T:{:.2f}".format(
                    epoch_num,
                    dice_sample_stu * 100,
                    dice_sample * 100,
                    best_dice_stu * 100,
                    best_dice * 100,
                )
            )
            logging.info(
                "          - AvgLoss(lb/ulb/all):{:.2f}/{:.2f}/{:.2f}".format(
                    meter_sup_losses.avg,
                    meter_uns_losses.avg,
                    meter_train_losses.avg,
                )
            )

            model.train()
            ema_model.train()

        if (epoch_num + 1) % args.get("save_interval_epoch", 1000000) == 0:
            save_mode_path = os.path.join(
                snapshot_path, "epoch_" + str(epoch_num) + ".pth"
            )
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if iter_num >= max_iterations:
            iterator.close()
            break
    save_mode_path = os.path.join(snapshot_path, "last.pth")
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
    return "Training Finished!"


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        III. main process
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
if __name__ == "__main__":
    # 1. set up config
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="", help="configuration file")

    # Basics: Data, results, model
    parser.add_argument(
        "--root_path", type=str, default="./data/LA", help="Name of Experiment"
    )
    parser.add_argument(
        "--res_path", type=str, default="./results/LA", help="Path to save resutls"
    )
    parser.add_argument("--exp", type=str, default="LA/POST", help="experiment_name")
    parser.add_argument("--model", type=str, default="vnet_hsseg", help="model_name")
    parser.add_argument(
        "--num_classes", type=int, default=2, help="output channel of network"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="the id of gpu used to train the model"
    )

    # Training Basics
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=15000,
        help="maximum epoch number to train",
    )
    parser.add_argument(
        "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
    )
    # https://blog.csdn.net/qq_43391414/article/details/122992458
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs="+",
        default=[112, 112, 80],
        help="patch size of network input",
    )

    parser.add_argument(
        "--max_samples", type=int, default=80, help="maximum samples to train"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether use deterministic training",
    )
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    parser.add_argument("--test_interval_iter", type=int, default=200, help="")
    parser.add_argument("--test_interval_ep", type=int, default=1, help="")
    parser.add_argument("--save_interval_epoch", type=int, default=1000000, help="")
    parser.add_argument(
        "-p",
        "--poly",
        default=False,
        action="store_true",
        help="whether poly scheduler",
    )

    # label and unlabel
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
    parser.add_argument(
        "--labeled_bs", type=int, default=2, help="labeled_batch_size per gpu"
    )
    parser.add_argument("--labeled_num", type=int, default=4, help="labeled data")

    # model related
    parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")

    # unlabeled loss
    parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
    parser.add_argument(
        "--consistency_rampup", type=float, default=40.0, help="consistency_rampup"
    )
    # ex
    parser.add_argument("--loss_type", type=str, default="rnkc")
    parser.add_argument(
        "--hist_weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--train_mode",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--stage_k",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
    )

    # parse args
    args = parser.parse_args()
    args = vars(args)

    # 2. update from the config files
    cfgs_file = args["cfg"]
    cfgs_file = os.path.join("./cfgs", cfgs_file)
    with open(cfgs_file, "r") as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    # convert "1e-x" to float
    for each in options_yaml.keys():
        tmp_var = options_yaml[each]
        if type(tmp_var) == str and "1e-" in tmp_var:
            options_yaml[each] = float(tmp_var)
    # update original parameters of argparse
    update_values(options_yaml, args)
    import pprint

    # 3. setup gpus and randomness
    # if args["gpu_id"] in range(8):
    if args["gpu_id"] in range(10):
        gid = args["gpu_id"]
    else:
        gid = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)

    if not args["deterministic"]:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args["seed"] > 0:
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])

    # 4. outputs and logger
    snapshot_path = "{}/{}_{}_labeled/{}".format(
        args["res_path"], args["exp"], args["labeled_num"], args["model"]
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{}".format(pprint.pformat(args)))

    train(args, snapshot_path)
