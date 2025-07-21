import argparse
import logging
import os
import os.path as osp
import random
import sys
import yaml
import torch.nn.functional as F

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.mixaugs import *
from dataloaders.dataset_2d import (
    BaseDataSets,
    TwoStreamBatchSampler,
    WeakStrongAugment,
)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.util import update_values, time_str, AverageMeter
from val_2D import test_single_volume


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        I. helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "140": 1312,
        }
    elif "Prostate":
        ref_dict = {
            "2": 27,
            "4": 53,
            "8": 120,
            "12": 179,
            "16": 256,
            "21": 312,
            "42": 623,
        }
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


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


# Cross-Entropy Loss function
def ce_loss_tmp(pred, target, ignore=None):
    if ignore is not None:
        ignore = ignore.unsqueeze(1)
        target = target * (1 - ignore)  # Ignore the target in the mask region
    return F.cross_entropy(
        pred, target.squeeze(1).long()
    )  # Squeeze target if necessary


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        II. trainer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def train(args, snapshot_path):
    base_lr = args["base_lr"]
    num_classes = args["num_classes"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
    loss_type = args["loss_type"]
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
        model = net_factory(net_type=args["model"], in_chns=1, class_num=num_classes)
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
    model.cuda()
    ema_model.cuda()
    model.train()
    ema_model.train()

    # + + + + + + + + + + + #
    # 2. dataset
    # + + + + + + + + + + + #
    db_train = BaseDataSets(
        base_dir=args["root_path"],
        split="train",
        num=None,
        transform=transforms.Compose([WeakStrongAugment(args["patch_size"])]),
    )
    db_val = BaseDataSets(base_dir=args["root_path"], split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args["root_path"], args["labeled_num"])
    logging.info(
        "Total silices is: {}, labeled slices is: {}".format(
            total_slices, labeled_slice
        )
    )
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(
        unlabeled_idxs, labeled_idxs, batch_size, args["labeled_bs"]
    )

    # + + + + + + + + + + + #
    # 3. dataloader
    # + + + + + + + + + + + #
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
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
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    dice_loss_ignore = losses.DiceLossNew(num_classes)

    def compute_label_loss(segs, target_lb):
        target_lb_long = target_lb.long()
        target_lb_float = target_lb.unsqueeze(1).float()
        loss = 0.0
        for seg in segs:
            ce = ce_loss(seg, target_lb_long)
            dice = dice_loss(torch.softmax(seg, dim=1), target_lb_float)
            loss += (ce + dice) / 2.0
        return loss

    def compute_ulb_his_loss(segs_u, his_segs_u, threshold=0.95, loss_type="dice"):
        loss = 0.0
        for seg_u, his_u in zip(segs_u, his_segs_u):
            prob = torch.softmax(his_u, dim=1)
            pred = torch.argmax(prob, dim=1).unsqueeze(1).float()
            mask = (torch.max(prob.detach(), dim=1)[0] < threshold).float()
            # print(torch.softmax(seg_u, dim=1).shape, pred.shape, mask.shape)
            if loss_type == "dice":
                loss += dice_loss_ignore(torch.softmax(seg_u, dim=1), pred, ignore=mask)
            elif loss_type == "ce":
                loss += ce_loss_tmp(torch.softmax(seg_u, dim=1), pred, ignore=mask)
            else:
                loss += (
                    1
                    / 2
                    * (
                        ce_loss_tmp(torch.softmax(seg_u, dim=1), pred, ignore=mask)
                        + dice_loss_ignore(
                            torch.softmax(seg_u, dim=1), pred, ignore=mask
                        )
                    )
                )
        return loss

    def compute_ulb_stru_loss(segs_u, ref_seg_u, threshold=0.95):
        ref_prob = torch.softmax(ref_seg_u, dim=1).detach()
        ref_pred = torch.argmax(ref_prob, dim=1).unsqueeze(1).float()
        ref_mask = (torch.max(ref_prob, dim=1)[0] < threshold).float()

        loss = 0.0
        for seg_u in segs_u:
            if loss_type == "dice":
                loss += dice_loss_ignore(
                    torch.softmax(seg_u, dim=1), ref_pred, ignore=ref_mask
                )
            elif loss_type == "ce":
                loss += ce_loss_tmp(
                    torch.softmax(seg_u, dim=1), ref_pred, ignore=ref_mask
                )
            else:
                loss += (
                    1
                    / 2
                    * (
                        ce_loss_tmp(
                            torch.softmax(seg_u, dim=1), ref_pred, ignore=ref_mask
                        )
                        + dice_loss_ignore(
                            torch.softmax(seg_u, dim=1), ref_pred, ignore=ref_mask
                        )
                    )
                )
        return loss

    # + + + + + + + + + + + #
    # 5. training loop
    # + + + + + + + + + + + #
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_performance_stu = 0.0
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
            # get batched data

            img_lb_w, target_lb = (
                weak_batch[num_ulb:],
                label_batch[num_ulb:],
            )
            img_ulb_w, img_ulb_s = weak_batch[:num_ulb], strong_batch[:num_ulb]

            # for label
            seg0, seg1, seg2, seg3 = model(img_lb_w, mode="train", is_aug=True)
            # cutmix aug
            cm_flag = random.random() < 0.2
            if cm_flag:
                del img_ulb_s
                cm_mask = generate_mask(img_ulb_w)  # HW
                img_ulb_s = cut_mix_single(img_ulb_w, cm_mask)
            # for unlabel
            seg0_u, seg1_u, seg2_u, seg3_u = model(img_ulb_s, mode="train", is_aug=True)
            with torch.no_grad():
                seg0_his_u, seg1_his_u, seg2_his_u, seg3_his_u = ema_model(
                    img_ulb_w, mode="train"
                )

            if cm_flag:
                seg3_his_u = cut_mix_single(seg3_his_u, cm_mask)
                seg2_his_u = cut_mix_single(seg2_his_u, cm_mask)
                seg1_his_u = cut_mix_single(seg1_his_u, cm_mask)
                seg0_his_u = cut_mix_single(seg0_his_u, cm_mask)

            segs = [seg0, seg1, seg2, seg3]
            # segs = [seg0]
            loss_lb = compute_label_loss(segs, target_lb)

            segs_u = [seg0_u, seg1_u, seg2_u, seg3_u]
            his_segs_u = [seg0_his_u, seg1_his_u, seg2_his_u, seg3_his_u]
            # segs_u = [seg0_u]
            # his_segs_u = [seg0_his_u]
            loss_ulb_his = compute_ulb_his_loss(segs_u, his_segs_u)

            # print(seg0_u.shape)
            loss_ulb_stru = compute_ulb_stru_loss(
                [seg1_u, seg2_u, seg3_u], seg0_u.detach()
            )
            # loss_ulb_stru = compute_ulb_stru_loss([seg1_u, seg2_u], seg0_u.detach())
            # print(loss_lb.shape, loss_ulb_stru.shape, loss_ulb_his.shape)
            # 6) unsupervised loss
            if iter_num < 1000:
                loss_ulb = torch.tensor(0.0)
            else:
                weight_his = args["weight_his"]  # 1 0.8 0.6 0.4 0.2 0.0
                loss_ulb = weight_his * loss_ulb_his + (1 - weight_his) * loss_ulb_stru
                # loss_ulb = loss_ulb_his

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

            metric_list = 0.0
            ema_metric_list = 0.0

            for _, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume(
                    sampled_batch["image"],
                    sampled_batch["label"],
                    model,
                    classes=num_classes,
                )
                metric_list += np.array(metric_i)

                ema_metric_i = test_single_volume(
                    sampled_batch["image"],
                    sampled_batch["label"],
                    ema_model,
                    classes=num_classes,
                )
                ema_metric_list += np.array(ema_metric_i)

            metric_list = metric_list / len(db_val)
            ema_metric_list = ema_metric_list / len(db_val)

            for class_i in range(num_classes - 1):
                writer.add_scalar(
                    "info/val_{}_dice".format(class_i + 1),
                    metric_list[class_i, 0],
                    epoch_num,
                )
                writer.add_scalar(
                    "info/val_{}_hd95".format(class_i + 1),
                    metric_list[class_i, 1],
                    epoch_num,
                )

                writer.add_scalar(
                    "info/ema_val_{}_dice".format(class_i + 1),
                    ema_metric_list[class_i, 0],
                    epoch_num,
                )
                writer.add_scalar(
                    "info/ema_val_{}_hd95".format(class_i + 1),
                    ema_metric_list[class_i, 1],
                    epoch_num,
                )

            performance = np.mean(metric_list, axis=0)[0]
            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar("info/val_mean_dice", performance, epoch_num)
            writer.add_scalar("info/val_mean_hd95", mean_hd95, epoch_num)

            ema_performance = np.mean(ema_metric_list, axis=0)[0]
            ema_mean_hd95 = np.mean(ema_metric_list, axis=0)[1]
            writer.add_scalar("info/ema_val_mean_dice", ema_performance, epoch_num)
            writer.add_scalar("info/ema_val_mean_hd95", ema_mean_hd95, epoch_num)

            if performance > best_performance_stu:
                best_performance_stu = performance
                tmp_stu_snapshot_path = os.path.join(snapshot_path, "student")
                if not os.path.exists(tmp_stu_snapshot_path):
                    os.makedirs(tmp_stu_snapshot_path, exist_ok=True)
                save_mode_path_stu = os.path.join(
                    tmp_stu_snapshot_path,
                    "ep_{:0>3}_dice_{}.pth".format(
                        epoch_num, round(best_performance_stu, 4)
                    ),
                )
                torch.save(model.state_dict(), save_mode_path_stu)

                save_best_path_stu = os.path.join(
                    snapshot_path, "{}_best_stu_model.pth".format(args["model"])
                )
                torch.save(model.state_dict(), save_best_path_stu)

            if ema_performance > best_performance:
                best_performance = ema_performance
                tmp_tea_snapshot_path = os.path.join(snapshot_path, "teacher")
                if not os.path.exists(tmp_tea_snapshot_path):
                    os.makedirs(tmp_tea_snapshot_path, exist_ok=True)
                save_mode_path = os.path.join(
                    tmp_tea_snapshot_path,
                    "ep_{:0>3}_dice_{}.pth".format(
                        epoch_num, round(best_performance, 4)
                    ),
                )
                torch.save(ema_model.state_dict(), save_mode_path)

                save_best_path = os.path.join(
                    snapshot_path, "{}_best_tea_model.pth".format(args["model"])
                )
                torch.save(ema_model.state_dict(), save_best_path)

            # csv
            tmp_results_ts = {
                "loss_total": meter_train_losses.avg,
                "loss_sup": meter_sup_losses.avg,
                "loss_unsup": meter_uns_losses.avg,
                "learning_rate": meter_learning_rates.avg,
                "Dice_tea": ema_performance,
                "Dice_tea_best": best_performance,
                "Dice_stu": performance,
                "Dice_stu_best": best_performance_stu,
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
                " <<Test>> - Ep:{}  - mean_dice/mean_h95 - S:{:.2f}/{:.2f}, Best-S:{:.2f}, T:{:.2f}/{:.2f}, Best-T:{:.2f}".format(
                    epoch_num,
                    performance * 100,
                    mean_hd95,
                    best_performance_stu * 100,
                    ema_performance * 100,
                    ema_mean_hd95,
                    best_performance * 100,
                )
            )
            logging.info(
                "          - AvgLoss(lb/ulb/all):{:.4f}/{:.4f}/{:.4f}".format(
                    meter_sup_losses.avg, meter_uns_losses.avg, meter_train_losses.avg
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
        "--root_path", type=str, default="./data/ACDC", help="Name of Experiment"
    )
    parser.add_argument(
        "--res_path", type=str, default="./results/ACDC", help="Path to save resutls"
    )
    parser.add_argument(
        "--exp", type=str, default="ACDC/POST-NoT", help="experiment_name"
    )
    parser.add_argument("--loss_type", type=str, default="Dice")
    parser.add_argument("--model", type=str, default="unet_hsseg", help="model_name")
    parser.add_argument(
        "--num_classes", type=int, default=4, help="output channel of network"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="the id of gpu used to train the model"
    )

    # Training Basics
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=30000,
        help="maximum epoch number to train",
    )
    parser.add_argument(
        "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
    )
    parser.add_argument(
        "--weight_his",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--patch_size",
        type=list,
        default=[256, 256],
        help="patch size of network input",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether use deterministic training",
    )
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
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
    parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
    parser.add_argument(
        "--labeled_bs", type=int, default=12, help="labeled_batch_size per gpu"
    )
    parser.add_argument("--labeled_num", type=int, default=136, help="labeled data")

    # model related
    parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
    parser.add_argument(
        "--flag_pseudo_from_student",
        default=False,
        action="store_true",
        help="using pseudo from student itself",
    )

    # unlabeled loss
    parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
    parser.add_argument(
        "--consistency_rampup", type=float, default=150.0, help="consistency_rampup"
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
    # print confg information
    import pprint

    # print("{}".format(pprint.pformat(args)))
    # assert 1==0, "break here"

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
    # logging.info(str(args))
    logging.info("{}".format(pprint.pformat(args)))

    train(args, snapshot_path)
