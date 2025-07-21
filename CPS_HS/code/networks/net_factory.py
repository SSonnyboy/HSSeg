from networks.unet import UNet
from networks.vnet import VNet
from networks.unet_hsseg import UNet_HSSEG


def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_hsseg":
        net = UNet_HSSEG(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(
            n_channels=in_chns,
            n_classes=class_num,
            normalization="batchnorm",
            has_dropout=True,
        ).cuda()
    else:
        net = None
    return net
