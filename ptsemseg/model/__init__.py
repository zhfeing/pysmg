import copy
from typing import Dict

from torch.nn import Module

from segmentation_models_pytorch.deeplabv3 import DeepLabV3, DeepLabV3Plus
from segmentation_models_pytorch.fpn import FPN
from segmentation_models_pytorch.linknet import Linknet
from segmentation_models_pytorch.pan import PAN
from segmentation_models_pytorch.pspnet import PSPNet
from segmentation_models_pytorch.unet import Unet


def get_model(cfg, n_classes) -> Module:
    model_dict: Dict = copy.deepcopy(cfg["model"])
    name = model_dict["arch"]
    model_dict.pop("arch")
    model = _get_model_instance(name)
    return model(classes=n_classes, **model_dict)


def _get_model_instance(name):
    try:
        return {
            "deeplabv3": DeepLabV3,
            "deeplabv3+": DeepLabV3Plus,
            "fpn": FPN,
            "linknet": Linknet,
            "pan": PAN,
            "pspnet": PSPNet,
            "unet": Unet
        }[name]
    except:
        raise ("Model {} not available".format(name))
