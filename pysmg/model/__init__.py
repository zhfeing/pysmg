import copy
from typing import Dict

from torch.nn import Module

from segmentation_models_pytorch.deeplabv3 import DeepLabV3, DeepLabV3Plus
from segmentation_models_pytorch.fpn import FPN
from segmentation_models_pytorch.linknet import Linknet
from segmentation_models_pytorch.pan import PAN
from segmentation_models_pytorch.pspnet import PSPNet
from segmentation_models_pytorch.unet import Unet
from pysmg.model.fcn import FCN8s, FCN16s, FCN32s
from pysmg.model.frrn import FRRN
from pysmg.model.icnet import ICNet
from pysmg.model.segnet import SegNet


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
            "unet": Unet,
            "fcn8s": FCN8s,
            "fcn16s": FCN16s,
            "fcn32s": FCN32s,
            "frrn_a": FRRN,
            "frrn_b": FRRN,
            "icnet": ICNet,
            "segnet": SegNet
        }[name]
    except:
        raise Exception("Model {} not available".format(name))
