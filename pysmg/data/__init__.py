from pysmg.data.pascal_voc import PascalVOC
from pysmg.data.camvid import Camvid
from pysmg.data.ade20k import ADE20K
from pysmg.data.cityscapes import Cityscapes
from pysmg.data.nyuv2 import NYUv2
from pysmg.data.sunrgbd import SUNRGBD
from pysmg.data.mapillary_vistas import MapillaryVistas


def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        "pascal": PascalVOC,
        "camvid": Camvid,
        "ade20k": ADE20K,
        "cityscapes": Cityscapes,
        "nyuv2": NYUv2,
        "sunrgbd": SUNRGBD,
        "vistas": MapillaryVistas,
    }[name]
