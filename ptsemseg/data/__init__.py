from ptsemseg.data.pascal_voc import PascalVOC
from ptsemseg.data.camvid import Camvid
from ptsemseg.data.ade20k import ADE20K
from ptsemseg.data.mit_sceneparsing_benchmark import MITSceneParsingBenchmark
from ptsemseg.data.cityscapes import Cityscapes
from ptsemseg.data.nyuv2 import NYUv2
from ptsemseg.data.sunrgbd import SUNRGBD
from ptsemseg.data.mapillary_vistas import MapillaryVistas


def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        "pascal": PascalVOC,
        "camvid": Camvid,
        "ade20k": ADE20K,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmark,
        "cityscapes": Cityscapes,
        "nyuv2": NYUv2,
        "sunrgbd": SUNRGBD,
        "vistas": MapillaryVistas,
    }[name]
