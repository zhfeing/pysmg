import yaml
import argparse
import copy
import logging
import os
import traceback
from typing import Dict, Any

import torch

import train
from ptsemseg.utils import str2bool


def get_dataset_iter(datasets_cfg: Dict[str, Any]):
    for cfg in datasets_cfg.values():
        new_dict = dict(data=cfg)
        yield new_dict


def get_encoder_iter(encoders_cfg: Dict[str, Any]):
    for arch, cfg in encoders_cfg.items():
        if isinstance(cfg, str):
            cfg = [cfg]
        for weights in cfg:
            new_dict = dict(encoder_name=arch, encoder_weights=weights)
            yield new_dict


def get_model_iter(models_cfg: Dict[str, Any]):
    for cfg in models_cfg.values():
        new_dict = dict(model=cfg)
        yield new_dict


def get_config_iter(global_config: Dict[str, Any]):
    datasets = global_config["datasets"]
    encoders = global_config["encoders"]
    models = global_config["models"]
    for data_cfg in get_dataset_iter(datasets):
        for model_cfg in get_model_iter(models):
            if model_cfg["model"]["customizable_encoder"]:
                for encoder_cfg in get_encoder_iter(encoders):
                    config_dict = dict()
                    config_dict.update(copy.deepcopy(data_cfg))
                    config_dict.update(copy.deepcopy(model_cfg))
                    config_dict["model"].update(copy.deepcopy(encoder_cfg))
                    config_dict["model"].pop("customizable_encoder")
                    yield config_dict
            else:
                config_dict = dict()
                config_dict.update(copy.deepcopy(data_cfg))
                config_dict.update(copy.deepcopy(model_cfg))
                config_dict["model"].pop("customizable_encoder")
                yield config_dict


def generate_models(global_config: str, cfg_filepath: str):
    """
    Args:
        global_config: global config yaml filepath
    """
    with open(global_config) as fp:
        global_config = yaml.load(fp, Loader=yaml.SafeLoader)

    for cfg in get_config_iter(global_config):
        cfg["training"] = global_config["training"]
        cfg["validation"] = dict()
        train_with_cfg(cfg, global_config["running_args"], cfg_filepath)


def train_with_cfg(train_cfg: Dict[str, Any], running_cfg: Dict[str, Any], cfg_filepath: str):
    """
    Args:
        cfg_filepath: filepath to save generated yaml config, using model name, encoder name
        and dataset name to format, e.g.
            `cfg_filepath.format(cfg["data"])`
    """

    bs = running_cfg["batch_size"]
    worker = min(bs, 32)
    seed = running_cfg["seed"]
    flag = True
    while flag:
        try:
            train_cfg["training"]["batch_size"] = bs
            train_cfg["training"]["seed"] = seed
            train_cfg["training"]["n_workers"] = worker
            train_cfg["validation"]["batch_size"] = bs
            train_cfg["validation"]["n_workers"] = worker

            if "encoder_name" in train_cfg["model"].keys():
                formater = (
                    train_cfg["model"]["arch"],
                    train_cfg["model"]["encoder_name"],
                    train_cfg["data"]["dataset"]
                )
            else:
                formater = (train_cfg["model"]["arch"], None, train_cfg["data"]["dataset"])
            cfg_filepath = cfg_filepath.format(*formater)
            with open(cfg_filepath, "w") as file:
                yaml.dump(train_cfg, file, yaml.SafeDumper)
            train.main(
                cfg_filepath=cfg_filepath,
                logdir=args.log_dir,
                gpu_preserve=args.gpu_preserve,
                debug=args.debug
            )
            logger.info("Training Done\n\n")
            flag = False
        except train.CUDAOutOfMemory:
            bs //= 2
            logger.warning("Model generator is out of memory, trying to reduce batch size to {}".format(bs))
            if bs < 1:
                logger.error("Even when using batchsize=1 memory is still not enough, skiping this config")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", type=str)
    parser.add_argument("--cfg-filepath", type=str)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--gpu-preserve", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.log_dir, "train.log"), "w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    generate_models(
        global_config=args.global_config,
        cfg_filepath=args.cfg_filepath
    )
