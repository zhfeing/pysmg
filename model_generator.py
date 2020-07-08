import yaml
import argparse
import copy
import logging
import os
import traceback
from typing import Dict, Any

import torch

import train
from ptsemseg.utils import str2bool, preserve_memory, MemoryPreserveError


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
    cfg_failed = False
    while flag:
        train_cfg["training"]["batch_size"] = bs
        train_cfg["training"]["seed"] = seed
        train_cfg["training"]["n_workers"] = worker
        train_cfg["validation"]["batch_size"] = bs
        train_cfg["validation"]["n_workers"] = worker

        if "encoder_name" not in train_cfg["model"].keys():
            train_cfg["model"]["encoder_name"] = None
            train_cfg["model"]["encoder_weights"] = None

        # preserving memory
        if args.gpu_preserve:
            try:
                logger.info("Preserving memory...")
                preserve_memory(0.99)
                logger.info("Preserving memory done")
            except MemoryPreserveError:
                logger.fatal("Preserving memory failed: CUDA out of memory, memory will not be preserved")

        # write config file
        formater = (
            train_cfg["model"]["arch"],
            train_cfg["model"]["encoder_name"],
            train_cfg["model"]["encoder_weights"],
            train_cfg["data"]["dataset"]
        )
        cfg_filepath = cfg_filepath.format(*formater)
        logger.info("Writing config file: %s", cfg_filepath)
        with open(cfg_filepath, "w") as file:
            yaml.dump(train_cfg, file, yaml.SafeDumper)

        # training
        try:
            train.main(
                cfg_filepath=cfg_filepath,
                logdir=args.log_dir,
                gpu_preserve=False,
                debug=args.debug
            )
            logger.info("Training Done\n\n")
            flag = False
        except train.CUDAOutOfMemory:
            bs //= 2
            worker = min(bs, 32)
            logger.warning("Model generator is out of memory, trying to reduce batch size to {}".format(bs))
            if bs < 2 * torch.cuda.device_count():
                logger.error("Minimum bathsize reached, skipping this config")
                flag = False
                cfg_failed = True
        except train.CUDAMemoryNotEnoughForModel:
            logger.error("Gpu memory is too small, skipping this config")
            flag = False
            cfg_failed = True
        except train.CodeBugs:
            logger.error("Found code bugs, skipping this config")
            flag = False
            cfg_failed = True
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Run time error: {},\ntraceback: {}".format(e, tb))
            flag = False
            cfg_failed = True
        finally:
            torch.cuda.empty_cache()
            if cfg_failed:
                failed_list_name = os.path.join(args.log_dir, "failed_list.txt")
                mode = "a" if os.path.isfile(failed_list_name) else "w"
                with open(failed_list_name, mode) as file:
                    file.write(yaml.dump(train_cfg))
                    file.write("\n--------------------------------------------------\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", type=str)
    parser.add_argument("--cfg-path", type=str)
    parser.add_argument("--cfg-formater", type=str)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--gpu-preserve", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.cfg_path, exist_ok=True)
    cfg_filepath = os.path.join(args.cfg_path, args.cfg_formater)

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
        cfg_filepath=cfg_filepath
    )
