import yaml
import argparse
import copy
import logging
import os
import signal
from typing import Dict, Any

import torch

import train
from ptsemseg.utils import str2bool, get_logger
from multiprocess_utils import multiprocess_runner


running_sub_processed: Dict[str, multiprocess_runner.Process] = dict()


def kill_handler(signum, frame):
    logger.warning("Got kill signal, exiting...")
    for name, p in running_sub_processed.items():
        logger.info("Killing subprocess: {}...".format(name))
        p.terminate()
        logger.info("Killed")
    logger.info("Killing main process...")
    exit()


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
    worker = min(bs, 8)
    seed = running_cfg["seed"]
    while True:
        train_cfg["training"]["batch_size"] = bs
        train_cfg["training"]["seed"] = seed
        train_cfg["training"]["n_workers"] = worker
        train_cfg["validation"]["batch_size"] = bs
        train_cfg["validation"]["n_workers"] = worker

        if "encoder_name" not in train_cfg["model"].keys():
            train_cfg["model"]["encoder_name"] = None
            train_cfg["model"]["encoder_weights"] = None

        # write config file
        formatter = (
            train_cfg["model"]["arch"],
            train_cfg["model"]["encoder_name"],
            train_cfg["model"]["encoder_weights"],
            train_cfg["data"]["dataset"]
        )
        cfg_filepath = cfg_filepath.format(*formatter)
        logger.info("Writing config file: %s", cfg_filepath)
        with open(cfg_filepath, "w") as file:
            yaml.dump(train_cfg, file, yaml.SafeDumper)

        # training
        result_queue = multiprocess_runner.Queue()
        train_runner = multiprocess_runner.Runner(
            target=train.main,
            name="train-model-{}-encoder-{}-weights-{}-dataset-{}".format(*formatter),
            kwargs=dict(
                cfg_filepath=cfg_filepath,
                logdir=args.log_dir,
                gpu_preserve=args.gpu_preserve,
                debug=args.debug
            ),
            result_queue=result_queue
        )
        running_sub_processed[train_runner.name] = train_runner
        train_runner.start()
        logger.info("Main process is waiting for subprocess join...")
        train_runner.join()
        running_sub_processed.pop(train_runner.name)

        result = result_queue.get()
        if isinstance(result, multiprocess_runner.RunResult):
            logger.info("Training Done, result: {}\n".format(result))
            break

        # runing config failed
        e = result.exception
        tb = result.traceback
        cfg_failed = False

        if isinstance(e, train.CUDAOutOfMemory):
            bs //= 2
            worker = min(bs, 8)
            logger.warning("Model generator is out of memory, trying to reduce batch size to {}".format(bs))
            if bs < 2 * torch.cuda.device_count():
                logger.error("Minimum bathsize reached, skipping this config")
                cfg_failed = True
        elif isinstance(e, train.CUDAMemoryNotEnoughForModel):
            logger.error("Gpu memory is too small, skipping this config")
            cfg_failed = True
        elif isinstance(e, train.TrainFailed):
            logger.error("While training, found `TrainFailed` error: %s", e)
            cfg_failed = True
        elif isinstance(e, train.CodeBugs):
            logger.error("Found code bugs, skipping this config")
            cfg_failed = True
        else:
            logger.fatal("Run time error: {},\ntraceback: {}, skiping this config".format(e, tb))
            cfg_failed = True

        if cfg_failed:
            failed_list_name = os.path.join(args.log_dir, "failed_list.txt")
            mode = "a" if os.path.isfile(failed_list_name) else "w"
            with open(failed_list_name, mode) as file:
                file.write("model-{}-encoder-{}-weights-{}-dataset-{}\n".format(*formatter))
            break


if __name__ == "__main__":
    # set spawn start method
    multiprocess_runner.set_spawn_start_method()

    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", type=str)
    parser.add_argument("--cfg-path", type=str)
    parser.add_argument("--cfg-formatter", type=str)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--gpu-preserve", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.cfg_path, exist_ok=True)
    cfg_filepath = os.path.join(args.cfg_path, args.cfg_formatter)

    logger = get_logger(
        name=None,
        level=logging.INFO,
        logger_fp=os.path.join(args.log_dir, "model_generator.log")
    )

    signal.signal(signal.SIGINT, kill_handler)
    signal.signal(signal.SIGHUP, kill_handler)
    signal.signal(signal.SIGTERM, kill_handler)

    generate_models(
        global_config=args.global_config,
        cfg_filepath=cfg_filepath
    )
