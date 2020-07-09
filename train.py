import os
import yaml
import argparse
import shutil
import copy
import time
import tqdm
import logging
import traceback
from typing import Dict, Any, Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ptsemseg.utils import make_deterministic, str2bool, preserve_memory, get_logger
from ptsemseg.metrics import RunningScore
from ptsemseg.model import get_model
from ptsemseg.optimizers import get_optimizer
from ptsemseg.schedulers import get_scheduler
from ptsemseg.loss import get_loss_function
from ptsemseg.metrics import AverageMeter
from ptsemseg.dataloader import get_dataloader


class CUDAOutOfMemory(Exception):
    pass


class CodeBugs(Exception):
    pass


class CUDAMemoryNotEnoughForModel(Exception):
    pass


def rm_tf_logger(writer: SummaryWriter):
    log_dir = writer.log_dir
    writer.close()
    shutil.rmtree(log_dir)
    time.sleep(1.5)


def eval(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device
):
    # Setup Metrics
    n_classes = val_loader.dataset.n_classes
    val_loss_meter = AverageMeter()
    running_metrics_val = RunningScore(n_classes)
    model.eval()
    with torch.no_grad():
        for images_val, labels_val in tqdm.tqdm(val_loader):
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            try:
                outputs = model(images_val)
            except:
                del model
                raise
            val_loss = loss_fn(input=outputs, target=labels_val)

            pred = outputs.max(1)[1].cpu().numpy()
            gt = labels_val.cpu().numpy()

            running_metrics_val.update(gt, pred)
            val_loss_meter.update(val_loss.item())
    return running_metrics_val, val_loss_meter


def train(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ckpt_dir: str,
    logger: logging.Logger,
    writer: torch.utils.tensorboard.SummaryWriter,
    device: torch.device,
    logdir: str
):
    model = torch.nn.DataParallel(model)
    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = copy.deepcopy(cfg["training"]["optimizer"])
    optimizer_params.pop("name")

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(cfg["training"]["loss"]["name"]))

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"], map_location="cpu")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["iter"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["iter"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    time_meter = AverageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True

    # start training
    logger.info("start training")
    while i <= cfg["training"]["train_iters"] and flag:
        for (images, labels) in train_loader:
            i += 1
            start_ts = time.time()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            time_meter.update(time.time() - start_ts)

            if i % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f} Lr: {} Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    scheduler.get_lr(),
                    time_meter.avg / cfg["training"]["batch_size"]
                )
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i)
                time_meter.reset()

            if i % cfg["training"]["val_interval"] == 0 or i == cfg["training"]["train_iters"]:
                logger.info("start evaling")
                running_metrics_val, val_loss_meter = eval(model, val_loader, loss_fn, device)

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i)
                logger.info("Iter %d Loss: %.4f" % (i, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i)

                # save ckpt
                iou = score["Mean IoU : \t"]
                state = {
                    "iter": i,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "iou": iou,
                }
                save_path = os.path.join(
                    ckpt_dir,
                    "arch-{}-encoder-{}-weight-{}-dataset-{}-iter-{}.pkl".format(
                        cfg["model"]["arch"],
                        cfg["model"]["encoder_name"],
                        cfg["model"]["encoder_weights"],
                        cfg["data"]["dataset"],
                        i
                    ),
                )
                torch.save(state, save_path)

                if iou >= best_iou:
                    best_iou = iou
                    save_path = os.path.join(
                        logdir,
                        "ckpt",
                        "arch-{}-encoder-{}-weight-{}-dataset-{}-best.pkl".format(
                            cfg["model"]["arch"],
                            cfg["model"]["encoder_name"],
                            cfg["model"]["encoder_weights"],
                            cfg["data"]["dataset"]
                        ),
                    )
                    torch.save(state, save_path)

            if i == cfg["training"]["train_iters"]:
                flag = False
                break
    return best_iou


def main(cfg_filepath, logdir, gpu_preserve: bool = False, debug: bool = False):
    with open(cfg_filepath) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    if debug:
        cfg["training"]["n_workers"] = 0
        cfg["validation"]["n_workers"] = 0

    seed = cfg["training"]["seed"]

    ckpt_dir = os.path.join(logdir, "ckpt")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    formatter = (
        cfg["model"]["arch"],
        cfg["model"]["encoder_name"],
        cfg["model"]["encoder_weights"],
        cfg["data"]["dataset"]
    )
    writer = SummaryWriter(
        log_dir=os.path.join(
            logdir,
            "tf-board-logs",
            "arch-{}-encoder-{}-weight-{}-data-{}".format(*formatter)
        ),
        flush_secs=1
    )

    train_log_dir = os.path.join(logdir, "train-logs")
    os.makedirs(train_log_dir, exist_ok=True)
    logger = get_logger(
        level=logging.INFO,
        mode="a",
        logger_fp=os.path.join(
            train_log_dir,
            "training-arch-{}-encoder-{}-weight-{}-data-{}.log".format(*formatter)
        )
    )
    logger.info("Start running with config: \n{}".format(yaml.dump(cfg)))

    # set seed
    make_deterministic(seed)
    logger.info("Set seed : {}".format(seed))

    if gpu_preserve:
        logger.info("Preserving memory...")
        preserve_memory(0.99)
        logger.info("Preserving memory done")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading dataset...")
    train_loader, val_loader, n_classes = get_dataloader(cfg)
    logger.info("Load dataset {} done, total classes: {}, train num: {}, val num: {}".format(
        cfg["data"]["dataset"],
        n_classes,
        len(train_loader.dataset),
        len(val_loader.dataset)
    ))
    # move model to gpu
    try:
        model = get_model(cfg, n_classes).to(device)
    except RuntimeError as e:
        tb = traceback.format_exc()
        logger.error("Model too huge to move to gpu, exception: {}\n".format(e))
        raise CUDAMemoryNotEnoughForModel
    except Exception as e:
        tb = traceback.format_exc()
        logger.fatal("While getting model, exception:\n{}\ntraceback:\n{}\n".format(e, tb))
        logger.info("Deleting failed tensorboard logger...")
        rm_tf_logger(writer)
        logger.info("Deleting failed tensorboard logger done")
        raise

    # start training
    success = False
    try:
        best_iou = train(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            ckpt_dir=ckpt_dir,
            logger=logger,
            writer=writer,
            device=device,
            logdir=logdir
        )
        success = True
        return best_iou
    except Exception as e:
        tb = traceback.format_exc()
        if isinstance(e, RuntimeError) and e.args[0].find("CUDA out of memory") >= 0:
            logger.warning("Running out of memory, exception:\n{} \ntrackback:\n{}".format(e, tb))
            raise CUDAOutOfMemory
        else:
            logger.error("Found code bugs, exception:\n{} \ntrackback:\n{}".format(e, tb))
            raise CodeBugs
    finally:
        if not success:
            logger.info("Deleting failed tensorboard logger...")
            rm_tf_logger(writer)
            logger.info("Deleting failed tensorboard logger done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", nargs="?", type=str, default="configs/fcn8s_pascal.yml")
    parser.add_argument("--logdir", nargs="?", type=str, default="logs/test")
    parser.add_argument("--gpu_preserve", type=str2bool, default=False)
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()

    print("running in test mode!!!")
    time.sleep(3)

    main(args.config, args.logdir, args.gpu_preserve, args.debug)
