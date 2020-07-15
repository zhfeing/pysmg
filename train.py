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

from pysmg.utils import make_deterministic, str2bool, preserve_memory, get_logger
from pysmg.metrics import RunningScore
from pysmg.model import get_model
from pysmg.optimizers import get_optimizer
from pysmg.schedulers import get_scheduler
from pysmg.loss import get_loss_function
from pysmg.metrics import AverageMeter
from pysmg.dataloader import get_dataloader
from pysmg.schedulers.criteria import EarlyStopping, SupportedDenoiseFn


class CUDAOutOfMemory(Exception):
    pass


class CodeBugs(Exception):
    pass


class CUDAMemoryNotEnoughForModel(Exception):
    pass


class TrainFailed(Exception):
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
    logdir: str,
    file_name_cfg: str
):
    model = torch.nn.DataParallel(model)
    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = copy.deepcopy(cfg["training"]["optimizer"])
    optimizer_params.pop("name")

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    use_early_stopping = False
    if "early_stopping" in cfg["training"]:
        use_early_stopping = True
        early_stopper = EarlyStopping(
            patience=cfg["training"]["early_stopping"]["patience"],
            mode=cfg["training"]["early_stopping"]["mode"],
            min_delta=cfg["training"]["early_stopping"]["min_delta"],
            baseline=cfg["training"]["early_stopping"]["baseline"],
            denoise_fn=SupportedDenoiseFn[cfg["training"]["early_stopping"]["denoise_fn"]]
        )

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
    it = start_iter
    flag = True

    # start training
    logger.info("start training")
    while it <= cfg["training"]["train_iters"] and flag:
        for (images, labels) in train_loader:
            it += 1
            start_ts = time.time()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(input=outputs, target=labels)
            # check loss legal or not
            if torch.isnan(loss):
                raise TrainFailed("Loss is NaN")

            loss.backward()
            optimizer.step()
            scheduler.step()

            time_meter.update(time.time() - start_ts)

            if (it - 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f} Lr: {} Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    it,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    scheduler.get_last_lr(),
                    time_meter.avg / cfg["training"]["batch_size"]
                )
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), it)
                time_meter.reset()

            if (it - 1) % cfg["training"]["val_interval"] == 0 or it == cfg["training"]["train_iters"]:
                logger.info("Start evaling...")
                running_metrics_val, val_loss_meter = eval(model, val_loader, loss_fn, device)

                ## write loggers
                writer.add_scalar("loss/val_loss", val_loss_meter.avg, it)
                logger.info("Iter %d Loss: %.4f" % (it, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, it)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, it)

                ## save checkpoints
                iou = score["Mean IoU : \t"]
                state = {
                    "iter": it,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "iou": iou,
                }
                save_path = os.path.join(
                    ckpt_dir,
                    file_name_cfg.format(
                        cfg["model"]["arch"],
                        cfg["model"]["encoder_name"],
                        cfg["model"]["encoder_weights"],
                        cfg["data"]["dataset"]
                    )
                )
                os.makedirs(save_path, exist_ok=True)
                torch.save(state, os.path.join(save_path, "iter-{}.pth".format(it)))

                ## save best checkpoint
                if iou >= best_iou:
                    best_iou = iou
                    torch.save(state, os.path.join(save_path, "best.pth"))

                # early stopping learning rate judgement
                if use_early_stopping:
                    early_stopper.update(val_loss_meter.avg)
                    if early_stopper.to_stop:
                        scheduler.base_lrs = list(map(lambda x: x / 10, scheduler.base_lrs))
                        logger.info("Decreased lr by 10")
                        early_stopper.reset()
                        # stop training if reset times reaches maximum
                        if early_stopper.reset_times >= cfg["training"]["early_stopping"]["max_reset_times"]:
                            logger.warning("Maximum reset time reached, stop training")
                            flag = False
                            break

            if it == cfg["training"]["train_iters"]:
                flag = False
                break
    return best_iou


def main(
    cfg_filepath: str,
    file_name_cfg: str,
    logdir: str,
    gpu_preserve: bool = False,
    debug: bool = False
):
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
            file_name_cfg.format(*formatter)
        ),
        flush_secs=1
    )

    train_log_dir = os.path.join(logdir, "train-logs")
    os.makedirs(train_log_dir, exist_ok=True)
    logger = get_logger(
        level=logging.INFO,
        mode="a",
        name=None,
        logger_fp=os.path.join(
            train_log_dir,
            "training-" + file_name_cfg.format(*formatter) + ".log"
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
            logdir=logdir,
            file_name_cfg=file_name_cfg
        )
        success = True
        return dict(best_iou=best_iou)
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
