import os
import yaml
import argparse
import shutil
import copy
import time
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ptsemseg.data import get_dataset
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.utils import get_logger, make_deterministic
from ptsemseg.metrics import RunningScore
from ptsemseg.model import get_model
from ptsemseg.optimizers import get_optimizer
from ptsemseg.schedulers import get_scheduler
from ptsemseg.loss import get_loss_function
from ptsemseg.metrics import AverageMeter


def get_dataloader(cfg):
    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    dataset = get_dataset(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    dataloader_args = cfg["data"].copy()
    dataloader_args.pop('dataset')
    dataloader_args.pop('train_split')
    dataloader_args.pop('val_split')
    dataloader_args.pop('img_rows')
    dataloader_args.pop('img_cols')
    dataloader_args.pop('path')

    train_dataset = dataset(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
        **dataloader_args
    )

    val_dataset = dataset(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        **dataloader_args
    )

    n_classes = train_dataset.n_classes
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["validation"]["batch_size"],
        num_workers=cfg["validation"]["n_workers"],
        shuffle=False,
        pin_memory=True
    )
    return train_loader, val_loader, n_classes


def train(cfg, model: torch.nn.Module, train_loader, val_loader):
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.cuda()
    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = copy.deepcopy(cfg["training"]["optimizer"])
    optimizer_params.pop("name")

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

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

    val_loss_meter = AverageMeter()
    time_meter = AverageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
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

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f} Lr: {} Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    scheduler.get_lr(),
                    time_meter.avg / cfg["training"]["batch_size"]
                )
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]:
                logger.info("start evaling")
                model.eval()
                with torch.no_grad():
                    for images_val, labels_val in tqdm.tqdm(val_loader):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        pred = outputs.max(1)[1].cpu().numpy()
                        gt = labels_val.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "iter": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    seed = cfg["training"]["seed"]
    logdir = os.path.join(
        "logs",
        os.path.basename(args.config)[:-4],
        str(seed)
    )
    writer = SummaryWriter(log_dir=logdir, flush_secs=1)

    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("start training")
    logger.info("RUNDIR: {}".format(logdir))

    make_deterministic(seed)
    logger.info("set seed : {}".format(seed))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, val_loader, n_classes = get_dataloader(cfg)

    # Setup Metrics
    running_metrics_val = RunningScore(n_classes)

    model = get_model(cfg, n_classes).to(device)

    train(cfg, model, train_loader, val_loader)
