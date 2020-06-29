import os
import yaml
import argparse
import shutil
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ptsemseg.data import get_dataset
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore
from ptsemseg.model import get_model

from segmentation_models_pytorch.unet import Unet


def get_data(cfg):
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

    valid_dataset = dataset(
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

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg["validation"]["batch_size"],
        num_workers=cfg["validation"]["n_workers"],
        shuffle=False,
        pin_memory=True
    )
    return train_loader, valid_loader, n_classes


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

    logdir = os.path.join(
        "logs",
        os.path.basename(args.config)[:-4],
        str(cfg["training"]["seed"])
    )
    writer = SummaryWriter(log_dir=logdir, flush_secs=1)

    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("start training")
    logger.info("RUNDIR: {}".format(logdir))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, valid_loader, n_classes = get_data(cfg)

    # # Setup Metrics
    # running_metrics_val = runningScore(n_classes)

    model = get_model(cfg, n_classes).to(device)

    for img, lbl in tqdm.tqdm(train_loader):
        img = img.to(device)
        lbl = lbl.to(device)
        pred = model(img)
        assert lbl.shape[1:3] == pred.shape[2:4], "shape: {} neq shape: {}".format(lbl.shape, pred.shape)

