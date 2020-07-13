from torch.utils.data import DataLoader
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.data import get_dataset


__all__ = ["get_dataloader"]


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
