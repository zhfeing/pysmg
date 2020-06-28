import yaml
import argparse
import matplotlib.pyplot as plt

from ptsemseg.data import get_dataset
from ptsemseg.augmentations import get_composed_augmentations


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

    t_dataset = dataset(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
        **dataloader_args
    )

    v_dataset = dataset(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        **dataloader_args
    )

    print(len(t_dataset), len(v_dataset))
    print(t_dataset[0])

    mask = v_dataset[100][1].numpy()
    plt.imshow(mask)
    plt.savefig("test.jpg")

    # n_classes = t_loader.n_classes
    # trainloader = data.DataLoader(
    #     t_loader,
    #     batch_size=cfg["training"]["batch_size"],
    #     num_workers=cfg["training"]["n_workers"],
    #     shuffle=True,
    # )

    # valloader = data.DataLoader(
    #     v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    # )


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

    get_data(cfg)

