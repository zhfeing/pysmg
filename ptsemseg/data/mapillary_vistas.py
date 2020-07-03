import os
import json
import torch
import numpy as np

from torch.utils import data
from PIL import Image

from ptsemseg.utils import recursive_glob
from ptsemseg.data.transforms import default_transforms


class MapillaryVistas(data.Dataset):
    def __init__(
        self,
        root,
        split="training",
        img_size="same",
        is_transform=True,
        augmentations=None,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
    ):
        self.root = root
        if split == "train":
            split = "training"
        if split == "val":
            split = "validation"
        if split == "test":
            split = "testing"
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 65

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        # self.mean = np.array([80.5423, 91.3162, 81.4312])
        self.normalize = (normalize_mean, normalize_std)
        self.files = {}

        self.images_base = os.path.join(self.root, self.split, "images")
        self.annotations_base = os.path.join(self.root, self.split, "labels")

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")

        self.class_ids, self.class_names, self.class_colors = self.parse_config()

        self.ignore_id = 250

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def parse_config(self):
        with open(os.path.join(self.root, "config.json")) as config_file:
            config = json.load(config_file)

        labels = config["labels"]

        class_names = []
        class_ids = []
        class_colors = []
        print("There are {} labels in the config file".format(len(labels)))
        for label_id, label in enumerate(labels):
            class_names.append(label["readable"])
            class_ids.append(label_id)
            class_colors.append(label["color"])

        return class_names, class_ids, class_colors

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png")
        )

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img, lbl = default_transforms(img, lbl, self.normalize, self.img_size)
        lbl[lbl == 65] = self.ignore_id
        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.class_colors[l][0]
            g[temp == l] = self.class_colors[l][1]
            b[temp == l] = self.class_colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
