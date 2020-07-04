import collections
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils import data

from ptsemseg.data.transforms import default_transforms


class ADE20K(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size="same",
        augmentations=None,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
    ):
        self.root = root
        if split == "train":
            split = "training"
        if split == "val":
            split = "validation"
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 150
        self.ignore_index = 250
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        # self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.normalize = (normalize_mean, normalize_std)
        self.files = collections.defaultdict(list)

        self.img_list = os.listdir(os.path.join(self.root, "images", self.split))
        self.len = len(self.img_list)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "images", self.split, self.img_list[index])
        seg_file = self.img_list[index][:-4] + ".png"
        lbl_path = os.path.join(self.root, "annotations", self.split, seg_file)

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        lbl -= 1
        lbl[lbl == -1] = self.ignore_index
        lbl[lbl == 249] = self.ignore_index
        return img, lbl

    def transform(self, img, lbl):
        img, lbl = default_transforms(img, lbl, self.normalize, self.img_size)
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        # TODO:(@meetshah1995)
        # Verify that the color mapping is 1-to-1
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = 10 * (l % 10)
            g[temp == l] = l
            b[temp == l] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
