import collections
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils import data

from ptsemseg.utils import recursive_glob
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
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        # self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.normalize = (normalize_mean, normalize_std)
        self.files = collections.defaultdict(list)

        for split in ["training", "validation"]:
            file_list = recursive_glob(
                rootdir=os.path.join(self.root, "images", self.split), suffix=".jpg"
            )
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path[:-4] + "_seg.png"

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        lbl = self.encode_segmap(np.array(lbl))
        lbl = Image.fromarray(lbl)
        img, lbl = default_transforms(img, lbl, self.normalize, self.img_size)
        return img, lbl

    def encode_segmap(self, mask):
        # Refer : http://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]))
        label_mask = (mask[:, :, 0] / 10.0) * 256 + mask[:, :, 1]
        return np.array(label_mask, dtype=np.uint8)

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
