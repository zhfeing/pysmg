import os
import torch
import numpy as np
from PIL import Image

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.data.transforms import default_transforms


class MITSceneParsingBenchmark(data.Dataset):
    def __init__(
        self,
        root,
        split="training",
        is_transform=False,
        img_size=512,
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
        self.n_classes = 151  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        # self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.normalize = (normalize_mean, normalize_std)
        self.files = {}

        self.images_base = os.path.join(self.root, "images", self.split)
        self.annotations_base = os.path.join(self.root, "annotations_instance", self.split)
        self.files[self.split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")
        if not self.files[self.split]:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.images_base))
        print("Found %d %s images" % (len(self.files[self.split]), self.split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path)[:-4] + ".png")

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img, lbl = default_transforms(img, lbl, self.normalize, self.img_size)
        return img, lbl
