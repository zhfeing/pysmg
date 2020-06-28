import collections
import numpy as np
from PIL import Image

import torch
from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.data.transforms import default_transforms


class SUNRGBD(data.Dataset):
    """SUNRGBD loader

    Download From:
    http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz
        test source: http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz
        train source: http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-train_images.tgz

        first 5050 in this is test, later 5051 is train
        test and train labels source:
        https://github.com/ankurhanda/sunrgbd-meta-data/raw/master/sunrgbd_train_test_labels.tar.gz
    """

    def __init__(
        self,
        root,
        split="training",
        is_transform=False,
        img_size=(480, 640),
        augmentations=None,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
    ):
        self.root = root
        self.is_transform = is_transform
        self.n_classes = 38
        self.augmentations = augmentations
        self.split = split
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.normalize = (normalize_mean, normalize_std)
        # self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.anno_files = collections.defaultdict(list)
        self.cmap = self.color_map(normalized=False)

        for split in ["train", "test"]:
            file_list = sorted(recursive_glob(
                rootdir=self.root + "/{}/".format(split),
                suffix="jpg"
            ))
            self.files[split] = file_list

        for split in ["train", "test"]:
            file_list = sorted(
                recursive_glob(rootdir=self.root + "/annot_{}/".format(split), suffix="png")
            )
            self.anno_files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = self.anno_files[self.split][index].rstrip()
        # img_number = img_path.split('/')[-1]
        # lbl_path = os.path.join(self.root, 'annotations', img_number).replace('jpg', 'png')

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        if not (img.mode == "RGB" and lbl.mode == "L"):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img, lbl = default_transforms(img, lbl, self.normalize, self.img_size)

        classes = torch.unique(lbl)
        assert torch.all(classes == torch.unique(lbl))
        return img, lbl

    def color_map(self, N=256, normalized=False):
        """
        Return Color Map in PASCAL VOC format
        """

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255.0 if normalized else cmap
        return cmap

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.cmap[l, 0]
            g[temp == l] = self.cmap[l, 1]
            b[temp == l] = self.cmap[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
