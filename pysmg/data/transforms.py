from typing import Tuple
from PIL import Image
import numpy as np

import torch
from torchvision import transforms


def default_transforms(img: Image, label: Image, normalize: Tuple, size: Tuple):
    if size == ("same", "same"):
        img_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize[0], normalize[1])
        ])
    else:
        img_tf = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(normalize[0], normalize[1])
        ])
        label = label.resize(size, Image.NEAREST)
    img = img_tf(img)
    label = torch.tensor(np.array(label, dtype=np.int64))
    return img, label
