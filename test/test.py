import multiprocessing
from multiprocessing import Process
import os
import time

import torch

from ptsemseg.utils import preserve_memory


def hehe():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    preserve_memory()
    print("sub process", os.getpid())
    a = torch.empty(100000, 10000).cuda()
    print("a done: {}".format(a))
    time.sleep(10)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    multiprocessing.set_start_method("spawn")
    print(os.getpid())
    bb = torch.empty(100000, 10000).cuda()
    p = Process(target=hehe, args=())
    print("starting")
    p.start()
    p.join()
