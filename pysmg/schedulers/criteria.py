import logging
from typing import List, Callable

import numpy as np


def identity(val: float, his: List[float]):
    return val


def average_5(val: float, his: List[float]):
    return np.mean(np.array(his[-5:]))


SupportedDenoiseFn = {
    "default": identity,
    "average_5": average_5
}


class EarlyStopping:
    """
    Arguments:
        min_delta: Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement.
        patience: Number of callbacks with no improvement after which training will be
            stopped.
        mode: One of `{"min", "max"}`. In `min` mode, training will stop when the
            quantity monitored has stopped decreasing; in `"max"` mode it will stop when
            the quantity monitored has stopped increasing.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the baseline.
    """
    def __init__(
        self,
        patience: int,
        min_delta: float,
        mode: str = "min",
        baseline: float = None,
        denoise_fn: Callable[[float, List[float]], float] = identity
    ):
        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater

        self.denoise_fn = denoise_fn
        self.logger = logging.getLogger("EarlyStopping")
        self.reset_times = -1
        self.reset()
        self.logger.info(
            "Early stopping with mode: %s, patience: %d, baseline: %s, denoise_fn: %s",
            mode, patience, baseline, denoise_fn
        )

    def reset(self):
        self.wait_its = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

        self.to_stop = False
        self.val_his = list()
        self.reset_times += 1
        self.logger.info("Reset early stopping, current reset times: %d", self.reset_times)

    def update(self, val: float):
        self.val_his.append(val)
        val = self.denoise_fn(val, self.val_his)
        if self.monitor_op(val - self.min_delta, self.best):
            self.best = val
            self.wait_its = 0
        else:
            self.wait_its += 1
            if self.wait_its >= self.patience:
                self.to_stop = True
        self.logger.info(
            "After update with value: %f, best value: %f, wait its: %d, stop next: %s",
            val, self.best, self.wait_its, self.to_stop
        )

