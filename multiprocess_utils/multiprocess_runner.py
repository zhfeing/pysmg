import logging
import os
import traceback
from typing import Callable, Dict, Any

import multiprocessing
from multiprocessing import Process, Queue


def set_spawn_start_method():
    multiprocessing.set_start_method("spawn")


class RunResult:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __repr__(self):
        return "Result: {}".format(self.kwargs)


class FatalResult:
    def __init__(self, exception: Exception, traceback: str):
        self.exception = exception
        self.traceback = traceback

    def __repr__(self):
        return "FatalResult: exception:\n{}\ntraceback:\n{}".format(self.exception, self.traceback)


class Runner(Process):
    def __init__(self, target: Callable, name: str, kwargs: Dict[str, Any], result_queue: Queue):
        super().__init__()
        self.target = target
        self.name = name
        self.kwargs = kwargs
        self.result_queue = result_queue
        self._logger = logging.getLogger(__name__)
        self._logger.info("Initial multiprocess runner done, main pid: %d", os.getpid())

    def run(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        logger.info(
            "Starting to run target: %s with kwargs: %s in pid: %d",
            self.target,
            self.kwargs,
            os.getpid()
        )
        try:
            ret = self.target(**self.kwargs)
            self.result_queue.put(RunResult(ret=ret))
        except Exception as e:
            tb = traceback.format_exc()
            self.result_queue.put(FatalResult(e, tb))
