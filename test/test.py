from multiprocess_utils.multiprocess_runner import (
    Runner,
    RunResult,
    FatalResult,
    Queue
)
import logging


def hehe():
    logger = logging.getLogger(__name__)
    logger.info("ffffffffff")


def runner():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info("sub process starting")
    hehe()


if __name__ == "__main__":
    q = Queue()

    multiprocess_runner = Runner(
        target=runner,
        name="test",
        kwargs=dict(),
        result_queue=q
    )

    root_logger = logging.getLogger("main")
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    root_logger.addHandler(console)

    root_logger.info("muhehe, main process")
    root_logger.info("waiting for subprocess quit")
    multiprocess_runner.start()
    multiprocess_runner.join()
    root_logger.info("exiting...")

