import logging
import time

from multiprocess_utils.multiprocess_runner import Runner, Queue


def target():
    while True:
        print("hehe")
        time.sleep(4)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info("start main")

    result_queue = Queue()
    runner = Runner(
        target=target,
        name="hehe",
        kwargs=dict(),
        result_queue=result_queue
    )
    runner.start()
    try:
        runner.join()
    except Exception as e:
        print(e)
    result = result_queue.get()

    # if isinstance(result, RunResult):
    #     print(result)
    # elif isinstance(result, FatalResult):
    #     print("ffff", result.traceback, result.exception)

