import config
from functools import wraps
from timeit import default_timer
from contextlib import contextmanager


@contextmanager
def elapsed_timer(logger, proc="operation", title=False):
    # type: (logger, str, bool) -> function
    """
    Counts time as context manager.

    :rtype: function
    :param logger: logger.
    :param proc: process name.
    :param title: log operation title in advance.
    :return: yields a function.
    """
    if title:
        logger.info("{}..".format(proc))
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start
    logger.info("..{} completed, time spent: {:.5f}".format(proc, end - start))


def timeit(logger, message, title=False):
    """
    Timing decorator.
    """

    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            with elapsed_timer(logger, message, title):
                response = fn(*args, **kwargs)
            return response

        return inner

    return outer


if __name__ == "__main__":
    with elapsed_timer(config.logger):
        a = [i * i for i in range(100000)]
