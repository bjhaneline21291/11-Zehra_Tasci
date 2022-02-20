import logging
from functools import wraps


def setup_logger(name, filename, level):
    # type: (str, str, int) -> logger
    """
    Setups logger and logfile.

    :rtype: logger
    :param name: logger name.
    :param filename: log filename.
    :param level: log level.
    :return: logger.
    """
    logger = logging.getLogger(name)
    handler = logging.FileHandler(filename)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.basicConfig(level=level)
    return logger


def logit(logger, message):
    """
    Logging decorator.
    """

    def outer(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            response = fn(*args, **kwargs)
            logger.info("..{} completed.".format(message))
            return response

        return inner

    return outer
