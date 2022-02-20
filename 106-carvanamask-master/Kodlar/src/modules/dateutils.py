from datetime import datetime


def get_datestr():
    # type: () -> str
    """
    Returns current datetime as string.

    :rtype: str
    :return: datetime string.
    """
    return datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
