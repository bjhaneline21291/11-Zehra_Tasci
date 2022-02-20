import cv2
import config
import numpy as np


def get_dice_coef(y_true, y_pred):
    # type: (ndarray, ndarray) -> float
    """
    Returns dice coefficient.

    :rtype: float
    :param y_true: true output.
    :param y_pred: predicted output.
    :return: dice coefficient.
    """
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def run_length_encode(mask):
    # type: (ndarray) -> str
    """
    Returns run length as formatted string.

    :rtype: str
    :param mask: mask ndarray.
    :return: run length as formatted string.
    """
    values = mask.flatten()
    runs = np.where(values[1:] != values[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = " ".join([str(r) for r in runs])
    return rle


def run_length_decode(rle, dims):
    # type: (str, list) -> ndarray
    """
    Returns mask ndarray from a given run length.

    :rtype: ndarray
    :param rle: run length encode.
    :param dims: the mask dimensions.
    :return: mask ndarray.
    """
    runs = np.array(rle.split(), np.int)
    runs[1::2] = runs[:-1:2] + runs[1::2]
    mask = np.zeros(dims[0]*dims[1])
    for i in range(0, len(runs), 2):
        try:
            mask[runs[i]:runs[i+1]] = 1
        except Exception as e:
            config.logger.error("..error occurred: {}".format(e))
    mask = mask.reshape((dims[0], dims[1]))
    return mask


def get_highlighted_mask(y_true, y_pred):
    # type: (ndarray, ndarray) -> ndarray
    """
    Returns highlighted predicted mask: (gray: intersection, red: missed pixels, green: false positive pixels).
    Pixels are supposed to be gray scaled and scaled between [0., 1.].

    :rtype: ndarray
    :param y_true: true output.
    :param y_pred: predicted output.
    :return: highlighted predicted mask.
    """
    y_t = cv2.cvtColor(y_true, cv2.COLOR_GRAY2RGB).astype(int)
    y_p = cv2.cvtColor(y_pred, cv2.COLOR_GRAY2RGB).astype(int)
    diff = (y_t - y_p)
    intersect = (y_t * y_p).astype(np.float32)
    diff[np.where((diff == [1, 1, 1]).all(axis=2))] = [1, 0, 0]
    diff[np.where((diff == [-1, -1, -1]).all(axis=2))] = [0, 1, 0]
    return (diff + intersect / 2).astype(np.float32)
