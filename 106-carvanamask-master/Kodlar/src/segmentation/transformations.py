import cv2
import numpy as np
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(image):
    # type: (ndarray) -> ndarray
    """
    Performs histogram equalization for a given image.

    :rtype: ndarray
    :param image: an image.
    :return: processed image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge((lab_planes[0], lab_planes[1], lab_planes[2]))
    image = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return image


def shift_scale_rotate(image, mask, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1),
                       rotate_limit=(-0, 0), aspect_limit=(0, 0), u=0.5):
    # type: (ndarray, ndarray, tuple, tuple, tuple, tuple, float) -> (ndarray, ndarray)
    """
    Performs random shift-scale-rotate for a given image and mask.

    :rtype: (ndarray, ndarray)
    :param image: an image.
    :param mask: a mask.
    :param shift_limit: shift limit.
    :param scale_limit: scale limit.
    :param rotate_limit: rotate limit.
    :param aspect_limit: aspect limit.
    :param u: random threshold.
    :return: processed image and mask.
    """
    if np.random.random() < u:
        border = cv2.BORDER_CONSTANT
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])

        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border,
                                    borderValue=(0, 0, 0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border,
                                   borderValue=(0, 0, 0,))
    return image, mask


def horizontal_flip(image, mask, u=0.5):
    # type: (ndarray, ndarray, tuple, tuple, tuple, tuple, float) -> (ndarray, ndarray)
    """
    Performs horizontal flip for a given image and mask.

    :rtype: (ndarray, ndarray)
    :param image: an image.
    :param mask: a mask.
    :param u: random threshold.
    :return: processed image and mask.
    """
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


def hue_saturation_value(image, hue_shift_limit=(-50, 50), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15), u=0.5):
    # type: (ndarray, tuple, tuple, tuple, float) -> (ndarray, ndarray)
    """
    Performs random hue-saturation-value augmentation for a given image.

    :rtype: (ndarray, ndarray)
    :param image: an image.
    :param hue_shift_limit: hue shift limit.
    :param sat_shift_limit: saturation shift limit.
    :param val_shift_limit: validation shift limit.
    :param u: random threshold.
    :return: processed image.
    """
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image
