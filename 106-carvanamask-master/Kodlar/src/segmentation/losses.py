from keras import backend as k
from keras.losses import binary_crossentropy


def binarized_dice_coef(y_true, y_pred):
    # type: (object, object) -> float
    """
    Returns dice coefficient calculated in keras backend.

    :rtype: float
    :param y_true: true output.
    :param y_pred: predicted output.
    :return: dice coefficient.
    """
    smooth = 1.
    y_true = k.cast(k.greater(y_true, 0.5), "float32")
    y_pred = k.cast(k.greater(y_pred, 0.5), "float32")
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = k.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)
    return score


def dice_coef(y_true, y_pred):
    # type: (object, object) -> float
    """
    Returns dice coefficient calculated in keras backend.

    :rtype: float
    :param y_true: true output.
    :param y_pred: predicted output.
    :return: dice coefficient.
    """
    smooth = 1.
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = k.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    # type: (object, object) -> float
    """
    Returns dice loss calculated in keras backend.

    :rtype: float
    :param y_true: true output.
    :param y_pred: predicted output.
    :return: dice loss.
    """
    loss = 1 - dice_coef(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    # type: (object, object) -> float
    """
    Returns binary cross-entropy dice loss calculated in keras backend.

    :rtype: float
    :param y_true: true output.
    :param y_pred: predicted output.
    :return: binary cross-entropy dice loss.
    """
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def weighted_dice_coef(y_true, y_pred, weight):
    # type: (object, object, float) -> float
    """
    Returns weighted dice coefficient calculated in keras backend.

    :rtype: float
    :param y_true: true output.
    :param y_pred: predicted output.
    :param weight: a weight.
    :return: weighted dice coefficient.
    """
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * k.sum(w * intersection) + smooth) / (k.sum(w * m1) + k.sum(w * m2) + smooth)
    return score


def weighted_dice_loss(y_true, y_pred):
    # type: (object, object) -> float
    """
    Returns weighted dice loss calculated in keras backend.

    :rtype: float
    :param y_true: true output.
    :param y_pred: predicted output.
    :return: weighted dice loss.
    """
    y_true = k.cast(y_true, "float32")
    y_pred = k.cast(y_pred, "float32")
    # if we want to get same size of output, kernel size must be odd number
    if k.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif k.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        kernel_size = 41
    averaged_mask = k.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding="same", pool_mode="avg")
    border = k.cast(k.greater(averaged_mask, 0.005), "float32") * k.cast(k.less(averaged_mask, 0.995), "float32")
    weight = k.ones_like(averaged_mask)
    w0 = k.sum(weight)
    weight += border * 2
    w1 = k.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coef(y_true, y_pred, weight)
    return loss


def weighted_bce_loss(y_true, y_pred, weight):
    # type: (object, object, float) -> float
    """
    Returns weighted binary cross-entropy loss calculated in keras backend.

    :rtype: float
    :param y_true: true output.
    :param y_pred: predicted output.
    :param weight: a weight.
    :return: weighted binary cross-entropy loss.
    """
    epsilon = 1e-7
    y_pred = k.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = k.log(y_pred / (1. - y_pred))
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (k.log(1. + k.exp(-k.abs(logit_y_pred))) + k.maximum(-logit_y_pred, 0.))
    return k.sum(loss) / k.sum(weight)


def weighted_bce_weighted_dice_loss(y_true, y_pred):
    # type: (object, object) -> float
    """
    Returns weighted binary cross-entropy weighted dice loss calculated in keras backend.

    :rtype: float
    :param y_true: true output.
    :param y_pred: predicted output.
    :return: weighted binary cross-entropy dice loss.
    """
    y_true = k.cast(y_true, "float32")
    y_pred = k.cast(y_pred, "float32")
    if k.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif k.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        kernel_size = 41
    averaged_mask = k.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding="same", pool_mode="avg")
    border = k.cast(k.greater(averaged_mask, 0.005), "float32") * k.cast(k.less(averaged_mask, 0.995), "float32")
    weight = k.ones_like(averaged_mask)
    w0 = k.sum(weight)
    weight += border * 2
    w1 = k.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + (1 - weighted_dice_coef(y_true, y_pred, weight))
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    # type: (object, object) -> float
    """
    Returns weighted binary cross-entropy dice loss calculated in keras backend.

    :rtype: float
    :param y_true: true output.
    :param y_pred: predicted output.
    :return: weighted binary cross-entropy dice loss.
    """
    if k.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif k.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        kernel_size = 41
    averaged_mask = k.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding="same", pool_mode="avg")
    border = k.cast(k.greater(averaged_mask, 0.005), "float32") * k.cast(k.less(averaged_mask, 0.995), "float32")
    weight = k.ones_like(averaged_mask)
    w0 = k.sum(weight)
    weight += border * 2
    w1 = k.sum(weight)
    weight *= (w0 / w1)
    loss = (weighted_bce_loss(y_true, y_pred, weight) + 4.*dice_loss(y_true, y_pred))/5.
    return loss
