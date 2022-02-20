import config
from keras.models import Model
from keras.optimizers import RMSprop
from modules.logering import setup_logger
from batch_renorm import BatchRenormalization
from losses import weighted_bce_weighted_dice_loss, dice_coef, weighted_bce_dice_loss, binarized_dice_coef
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, concatenate
logger = setup_logger(__name__, config.log_name, config.log_level)


def get_unet_128_model(dims, renorm=False):
    # type: (list, bool) -> model
    """
    Returns a 128 U-nets model.

    :rtype: model
    :param dims: image dimensions.
    :param renorm: use batch renormalization.
    :return: model.
    """
    batch_proceed = BatchRenormalization if renorm else BatchNormalization
    # 128 layer
    inputs = Input(shape=dims + [3])
    # region 64 layer
    down1 = Conv2D(64, (3, 3), padding="same")(inputs)
    down1 = batch_proceed()(down1)
    down1 = Activation("relu")(down1)
    down1 = Conv2D(64, (3, 3), padding="same")(down1)
    down1 = batch_proceed()(down1)
    down1 = Activation("relu")(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # endregion
    # region 32 layer
    down2 = Conv2D(128, (3, 3), padding="same")(down1_pool)
    down2 = batch_proceed()(down2)
    down2 = Activation("relu")(down2)
    down2 = Conv2D(128, (3, 3), padding="same")(down2)
    down2 = batch_proceed()(down2)
    down2 = Activation("relu")(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # endregion
    # region 16 layer
    down3 = Conv2D(256, (3, 3), padding="same")(down2_pool)
    down3 = batch_proceed()(down3)
    down3 = Activation("relu")(down3)
    down3 = Conv2D(256, (3, 3), padding="same")(down3)
    down3 = batch_proceed()(down3)
    down3 = Activation("relu")(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # endregion
    # region 8 layer
    down4 = Conv2D(512, (3, 3), padding="same")(down3_pool)
    down4 = batch_proceed()(down4)
    down4 = Activation("relu")(down4)
    down4 = Conv2D(512, (3, 3), padding="same")(down4)
    down4 = batch_proceed()(down4)
    down4 = Activation("relu")(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # endregion
    # region center layer
    center = Conv2D(1024, (3, 3), padding="same")(down4_pool)
    center = batch_proceed()(center)
    center = Activation("relu")(center)
    center = Conv2D(1024, (3, 3), padding="same")(center)
    center = batch_proceed()(center)
    center = Activation("relu")(center)
    # endregion
    # region 8 layer
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding="same")(up4)
    up4 = batch_proceed()(up4)
    up4 = Activation("relu")(up4)
    up4 = Conv2D(512, (3, 3), padding="same")(up4)
    up4 = batch_proceed()(up4)
    up4 = Activation("relu")(up4)
    up4 = Conv2D(512, (3, 3), padding="same")(up4)
    up4 = batch_proceed()(up4)
    up4 = Activation("relu")(up4)
    # endregion
    # region 16 layer
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    up3 = Conv2D(256, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    up3 = Conv2D(256, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    # endregion
    # region 32 layer
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    up2 = Conv2D(128, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    up2 = Conv2D(128, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    # endregion
    # region 64 layer
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    up1 = Conv2D(64, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    up1 = Conv2D(64, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    # endregion
    # 128 layer
    classify = Conv2D(1, (1, 1), activation="sigmoid")(up1)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=weighted_bce_weighted_dice_loss, metrics=[dice_coef])
    return model


def get_unet_1024_model(dims, renorm):
    # type: (list, bool) -> model
    """
    Returns a 1024 U-nets model.

    :rtype: model
    :param dims: image dimensions.
    :param renorm: use batch renormalization.
    :return: model.
    """
    batch_proceed = BatchRenormalization if renorm else BatchNormalization
    # 1024 layer
    inputs = Input(shape=dims + [3])
    # region 512 layer
    down0b = Conv2D(8, (3, 3), padding="same")(inputs)
    down0b = batch_proceed()(down0b)
    down0b = Activation("relu")(down0b)
    down0b = Conv2D(8, (3, 3), padding="same")(down0b)
    down0b = batch_proceed()(down0b)
    down0b = Activation("relu")(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # endregion
    # region 256 layer
    down0a = Conv2D(16, (3, 3), padding="same")(down0b_pool)
    down0a = batch_proceed()(down0a)
    down0a = Activation("relu")(down0a)
    down0a = Conv2D(16, (3, 3), padding="same")(down0a)
    down0a = batch_proceed()(down0a)
    down0a = Activation("relu")(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # endregion
    # region 128 layer
    down0 = Conv2D(32, (3, 3), padding="same")(down0a_pool)
    down0 = batch_proceed()(down0)
    down0 = Activation("relu")(down0)
    down0 = Conv2D(32, (3, 3), padding="same")(down0)
    down0 = batch_proceed()(down0)
    down0 = Activation("relu")(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # endregion
    # region 64 layer
    down1 = Conv2D(64, (3, 3), padding="same")(down0_pool)
    down1 = batch_proceed()(down1)
    down1 = Activation("relu")(down1)
    down1 = Conv2D(64, (3, 3), padding="same")(down1)
    down1 = batch_proceed()(down1)
    down1 = Activation("relu")(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # endregion
    # region 32 layer
    down2 = Conv2D(128, (3, 3), padding="same")(down1_pool)
    down2 = batch_proceed()(down2)
    down2 = Activation("relu")(down2)
    down2 = Conv2D(128, (3, 3), padding="same")(down2)
    down2 = batch_proceed()(down2)
    down2 = Activation("relu")(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # endregion
    # region 16 layer
    down3 = Conv2D(256, (3, 3), padding="same")(down2_pool)
    down3 = batch_proceed()(down3)
    down3 = Activation("relu")(down3)
    down3 = Conv2D(256, (3, 3), padding="same")(down3)
    down3 = batch_proceed()(down3)
    down3 = Activation("relu")(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # endregion
    # region 8 layer
    down4 = Conv2D(512, (3, 3), padding="same")(down3_pool)
    down4 = batch_proceed()(down4)
    down4 = Activation("relu")(down4)
    down4 = Conv2D(512, (3, 3), padding="same")(down4)
    down4 = batch_proceed()(down4)
    down4 = Activation("relu")(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # endregion
    # region center layer
    center = Conv2D(1024, (3, 3), padding="same")(down4_pool)
    center = batch_proceed()(center)
    center = Activation("relu")(center)
    center = Conv2D(1024, (3, 3), padding="same")(center)
    center = batch_proceed()(center)
    center = Activation("relu")(center)
    # endregion
    # region 8 layer
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding="same")(up4)
    up4 = batch_proceed()(up4)
    up4 = Activation("relu")(up4)
    up4 = Conv2D(512, (3, 3), padding="same")(up4)
    up4 = batch_proceed()(up4)
    up4 = Activation("relu")(up4)
    up4 = Conv2D(512, (3, 3), padding="same")(up4)
    up4 = batch_proceed()(up4)
    up4 = Activation("relu")(up4)
    # endregion
    # region 16 layer
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    up3 = Conv2D(256, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    up3 = Conv2D(256, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    # endregion
    # region 32 layer
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    up2 = Conv2D(128, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    up2 = Conv2D(128, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    # endregion
    # region 64 layer
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    up1 = Conv2D(64, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    up1 = Conv2D(64, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    # endregion
    # region 128 layer
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding="same")(up0)
    up0 = batch_proceed()(up0)
    up0 = Activation("relu")(up0)
    up0 = Conv2D(32, (3, 3), padding="same")(up0)
    up0 = batch_proceed()(up0)
    up0 = Activation("relu")(up0)
    up0 = Conv2D(32, (3, 3), padding="same")(up0)
    up0 = batch_proceed()(up0)
    up0 = Activation("relu")(up0)
    # endregion
    # region 256 layer
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding="same")(up0a)
    up0a = batch_proceed()(up0a)
    up0a = Activation("relu")(up0a)
    up0a = Conv2D(16, (3, 3), padding="same")(up0a)
    up0a = batch_proceed()(up0a)
    up0a = Activation("relu")(up0a)
    up0a = Conv2D(16, (3, 3), padding="same")(up0a)
    up0a = batch_proceed()(up0a)
    up0a = Activation("relu")(up0a)
    # endregion
    # region 512 layer
    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3, 3), padding="same")(up0b)
    up0b = batch_proceed()(up0b)
    up0b = Activation("relu")(up0b)
    up0b = Conv2D(8, (3, 3), padding="same")(up0b)
    up0b = batch_proceed()(up0b)
    up0b = Activation("relu")(up0b)
    up0b = Conv2D(8, (3, 3), padding="same")(up0b)
    up0b = batch_proceed()(up0b)
    up0b = Activation("relu")(up0b)
    # endregion
    # 1024 layer
    classify = Conv2D(1, (1, 1), activation="sigmoid")(up0b)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=weighted_bce_weighted_dice_loss, metrics=[dice_coef])
    return model


def get_unet_1280_model(dims, renorm):
    # type: (list, bool) -> model
    """
    Returns a 1280 U-nets model.

    :rtype: model
    :param dims: image dimensions.
    :param renorm: use batch renormalization.
    :return: model.
    """
    batch_proceed = BatchRenormalization if renorm else BatchNormalization
    # 1280 layer
    inputs = Input(shape=dims + [3])
    # region 640 layer
    down0b = Conv2D(10, (3, 3), padding="same")(inputs)
    down0b = batch_proceed()(down0b)
    down0b = Activation("relu")(down0b)
    down0b = Conv2D(10, (3, 3), padding="same")(down0b)
    down0b = batch_proceed()(down0b)
    down0b = Activation("relu")(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # endregion
    # region 320 layer
    down0a = Conv2D(20, (3, 3), padding="same")(down0b_pool)
    down0a = batch_proceed()(down0a)
    down0a = Activation("relu")(down0a)
    down0a = Conv2D(20, (3, 3), padding="same")(down0a)
    down0a = batch_proceed()(down0a)
    down0a = Activation("relu")(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # endregion
    # region 160 layer
    down0 = Conv2D(40, (3, 3), padding="same")(down0a_pool)
    down0 = batch_proceed()(down0)
    down0 = Activation("relu")(down0)
    down0 = Conv2D(40, (3, 3), padding="same")(down0)
    down0 = batch_proceed()(down0)
    down0 = Activation("relu")(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # endregion
    # region 80 layer
    down1 = Conv2D(80, (3, 3), padding="same")(down0_pool)
    down1 = batch_proceed()(down1)
    down1 = Activation("relu")(down1)
    down1 = Conv2D(80, (3, 3), padding="same")(down1)
    down1 = batch_proceed()(down1)
    down1 = Activation("relu")(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # endregion
    # region 40 layer
    down2 = Conv2D(160, (3, 3), padding="same")(down1_pool)
    down2 = batch_proceed()(down2)
    down2 = Activation("relu")(down2)
    down2 = Conv2D(160, (3, 3), padding="same")(down2)
    down2 = batch_proceed()(down2)
    down2 = Activation("relu")(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # endregion
    # region 20 layer
    down3 = Conv2D(320, (3, 3), padding="same")(down2_pool)
    down3 = batch_proceed()(down3)
    down3 = Activation("relu")(down3)
    down3 = Conv2D(320, (3, 3), padding="same")(down3)
    down3 = batch_proceed()(down3)
    down3 = Activation("relu")(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # endregion
    # region 10 layer
    down4 = Conv2D(640, (3, 3), padding="same")(down3_pool)
    down4 = batch_proceed()(down4)
    down4 = Activation("relu")(down4)
    down4 = Conv2D(640, (3, 3), padding="same")(down4)
    down4 = batch_proceed()(down4)
    down4 = Activation("relu")(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # endregion
    # region center layer
    center = Conv2D(1280, (3, 3), padding="same")(down4_pool)
    center = batch_proceed()(center)
    center = Activation("relu")(center)
    center = Conv2D(1280, (3, 3), padding="same")(center)
    center = batch_proceed()(center)
    center = Activation("relu")(center)
    # endregion
    # region 10 layer
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(640, (3, 3), padding="same")(up4)
    up4 = batch_proceed()(up4)
    up4 = Activation("relu")(up4)
    up4 = Conv2D(640, (3, 3), padding="same")(up4)
    up4 = batch_proceed()(up4)
    up4 = Activation("relu")(up4)
    up4 = Conv2D(640, (3, 3), padding="same")(up4)
    up4 = batch_proceed()(up4)
    up4 = Activation("relu")(up4)
    # endregion
    # region 20 layer
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(320, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    up3 = Conv2D(320, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    up3 = Conv2D(320, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    # endregion
    # region 40 layer
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(160, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    up2 = Conv2D(160, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    up2 = Conv2D(160, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    # endregion
    # region 80 layer
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(80, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    up1 = Conv2D(80, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    up1 = Conv2D(80, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    # endregion
    # region 160 layer
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(40, (3, 3), padding="same")(up0)
    up0 = batch_proceed()(up0)
    up0 = Activation("relu")(up0)
    up0 = Conv2D(40, (3, 3), padding="same")(up0)
    up0 = batch_proceed()(up0)
    up0 = Activation("relu")(up0)
    up0 = Conv2D(40, (3, 3), padding="same")(up0)
    up0 = batch_proceed()(up0)
    up0 = Activation("relu")(up0)
    # endregion
    # region 320 layer
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(20, (3, 3), padding="same")(up0a)
    up0a = batch_proceed()(up0a)
    up0a = Activation("relu")(up0a)
    up0a = Conv2D(20, (3, 3), padding="same")(up0a)
    up0a = batch_proceed()(up0a)
    up0a = Activation("relu")(up0a)
    up0a = Conv2D(20, (3, 3), padding="same")(up0a)
    up0a = batch_proceed()(up0a)
    up0a = Activation("relu")(up0a)
    # endregion
    # region 640 layer
    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(10, (3, 3), padding="same")(up0b)
    up0b = batch_proceed()(up0b)
    up0b = Activation("relu")(up0b)
    up0b = Conv2D(10, (3, 3), padding="same")(up0b)
    up0b = batch_proceed()(up0b)
    up0b = Activation("relu")(up0b)
    up0b = Conv2D(10, (3, 3), padding="same")(up0b)
    up0b = batch_proceed()(up0b)
    up0b = Activation("relu")(up0b)
    # endregion
    # 1280 layer
    classify = Conv2D(1, (1, 1), activation="sigmoid")(up0b)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=weighted_bce_weighted_dice_loss, metrics=[dice_coef])
    return model


def get_unet_1344_896_model(dims, renorm):
    # type: (list, bool) -> model
    """
    Returns a 1344x896 U-nets model.

    :rtype: model
    :param dims: image dimensions.
    :param renorm: use batch renormalization.
    :return: model.
    """
    batch_proceed = BatchRenormalization if renorm else BatchNormalization
    # 1024 layer
    inputs = Input(shape=[dims[1], dims[0], 3])
    # region 512 layer
    down0b = Conv2D(16, (3, 3), padding="same")(inputs)
    down0b = batch_proceed()(down0b)
    down0b = Activation("relu")(down0b)
    down0b = Conv2D(16, (3, 3), padding="same")(down0b)
    down0b = batch_proceed()(down0b)
    down0b = Activation("relu")(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # endregion
    # region 256 layer
    down0a = Conv2D(32, (3, 3), padding="same")(down0b_pool)
    down0a = batch_proceed()(down0a)
    down0a = Activation("relu")(down0a)
    down0a = Conv2D(32, (3, 3), padding="same")(down0a)
    down0a = batch_proceed()(down0a)
    down0a = Activation("relu")(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # endregion
    # region 128 layer
    down0 = Conv2D(64, (3, 3), padding="same")(down0a_pool)
    down0 = batch_proceed()(down0)
    down0 = Activation("relu")(down0)
    down0 = Conv2D(64, (3, 3), padding="same")(down0)
    down0 = batch_proceed()(down0)
    down0 = Activation("relu")(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # endregion
    # region 64 layer
    down1 = Conv2D(128, (3, 3), padding="same")(down0_pool)
    down1 = batch_proceed()(down1)
    down1 = Activation("relu")(down1)
    down1 = Conv2D(128, (3, 3), padding="same")(down1)
    down1 = batch_proceed()(down1)
    down1 = Activation("relu")(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # endregion
    # region 32 layer
    down2 = Conv2D(256, (3, 3), padding="same")(down1_pool)
    down2 = batch_proceed()(down2)
    down2 = Activation("relu")(down2)
    down2 = Conv2D(256, (3, 3), padding="same")(down2)
    down2 = batch_proceed()(down2)
    down2 = Activation("relu")(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # endregion
    # region 16 layer
    down3 = Conv2D(512, (3, 3), padding="same")(down2_pool)
    down3 = batch_proceed()(down3)
    down3 = Activation("relu")(down3)
    down3 = Conv2D(512, (3, 3), padding="same")(down3)
    down3 = batch_proceed()(down3)
    down3 = Activation("relu")(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # endregion
    # region center layer
    center = Conv2D(1024, (3, 3), padding="same")(down3_pool)
    center = batch_proceed()(center)
    center = Activation("relu")(center)
    center = Conv2D(1024, (3, 3), padding="same")(center)
    center = batch_proceed()(center)
    center = Activation("relu")(center)
    # endregion
    # region 16 layer
    up3 = UpSampling2D((2, 2))(center)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(512, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    up3 = Conv2D(512, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    up3 = Conv2D(512, (3, 3), padding="same")(up3)
    up3 = batch_proceed()(up3)
    up3 = Activation("relu")(up3)
    # endregion
    # region 32 layer
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(256, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    up2 = Conv2D(256, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    up2 = Conv2D(256, (3, 3), padding="same")(up2)
    up2 = batch_proceed()(up2)
    up2 = Activation("relu")(up2)
    # endregion
    # region 64 layer
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(128, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    up1 = Conv2D(128, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    up1 = Conv2D(128, (3, 3), padding="same")(up1)
    up1 = batch_proceed()(up1)
    up1 = Activation("relu")(up1)
    # endregion
    # region 128 layer
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(64, (3, 3), padding="same")(up0)
    up0 = batch_proceed()(up0)
    up0 = Activation("relu")(up0)
    up0 = Conv2D(64, (3, 3), padding="same")(up0)
    up0 = batch_proceed()(up0)
    up0 = Activation("relu")(up0)
    up0 = Conv2D(64, (3, 3), padding="same")(up0)
    up0 = batch_proceed()(up0)
    up0 = Activation("relu")(up0)
    # endregion
    # region 256 layer
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(32, (3, 3), padding="same")(up0a)
    up0a = batch_proceed()(up0a)
    up0a = Activation("relu")(up0a)
    up0a = Conv2D(32, (3, 3), padding="same")(up0a)
    up0a = batch_proceed()(up0a)
    up0a = Activation("relu")(up0a)
    up0a = Conv2D(32, (3, 3), padding="same")(up0a)
    up0a = batch_proceed()(up0a)
    up0a = Activation("relu")(up0a)
    # endregion
    # region 512 layer
    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(16, (3, 3), padding="same")(up0b)
    up0b = batch_proceed()(up0b)
    up0b = Activation("relu")(up0b)
    up0b = Conv2D(16, (3, 3), padding="same")(up0b)
    up0b = batch_proceed()(up0b)
    up0b = Activation("relu")(up0b)
    up0b = Conv2D(16, (3, 3), padding="same")(up0b)
    up0b = batch_proceed()(up0b)
    up0b = Activation("relu")(up0b)
    # endregion
    # 1024 layer
    classify = Conv2D(1, (1, 1), activation="sigmoid")(up0b)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=weighted_bce_dice_loss, metrics=[dice_coef, binarized_dice_coef])
    return model
