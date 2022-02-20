import os
import cv2
import config
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
import modules.timing as t
import matplotlib.pyplot as plt
from numpy import ceil
from scipy.ndimage import imread
from modules.timing import timeit
from multiprocessing import Queue
from modules.dateutils import get_datestr
from modules.logering import setup_logger
from sklearn.model_selection import train_test_split
from segmentation.utils import get_dice_coef, run_length_encode, get_highlighted_mask
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from segmentation.transformations import shift_scale_rotate, horizontal_flip, hue_saturation_value
logger = setup_logger(__name__, config.log_name, config.log_level)
graph = tf.get_default_graph()


class Carvanmask(object):
    def __init__(self, train_folder, dims=None, batch_size=2, name=None):
        # type: (str, list, int, str) -> None
        """
        Initialises an object.

        :rtype: None
        :param train_folder: folder with training images.
        :param dims: the dimensions in which we want to rescale our images.
        :param batch_size: batch size.
        :param name: instance name.
        :return: None.
        """
        # main
        dims = [1024, 1024] if dims is None else dims
        self.dims = dims
        name = get_datestr() if name is None else name
        self.name = name
        # images
        self.batch_size = batch_size
        self.images = os.listdir(train_folder)
        self.train_images = None
        self.validation_images = None
        # training
        self.model = None

    @timeit(logger, "train-test splitting", title=True)
    def train_test_split(self, test_size=0.15):
        # type: (float) -> None
        """
        Splits data into training and validation sets.

        :rtype: None
        :param test_size: validation share.
        :return: None.
        """
        self.train_images, self.validation_images = \
            train_test_split(self.images, test_size=test_size, random_state=config.seed)

    def get_train_data(self, images, train_folder, masks_folder):
        # type: (list, str, str) -> generator
        """
        Iterates over train images and masks in batches preprocessing them.

        :rtype: generator
        :param images: list of images names.
        :param train_folder: folder with training images.
        :param masks_folder: folder with masks images.
        :return: generator for images and masks in batches.
        """
        while True:
            for start in range(0, len(images), self.batch_size):
                x = list()
                y = list()
                end = min(start + self.batch_size, len(images))
                images_batch = images[start:end]
                for image_name in images_batch:
                    image = cv2.imread(os.path.join(train_folder, image_name))
                    image = cv2.resize(image, tuple(self.dims))
                    mask_name = "{}_mask.png".format(image_name.split(".")[0])
                    mask = imread(os.path.join(masks_folder, mask_name), flatten=True)
                    mask = cv2.resize(mask, tuple(self.dims))

                    image = hue_saturation_value(image)
                    image, mask = shift_scale_rotate(image, mask)
                    image, mask = horizontal_flip(image, mask)
                    mask = np.expand_dims(mask, axis=2)
                    x.append(image)
                    y.append(mask)
                x = np.array(x, np.float32) / 255
                y = np.array(y, np.float32) / 255
                yield x, y

    def get_validation_data(self, images, train_folder, masks_folder):
        # type: (list, str, str) -> generator
        """
        Iterates over validation images and masks in batches preprocessing them.

        :rtype: generator
        :param images: list of images names.
        :param train_folder: folder with training images.
        :param masks_folder: folder with masks images.
        :return: generator for images and masks in batches.
        """
        while True:
            for start in range(0, len(images), self.batch_size):
                x = list()
                y = list()
                end = min(start + self.batch_size, len(images))
                images_batch = images[start:end]
                for image_name in images_batch:
                    image = cv2.imread(os.path.join(train_folder, image_name))
                    image = cv2.resize(image, tuple(self.dims))
                    mask_name = "{}_mask.png".format(image_name.split(".")[0])
                    mask = imread(os.path.join(masks_folder, mask_name), flatten=True)
                    mask = cv2.resize(mask, tuple(self.dims))
                    mask = np.expand_dims(mask, axis=2)
                    x.append(image)
                    y.append(mask)
                x = np.array(x, np.float32) / 255
                y = np.array(y, np.float32) / 255
                yield x, y

    @timeit(logger, "setting neural network model", title=True)
    def set_model(self, func, renorm):
        # type: (func, bool) -> None
        """
        Returns compiled model for a given "get" function.

        :rtype: None
        :param func: get_model function.
        :param renorm: use batch renormalization.
        :return: None.
        """
        self.model = func(self.dims, renorm)

    @timeit(logger, "training neural network", title=True)
    def train_model(self, params, train_folder, masks_folder):
        # type: (dict, str, str) -> None
        """
        Trains compiled model for given params.

        :rtype: None
        :param params: model parameters.
        :param train_folder: folder with training images.
        :param masks_folder: folder with masks images.
        :return: None.
        """
        callbacks = [EarlyStopping(monitor="val_loss", patience=8, verbose=1, min_delta=1e-4),
                     ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, verbose=1, epsilon=1e-4),
                     ModelCheckpoint(monitor="val_loss", save_best_only=True, save_weights_only=True,
                                     filepath=os.path.join(config.models_folder, "{}.hdf5".format(self.name))),
                     TensorBoard(log_dir="logs")]
        steps_per_epoch = ceil(float(len(self.train_images)) / float(self.batch_size))
        validation_steps = ceil(float(len(self.validation_images)) / float(self.batch_size))
        self.model.fit_generator(generator=self.get_train_data(self.train_images, train_folder, masks_folder),
                                 steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                                 validation_data=self.get_validation_data(self.validation_images,
                                                                          train_folder, masks_folder),
                                 validation_steps=validation_steps, **params)

    @timeit(logger, "loading model weights", title=True)
    def load_weights(self, model_name):
        # type: (str) -> None
        """
        Loads model weights from a given file.

        :rtype: None
        :type model_name: model name.
        :return: None.
        """
        self.model.load_weights(os.path.join(config.models_folder, "{}.hdf5".format(model_name)))

    def eval(self, filename, train_folder, masks_folder):
        # type: (str, str, str) -> None
        """
        Predicts and saves output for a given image filename from a training set. Calculates a dice score.

        :rtype: None
        :param filename: image filename.
        :param train_folder: folder with images.
        :param masks_folder: folder with masks.
        :return: None.
        """
        with t.elapsed_timer(logger, "eval image {}".format(filename)):
            original = cv2.imread(os.path.join(train_folder, filename))
            filename = filename[:filename.rfind(".")]
            resized = cv2.resize(original, tuple(self.dims))
            flipped_resized = cv2.flip(resized, 1)
            x = np.array(resized, np.float32) / 255
            flipped_x = np.array(flipped_resized, np.float32) / 255
            original[:, :, 0], original[:, :, 2] = original[:, :, 2], original[:, :, 1]

            mask_name = "{}_mask.png".format(filename.split(".")[0])
            y = imread(os.path.join(masks_folder, mask_name), flatten=True)
            y = np.array(y, np.float32) / 255
            y_pred = self.model.predict(np.array((x,)))[0, :, :, 0]
            flipped_y_pred = self.model.predict(np.array((flipped_x,)))[0, :, :, 0]
            flipped_y_pred = cv2.flip(flipped_y_pred, 1)
            y_pred = (y_pred + flipped_y_pred) / 2.
            y_pred = np.array(cv2.resize(y_pred, tuple(config.orig_dims)) > config.threshold, np.float32)

            dice = get_dice_coef(y, y_pred)
            y_pred = get_highlighted_mask(y, y_pred)
            # region plot result
            fig, ax = plt.subplots(1, 3, figsize=(16, 4))
            ax = ax.ravel()
            ax[0].imshow(np.array(original, np.float32) / 255)
            ax[0].set_title("x")
            ax[1].imshow(y, cmap="gray")
            ax[1].set_title("y")
            ax[2].imshow(y_pred)
            ax[2].set_title("y_pred: {:.4f}".format(dice))
            fig_name = "{}_eval_{}.png".format(self.name, filename)
            fig.savefig(os.path.join(config.results_folder, fig_name), dpi=600)
            # endregion

    def predict(self, filename, folder):
        # type: (str, str) -> ndarray
        """
        Predicts mask for a given image filename.

        :rtype: image
        :param filename: image filename.
        :param folder: folder with images.
        :return: predicted mask.
        """
        with t.elapsed_timer(logger, "predict image {}".format(filename)):
            original = cv2.imread(os.path.join(folder, filename))
            resized = cv2.resize(original, tuple(self.dims))
            x = np.array(resized, np.float32) / 255
            y_pred = self.model.predict(np.array((x,)))[0, :, :, 0]
            y_pred = cv2.resize(y_pred, tuple(config.orig_dims)) > config.threshold
            return y_pred

    def create_submission(self, input_folder, output_folder, save=False):
        # type: (str, str, bool) -> None
        """
        Creates submission for Kaggle challenge.

        :rtype: None
        :param input_folder: folder with images.
        :param output_folder: folder with predictions.
        :param save: save predicted masks.
        :return: None.
        """

        def load_batch(q):
            # type: (queue) -> None
            """
            Puts loaded images batch into a queue.

            :rtype: None
            :param q: queue.
            :return: None.
            """
            for start in range(0, len(images), self.batch_size):
                end = min(start + self.batch_size, len(images))
                images_batch = images[start:end]
                x_batch = list()
                x_flipped_batch = list()
                names = list()
                for image_name in images_batch:
                    image = cv2.imread(os.path.join(input_folder, image_name))
                    image = cv2.resize(image, tuple(self.dims))
                    x_batch.append(image)
                    image_flipped = cv2.flip(image, 1)
                    x_flipped_batch.append(image_flipped)
                    names.append(image_name)
                x_batch = np.array(x_batch, np.float32) / 255
                x_flipped_batch = np.array(x_flipped_batch, np.float32) / 255
                q.put((x_batch, x_flipped_batch, end, names))

        def predict_batch(q):
            # type: (queue) -> None
            """
            Predicts masks batch and coverts into a rles.

            :rtype: None
            :param q: queue.
            :return: None.
            """
            for i in range(0, len(images), self.batch_size):
                x_batch, x_flipped_batch, end, names = q.get()
                with t.elapsed_timer(logger, "predicting images {}/{}".format(end, len(images))):
                    y_pred_batch = self.model.predict_on_batch(x_batch)
                    with graph.as_default():
                        y_pred_flipped_batch = self.model.predict_on_batch(x_flipped_batch)
                    y_pred_batch = np.squeeze(y_pred_batch, axis=3)
                    y_pred_flipped_batch = np.squeeze(y_pred_flipped_batch, axis=3)
                    index = 0
                    for y_pred, y_pred_flipped in zip(y_pred_batch, y_pred_flipped_batch):
                        y_pred_flipped = cv2.flip(y_pred_flipped, 1)
                        y_pred = (y_pred + y_pred_flipped) / 2.
                        prob = cv2.resize(y_pred, tuple(config.orig_dims))
                        if save:
                            filename = os.path.join(output_folder, names[index].replace("jpg", "png"))
                            cv2.imwrite(filename, prob * 255)
                        mask = prob > config.threshold
                        rle = run_length_encode(mask)
                        rles.append(rle)
                        index += 1

        images = os.listdir(input_folder)
        rles = []
        with t.elapsed_timer(logger, "predicting images rles", title=True):
            queue = Queue(maxsize=10)
            t1 = threading.Thread(target=load_batch, name="load_batch", args=(queue,))
            t2 = threading.Thread(target=predict_batch, name="predict_batch", args=(queue,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        with t.elapsed_timer(logger, "creating submission file", title=True):
            df = pd.DataFrame({"img": images, "rle_mask": rles})
            df.to_csv("submission.csv.gz", index=False, compression="gzip")
