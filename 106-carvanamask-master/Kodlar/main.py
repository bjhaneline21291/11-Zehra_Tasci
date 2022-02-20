import sys
import config
import argparse
import numpy as np
import segmentation.models as models
from modules.logering import setup_logger
from segmentation.carvanmask import Carvanmask
sys.setrecursionlimit(10000)
logger = setup_logger(__name__, config.log_name, config.log_level)


def train(model, debug):
    # type: (str, bool) -> None
    """
    Trains model.

    :rtype: None
    :param model: model name to load.
    :param debug: training mode.
    :return: None.
    """
    carvan = Carvanmask(config.train_folder, dims=[1344, 896] if not debug else [128, 128], batch_size=2)
    carvan.train_test_split(test_size=config.test_size)
    carvan.set_model(models.get_unet_1344_896_model if not debug else models.get_unet_128_model, renorm=False)
    if model != "":
        carvan.load_weights(model)
    params = {"epochs": 50, "verbose": 1}
    carvan.train_model(params, train_folder=config.train_folder, masks_folder=config.masks_folder)


def evaluate(model, size):
    # type: (str, int) -> None
    """
    Evaluates model on 10 random validation examples.

    :rtype: None
    :param model: model name to load.
    :param size: number images to evaluate.
    :return: None.
    """
    carvan = Carvanmask(config.train_folder, dims=[1344, 896], batch_size=2, name=model)
    carvan.train_test_split(test_size=config.test_size)
    carvan.set_model(models.get_unet_1344_896_model, renorm=False)
    carvan.load_weights(model)
    for i in range(size):
        k = np.random.choice(np.arange(len(carvan.images)), 1)[0]
        carvan.eval(carvan.images[k], train_folder=config.train_folder, masks_folder=config.masks_folder)


def submit(model, save):
    # type: (str, bool) -> None
    """
    Creates submission file and predictions.

    :rtype: None
    :param model: model name to load.
    :param save: save predicted masks.
    :return: None.
    """
    carvan = Carvanmask(config.train_folder, dims=[1344, 896], batch_size=2, name=model)
    carvan.train_test_split(test_size=config.test_size)
    carvan.set_model(models.get_unet_1344_896_model, renorm=False)
    carvan.load_weights(model)
    carvan.create_submission(input_folder=config.test_folder, output_folder=config.predictions_folder, save=save)


def main():
    """
    Module main function.
    """
    parser = argparse.ArgumentParser(description="Carvanamask: image segmentation kaggle competition.")
    subparsers = parser.add_subparsers(help="commands")
    train_parser = subparsers.add_parser("train", help="train model")
    train_parser.add_argument("--debug", default=False,
                              help="use [128, 128] for downscaling (default: False)", required=False)
    train_parser.add_argument("--model", default="", type=str,
                              help="model name to load (default: not load)", required=False)
    train_parser.set_defaults(which="train")
    eval_parser = subparsers.add_parser("eval", help="eval model on random examples")
    eval_parser.add_argument("--model", type=str, help="model name to load", required=True)
    eval_parser.add_argument("--size", type=int, default=5, help="number of images to evaluate (default: 5")
    eval_parser.set_defaults(which="eval")
    submit_parser = subparsers.add_parser("submit", help="create submission file")
    submit_parser.add_argument("--model", type=str, help="model name to load", required=True)
    submit_parser.add_argument("--save", type=bool, default=True,
                               help="save predicted masks (default: True)", required=False)
    submit_parser.set_defaults(which="submit")
    args = vars(parser.parse_args())
    functions = {"train": train, "eval": evaluate, "submit": submit}
    func = functions[args.pop("which", None)]
    func(**args)


if __name__ == "__main__":
    main()
