import os
import logging
import platform
import multiprocessing
from modules.logering import setup_logger

# operating system
system = platform.system()

# cores number
cores = multiprocessing.cpu_count()

# root and data paths
root_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(root_path, "data")

# folders and files
models_folder = os.path.join(root_path, "src/segmentation/models")
results_folder = os.path.join(root_path, "src/segmentation/results")
train_folder = os.path.join(data_path, "train_hq")
masks_folder = os.path.join(data_path, "train_masks")
test_folder = os.path.join(data_path, "test")
predictions_folder = os.path.join(data_path, "predictions")

# logging
log_level = logging.DEBUG
log_name = os.path.join(root_path, "carvanmask.log")
logger = setup_logger("logger", log_name, log_level)

# seed
seed = 42

# training
test_size = 0.15
threshold = 0.5
orig_dims = [1918, 1280]
