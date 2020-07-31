import argparse
import errno
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch

from .test.test_config import CommonConfig

SEED_VAL = 42
DNN_LIB = "DNN"
# common path for model savings
MODEL_PATH_ROOT = os.path.join(CommonConfig().output_data_root_dir, "{}/models")


def make_dir(dir_to_create):
    if not os.path.exists(dir_to_create):
        try:
            os.makedirs(dir_to_create)
        except OSError as error_obj:
            if error_obj.errno != errno.EEXIST:
                raise


def get_full_model_path(lib_name, model_full_name):
    model_path = MODEL_PATH_ROOT.format(lib_name)
    return {
        "path": model_path,
        "full_path": os.path.join(model_path, model_full_name)
    }


def plot_acc(accuracies_list, experiment_name):
    plt.figure(figsize=[8, 6])
    plt.plot(accuracies_list[:, 0], "r", linewidth=2.5, label="Original Model")
    plt.plot(accuracies_list[:, 1], "b", linewidth=2.5, label="Converted DNN Model")
    plt.xlabel("Iterations ", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.title(experiment_name, fontsize=15)
    plt.legend()
    full_path_to_fig = os.path.join(CommonConfig().output_data_root_dir, experiment_name + ".png")
    plt.savefig(full_path_to_fig, bbox_inches="tight")
    pass


def set_common_reproducibility():
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)


def set_pytorch_env():
    set_common_reproducibility()
    torch.manual_seed(SEED_VAL)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VAL)
        torch.backends.cudnn_benchmark_enabled = False
        torch.backends.cudnn.deterministic = True


def set_tf_env(is_use_gpu=True):
    set_common_reproducibility()
    tf.random.set_seed(SEED_VAL)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    if tf.config.list_physical_devices("GPU") and is_use_gpu:
        gpu_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_visible_devices(gpu_devices[0], "GPU")
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        os.environ["TF_USE_CUDNN"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def str_bool(input_val):
    if input_val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value was expected')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_evaluate",
        type=str_bool,
        help="Set True if you want to run evaluation of the models (ex.: TF vs OpenCV Net)",
        required=True
    )
    return parser
