import os
import errno
import random

import numpy as np

import torch
import tensorflow as tf


SEED_VAL = 42
DNN_LIB = "DNN"
# common path for model savings
MODEL_PATH_ROOT = os.path.join(DNN_LIB.lower(), "{}/models")


def make_dir(dir_to_create):
    if not os.path.exists(dir_to_create):
        # create defined directory
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
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    if tf.config.list_physical_devices('GPU') and is_use_gpu:
        gpu_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpu_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        os.environ['TF_USE_CUDNN'] = '1'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
