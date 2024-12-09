import argparse
import importlib.util
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch

from .test.configs.test_config import CommonConfig

SEED_VAL = 42
DNN_LIB = "DNN"
# common path for model savings
MODEL_PATH_ROOT = os.path.join(CommonConfig().output_data_root_dir, "{}/models")


def get_full_model_path(lib_name, model_full_name):
    model_path = MODEL_PATH_ROOT.format(lib_name)
    return {
        "path": model_path,
        "full_path": os.path.join(model_path, model_full_name)
    }


def plot_acc(data_list, experiment_name):
    plt.figure(figsize=[8, 6])
    plt.plot(data_list[:, 0], "r", linewidth=2.5, label="Original Model")
    plt.plot(data_list[:, 1], "b", linewidth=2.5, label="Converted DNN Model")
    plt.xlabel("Iterations ", fontsize=15)
    plt.ylabel("Time (ms)", fontsize=15)
    plt.title(experiment_name, fontsize=15)
    plt.legend()
    full_path_to_fig = os.path.join(CommonConfig().output_data_root_dir, experiment_name + ".png")
    plt.savefig(full_path_to_fig, bbox_inches="tight")


def get_final_summary_info(general_quality_metric, general_inference_time, metric_name):
    general_quality_metric = np.array(general_quality_metric)
    general_inference_time = np.array(general_inference_time)
    summary_line = "===== End of processing. General results:\n"
    "\t* mean {} for the original model: {}\t"
    "\t* mean time (min) for the original model inferences: {}\n"
    "\t* mean {} for the DNN model: {}\t"
    "\t* mean time (min) for the DNN model inferences: {}\n".format(
        metric_name, np.mean(general_quality_metric[:, 0]),
        np.mean(general_inference_time[:, 0]) / 60000,
        metric_name, np.mean(general_quality_metric[:, 1]),
        np.mean(general_inference_time[:, 1]) / 60000,
    )
    return summary_line


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


def get_formatted_model_list(model_list):
    note_line = 'Please, choose the model from the below list:\n'
    spaces_to_set = ' ' * (len(note_line) - 2)
    return note_line + ''.join([spaces_to_set, '{} \n'] * len(model_list)).format(*model_list)


def model_str(model_list):
    def type_model_list(input_val):
        if input_val.lower() in model_list:
            return input_val.lower()
        else:
            raise argparse.ArgumentTypeError(
                'The model is currently unavailable for test.\n' +
                get_formatted_model_list(model_list)
            )

    return type_model_list


def get_test_module(test_module_name, test_module_path):
    module_spec = importlib.util.spec_from_file_location(test_module_name, test_module_path)
    test_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(test_module)
    module_spec.loader.exec_module(test_module)
    return test_module


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--test",
        type=str_bool,
        help="Define whether you'd like to run the model with OpenCV for testing.",
        default=False
    ),
    parser.add_argument(
        "--default_img_preprocess",
        type=str_bool,
        help="Define whether you'd like to preprocess the input image with defined"
             " PyTorch or TF functions for model test with OpenCV.",
        default=False
    ),
    parser.add_argument(
        "--evaluate",
        type=str_bool,
        help="Define whether you'd like to run evaluation of the models (ex.: TF vs OpenCV networks).",
        default=True
    )
    return parser


def create_extended_parser(model_list):
    parser = create_parser()
    parser.add_argument(
        "--model_name",
        type=model_str(model_list=model_list),
        help="\nDefine the model name to test.\n" +
             get_formatted_model_list(model_list),
        required=True
    )
    return parser
