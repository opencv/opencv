import numpy as np

from ..accuracy_eval import SemSegmEvaluation
from ..utils import plot_acc


def test_segm_models(models_list, data_fetcher, eval_params, experiment_name, is_print_eval_params=True,
                     is_plot_acc=True):
    if is_print_eval_params:
        print(
            "===== Running evaluation of the classification models with the following params:\n"
            "\t0. val data location: {}\n"
            "\t1. val data labels: {}\n"
            "\t2. frame size: {}\n"
            "\t3. batch size: {}\n"
            "\t4. transform to RGB: {}\n"
            "\t5. log file location: {}\n".format(
                eval_params.imgs_segm_dir,
                eval_params.img_cls_file,
                eval_params.frame_size,
                eval_params.batch_size,
                eval_params.bgr_to_rgb,
                eval_params.log
            )
        )

    accuracy_evaluator = SemSegmEvaluation(eval_params.log, eval_params.img_cls_file, eval_params.batch_size)
    accuracy_evaluator.process(models_list, data_fetcher)
    accuracy_array = np.array(accuracy_evaluator.general_fw_accuracy)

    print(
        "===== End of processing. Accuracy results:\n"
        "\t1. max accuracy (top-5) for the original model: {}\n"
        "\t2. max accuracy (top-5) for the DNN model: {}\n".format(
            max(accuracy_array[:, 0]),
            max(accuracy_array[:, 1]),
        )
    )

    if is_plot_acc:
        plot_acc(accuracy_array, experiment_name)
