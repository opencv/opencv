import numpy as np

from .test_config import CommonConfig
from ..utils import create_parser, plot_acc


class ModelTestPipeline:
    def __init__(
            self,
            network_model,
            model_processor,
            dnn_model_processor
    ):
        self._net_model = network_model
        self._model_processor = model_processor
        self._dnn_model_processor = dnn_model_processor

        self._test_config = None
        self._data_fetcher = None
        self._accuracy_evaluator = None

    def init_test_pipeline(self):
        parser = create_parser()
        cmd_args = parser.parse_args()

        model_dict = self._net_model.get_prepared_models()

        model_names = list(model_dict.keys())
        print(model_names)
        print(
            "The model {} was successfully obtained and converted to OpenCV {}".format(model_names[0], model_names[1])
        )

        if cmd_args.is_evaluate:
            original_model_name = model_names[0]
            dnn_model_name = model_names[1]

            self.run_test_pipeline(
                [
                    self._model_processor(model_dict[original_model_name], original_model_name),
                    self._dnn_model_processor(model_dict[dnn_model_name], dnn_model_name)
                ],
                original_model_name.replace(" ", "_")
            )

    def run_test_pipeline(
            self,
            models_list,
            formatted_exp_name,
            is_plot_acc=True
    ):
        log_path = self._configure_eval_log(formatted_exp_name)

        print(
            "===== Running evaluation of the model with the following params:\n"
            "\t1. val data location: {}\n"
            "\t2. transform to RGB: {}\n"
            "\t3. log file location: {}\n".format(
                self._test_config.img_root_dir,
                self._test_config.bgr_to_rgb,
                log_path
            )
        )

        self._configure_acc_eval(log_path)
        self._accuracy_evaluator.process(models_list, self._data_fetcher)
        metric_array = np.array(self._accuracy_evaluator.general_quality_metric)

        self.get_max_eval_vals(metric_array)

        if is_plot_acc:
            plot_acc(metric_array, formatted_exp_name)

    def _configure_acc_eval(self, log_path):
        pass

    @staticmethod
    def _configure_eval_log(formatted_exp_name):
        common_test_config = CommonConfig()
        return common_test_config.log.format(formatted_exp_name)

    @staticmethod
    def _get_max_eval_vals(val_array):
        pass
