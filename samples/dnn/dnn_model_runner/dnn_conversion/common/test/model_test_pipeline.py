import os

import numpy as np

from .configs.test_config import CommonConfig
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

        self._parser = create_parser()

        self._test_module = None
        self._test_module_config = None
        self._test_module_param_list = None

        self.test_config = None
        self._data_fetcher = None

        self._default_input_blob_preproc = None
        self._accuracy_evaluator = None

    def init_test_pipeline(self):
        cmd_args = self._parser.parse_args()
        model_dict = self._net_model.get_prepared_models()

        model_names = list(model_dict.keys())
        print(
            "The model {} was successfully obtained and converted to OpenCV {}".format(model_names[0], model_names[1])
        )

        if cmd_args.test:
            if not self._test_module_config.model:
                self._test_module_config.model = self._net_model.model_path["full_path"]

            if cmd_args.default_img_preprocess:
                self._test_module_config.scale = self._default_input_blob_preproc["scale"]
                self._test_module_config.mean = self._default_input_blob_preproc["mean"]
                self._test_module_config.std = self._default_input_blob_preproc["std"]
                self._test_module_config.crop = self._default_input_blob_preproc["crop"]

                if "rsz_height" in self._default_input_blob_preproc and "rsz_width" in self._default_input_blob_preproc:
                    self._test_module_config.rsz_height = self._default_input_blob_preproc["rsz_height"]
                    self._test_module_config.rsz_width = self._default_input_blob_preproc["rsz_width"]

                self._test_module_param_list = [
                    '--model', self._test_module_config.model,
                    '--input', self._test_module_config.input_img,
                    '--width', self._test_module_config.frame_width,
                    '--height', self._test_module_config.frame_height,
                    '--scale', self._test_module_config.scale,
                    '--mean', *self._test_module_config.mean,
                    '--std', *self._test_module_config.std,
                    '--classes', self._test_module_config.classes,
                ]

                if self._default_input_blob_preproc["rgb"]:
                    self._test_module_param_list.append('--rgb')

                self._configure_test_module_params()

            self._test_module.main(
                self._test_module_param_list
            )

        if cmd_args.evaluate:
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
        log_path, logs_dir = self._configure_eval_log(formatted_exp_name)

        print(
            "===== Running evaluation of the model with the following params:\n"
            "\t* val data location: {}\n"
            "\t* log file location: {}\n".format(
                self.test_config.img_root_dir,
                log_path
            )
        )

        os.makedirs(logs_dir, exist_ok=True)

        self._configure_acc_eval(log_path)
        self._accuracy_evaluator.process(models_list, self._data_fetcher)

        if is_plot_acc:
            plot_acc(
                np.array(self._accuracy_evaluator.general_inference_time),
                formatted_exp_name
            )

        print("===== End of the evaluation pipeline =====")

    def _configure_acc_eval(self, log_path):
        pass

    def _configure_test_module_params(self):
        pass

    @staticmethod
    def _configure_eval_log(formatted_exp_name):
        common_test_config = CommonConfig()
        return common_test_config.log_file_path.format(formatted_exp_name), common_test_config.logs_dir
