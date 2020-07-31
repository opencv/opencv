from .model_test_pipeline import ModelTestPipeline
from .test_config import TestClsConfig

from ..evaluation.classification.cls_accuracy_evaluator import ClsAccEvaluation


class ClsModelTestPipeline(ModelTestPipeline):
    def __init__(
            self,
            network_model,
            model_processor,
            dnn_model_processor,
            data_fetcher
    ):
        super(ClsModelTestPipeline, self).__init__(
            network_model,
            model_processor,
            dnn_model_processor
        )

        self._test_config = TestClsConfig()

        self._data_fetcher = data_fetcher(
            imgs_dir=self._test_config.img_root_dir,
            frame_size=self._test_config.frame_size,
            bgr_to_rgb=self._test_config.bgr_to_rgb
        )

    def _configure_acc_eval(self, log_path):
        self._accuracy_evaluator = ClsAccEvaluation(
            log_path,
            self._test_config.img_cls_file,
            self._test_config.batch_size
        )

    @staticmethod
    def _get_max_eval_vals(val_array):
        print(
            "===== End of processing. Accuracy results:\n"
            "\t1. max accuracy (top-5) for the original model: {}\n"
            "\t2. max accuracy (top-5) for the DNN model: {}\n".format(
                max(val_array[:, 0]),
                max(val_array[:, 1]),
            )
        )
