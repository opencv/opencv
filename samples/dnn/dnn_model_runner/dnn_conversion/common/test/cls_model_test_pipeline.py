from .configs.test_config import TestClsConfig, TestClsModuleConfig
from .model_test_pipeline import ModelTestPipeline
from ..evaluation.classification.cls_accuracy_evaluator import ClsAccEvaluation
from ..utils import get_test_module


class ClsModelTestPipeline(ModelTestPipeline):
    def __init__(
            self,
            network_model,
            model_processor,
            dnn_model_processor,
            data_fetcher,
            img_processor=None,
            cls_args_parser=None,
            default_input_blob_preproc=None
    ):
        super(ClsModelTestPipeline, self).__init__(
            network_model,
            model_processor,
            dnn_model_processor
        )

        if cls_args_parser:
            self._parser = cls_args_parser

        self.test_config = TestClsConfig()

        parser_args = self._parser.parse_args()

        if parser_args.test:
            self._test_module_config = TestClsModuleConfig()
            self._test_module = get_test_module(
                self._test_module_config.test_module_name,
                self._test_module_config.test_module_path
            )

            if parser_args.default_img_preprocess:
                self._default_input_blob_preproc = default_input_blob_preproc
        if parser_args.evaluate:
            self._data_fetcher = data_fetcher(self.test_config, img_processor)

    def _configure_test_module_params(self):
        self._test_module_param_list.extend((
            '--crop', self._test_module_config.crop,
            '--std', *self._test_module_config.std
        ))

        if self._test_module_config.rsz_height and self._test_module_config.rsz_width:
            self._test_module_param_list.extend((
                '--initial_height', self._test_module_config.rsz_height,
                '--initial_width', self._test_module_config.rsz_width,
            ))

    def _configure_acc_eval(self, log_path):
        self._accuracy_evaluator = ClsAccEvaluation(
            log_path,
            self.test_config.img_cls_file,
            self.test_config.batch_size
        )
