from torchvision import models

from ..pytorch_model import (
    PyTorchModelPreparer,
    PyTorchModelProcessor,
    PyTorchDnnModelProcessor
)
from ...common.utils import set_pytorch_env, create_parser


class PyTorchFcnResNet50(PyTorchModelPreparer):
    def __init__(self, model_name, original_model):
        super(PyTorchFcnResNet50, self).__init__(model_name, original_model)


def main():
    parser = create_parser()
    cmd_args = parser.parse_args()
    set_pytorch_env()

    # Test the base process of model retrieval
    resnets = PyTorchFcnResNet50(
        model_name="resnet50",
        original_model=models.segmentation.fcn_resnet50(pretrained=True)
    )
    model_dict = resnets.get_prepared_models()

    if cmd_args.is_evaluate:
        from ...common.test_config import TestConfig
        from ...common.accuracy_eval import PASCALDataFetch
        from ...common.test.voc_segm_test import test_segm_models

        eval_params = TestConfig()

        model_names = list(model_dict.keys())
        original_model_name = model_names[0]
        dnn_model_name = model_names[1]

        #img_dir, segm_dir, names_file, segm_cls_colors_file, preproc)
        data_fetcher = PASCALDataFetch(
            imgs_dir=eval_params.imgs_segm_dir,
            frame_size=eval_params.frame_size,
            bgr_to_rgb=eval_params.bgr_to_rgb,

        )

        test_segm_models(
            [
                PyTorchModelProcessor(model_dict[original_model_name], original_model_name),
                PyTorchDnnModelProcessor(model_dict[dnn_model_name], dnn_model_name)
            ],
            data_fetcher,
            eval_params,
            original_model_name
        )


if __name__ == "__main__":
    main()
