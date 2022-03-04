from torchvision import models

from ..pytorch_model import (
    PyTorchModelPreparer,
    PyTorchModelProcessor,
    PyTorchDnnModelProcessor
)
from ...common.evaluation.classification.cls_data_fetcher import PyTorchPreprocessedFetch
from ...common.test.cls_model_test_pipeline import ClsModelTestPipeline
from ...common.test.configs.default_preprocess_config import pytorch_resize_input_blob
from ...common.test.configs.test_config import TestClsConfig
from ...common.utils import set_pytorch_env, create_extended_parser

model_dict = {
    "alexnet": models.alexnet,

    "vgg11": models.vgg11,
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,

    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,

    "squeezenet1_0": models.squeezenet1_0,
    "squeezenet1_1": models.squeezenet1_1,

    "resnext50_32x4d": models.resnext50_32x4d,
    "resnext101_32x8d": models.resnext101_32x8d,

    "wide_resnet50_2": models.wide_resnet50_2,
    "wide_resnet101_2": models.wide_resnet101_2
}


class PyTorchClsModel(PyTorchModelPreparer):
    def __init__(self, height, width, model_name, original_model):
        super(PyTorchClsModel, self).__init__(height, width, model_name, original_model)


def main():
    set_pytorch_env()

    parser = create_extended_parser(list(model_dict.keys()))
    cmd_args = parser.parse_args()
    model_name = cmd_args.model_name

    cls_model = PyTorchClsModel(
        height=TestClsConfig().frame_size,
        width=TestClsConfig().frame_size,
        model_name=model_name,
        original_model=model_dict[model_name](pretrained=True)
    )

    pytorch_cls_pipeline = ClsModelTestPipeline(
        network_model=cls_model,
        model_processor=PyTorchModelProcessor,
        dnn_model_processor=PyTorchDnnModelProcessor,
        data_fetcher=PyTorchPreprocessedFetch,
        cls_args_parser=parser,
        default_input_blob_preproc=pytorch_resize_input_blob
    )

    pytorch_cls_pipeline.init_test_pipeline()


if __name__ == "__main__":
    main()
