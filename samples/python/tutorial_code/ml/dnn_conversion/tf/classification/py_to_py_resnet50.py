from tensorflow.keras.applications import ResNet50

from ..tf_model import TFModelPreparer
from ..tf_model import (
    TFModelProcessor,
    TFDnnModelProcessor
)
from ...common.utils import set_tf_env, create_parser


class TFResNet50(TFModelPreparer):
    def __init__(self, model_name, original_model):
        super(TFResNet50, self).__init__(model_name, original_model)


def main():
    parser = create_parser()
    cmd_args = parser.parse_args()
    set_tf_env()

    # Test the process of model retrieval
    resnets = TFResNet50(
        model_name="resnet50",
        original_model=ResNet50(
            include_top=True, weights="imagenet"
        )
    )
    model_dict = resnets.get_prepared_models()
    layers = model_dict["DNN resnet50"].getLayerNames()
    print(layers)

    if cmd_args.is_evaluate:
        from ...common.test_config import TestConfig
        from ...common.accuracy_eval import MeanValueFetch
        from ...common.test.imagenet_cls_test_resnet50 import test_cls_models

        eval_params = TestConfig(batch_size=10)

        model_names = list(model_dict.keys())
        original_model_name = model_names[0]
        dnn_model_name = model_names[1]

        data_fetcher = MeanValueFetch(
            imgs_dir=eval_params.imgs_dir,
            frame_size=eval_params.frame_size,
            bgr_to_rgb=eval_params.bgr_to_rgb
        )

        test_cls_models(
            [
                TFModelProcessor(model_dict[original_model_name], original_model_name),
                TFDnnModelProcessor(model_dict[dnn_model_name], dnn_model_name)
            ],
            data_fetcher,
            eval_params,
            original_model_name
        )


if __name__ == "__main__":
    main()
