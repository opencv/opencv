from tensorflow.keras.applications import (
    VGG16, vgg16,
    VGG19, vgg19,

    ResNet50, resnet,
    ResNet101,
    ResNet152,

    DenseNet121, densenet,
    DenseNet169,
    DenseNet201,

    InceptionResNetV2, inception_resnet_v2,
    InceptionV3, inception_v3,

    MobileNet, mobilenet,
    MobileNetV2, mobilenet_v2,

    NASNetLarge, nasnet,
    NASNetMobile,

    Xception, xception
)

from ..tf_model import TFModelPreparer
from ..tf_model import (
    TFModelProcessor,
    TFDnnModelProcessor
)
from ...common.evaluation.classification.cls_data_fetcher import TFPreprocessedFetch
from ...common.test.cls_model_test_pipeline import ClsModelTestPipeline
from ...common.test.configs.default_preprocess_config import (
    tf_input_blob,
    pytorch_input_blob,
    tf_model_blob_caffe_mode
)
from ...common.utils import set_tf_env, create_extended_parser

model_dict = {
    "vgg16": [VGG16, vgg16, tf_model_blob_caffe_mode],
    "vgg19": [VGG19, vgg19, tf_model_blob_caffe_mode],

    "resnet50": [ResNet50, resnet, tf_model_blob_caffe_mode],
    "resnet101": [ResNet101, resnet, tf_model_blob_caffe_mode],
    "resnet152": [ResNet152, resnet, tf_model_blob_caffe_mode],

    "densenet121": [DenseNet121, densenet, pytorch_input_blob],
    "densenet169": [DenseNet169, densenet, pytorch_input_blob],
    "densenet201": [DenseNet201, densenet, pytorch_input_blob],

    "inceptionresnetv2": [InceptionResNetV2, inception_resnet_v2, tf_input_blob],
    "inceptionv3": [InceptionV3, inception_v3, tf_input_blob],

    "mobilenet": [MobileNet, mobilenet, tf_input_blob],
    "mobilenetv2": [MobileNetV2, mobilenet_v2, tf_input_blob],

    "nasnetlarge": [NASNetLarge, nasnet, tf_input_blob],
    "nasnetmobile": [NASNetMobile, nasnet, tf_input_blob],

    "xception": [Xception, xception, tf_input_blob]
}

CNN_CLASS_ID = 0
CNN_UTILS_ID = 1
DEFAULT_BLOB_PARAMS_ID = 2


class TFClsModel(TFModelPreparer):
    def __init__(self, model_name, original_model):
        super(TFClsModel, self).__init__(model_name, original_model)


def main():
    set_tf_env()

    parser = create_extended_parser(list(model_dict.keys()))
    cmd_args = parser.parse_args()

    model_name = cmd_args.model_name
    model_name_val = model_dict[model_name]

    cls_model = TFClsModel(
        model_name=model_name,
        original_model=model_name_val[CNN_CLASS_ID](
            include_top=True,
            weights="imagenet"
        )
    )

    tf_cls_pipeline = ClsModelTestPipeline(
        network_model=cls_model,
        model_processor=TFModelProcessor,
        dnn_model_processor=TFDnnModelProcessor,
        data_fetcher=TFPreprocessedFetch,
        img_processor=model_name_val[CNN_UTILS_ID].preprocess_input,
        cls_args_parser=parser,
        default_input_blob_preproc=model_name_val[DEFAULT_BLOB_PARAMS_ID]
    )

    tf_cls_pipeline.init_test_pipeline()


if __name__ == "__main__":
    main()
