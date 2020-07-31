from tensorflow.keras.applications import ResNet50

from ..tf_model import TFModelPreparer
from ..tf_model import (
    TFModelProcessor,
    TFDnnModelProcessor
)
from ...common.evaluation.classification.cls_data_fetcher import TFPreprocessedFetch
from ...common.test.cls_model_test_pipeline import ClsModelTestPipeline
from ...common.utils import set_tf_env


class TFResNet50(TFModelPreparer):
    def __init__(self, model_name, original_model):
        super(TFResNet50, self).__init__(model_name, original_model)


def main():
    set_tf_env()

    resnets = TFResNet50(
        model_name="resnet50",
        original_model=ResNet50(
            include_top=True, weights="imagenet"
        )
    )

    tf_resnet50_pipeline = ClsModelTestPipeline(
        network_model=resnets,
        model_processor=TFModelProcessor,
        dnn_model_processor=TFDnnModelProcessor,
        data_fetcher=TFPreprocessedFetch
    )

    # Test the process of model retrieval
    tf_resnet50_pipeline.init_test_pipeline()


if __name__ == "__main__":
    main()
