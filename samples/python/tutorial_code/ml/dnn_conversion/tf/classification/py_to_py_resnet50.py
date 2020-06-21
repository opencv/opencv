from tensorflow.keras.applications import ResNet50

from ..tf_model import TFModelPreparer


class TFResNet50(TFModelPreparer):
    def __init__(self, model_name, original_model):
        super(TFResNet50, self).__init__(model_name, original_model)


def main():
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


if __name__ == "__main__":
    main()
