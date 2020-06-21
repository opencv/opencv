from torchvision import models

from ..pytorch_model import PyTorchModelPreparer


class PyTorchResNet50(PyTorchModelPreparer):
    def __init__(self, model_name, original_model):
        super(PyTorchResNet50, self).__init__(model_name, original_model)


def main():
    # Test the process of model retrieval
    resnets = PyTorchResNet50(
        model_name="resnet50",
        original_model=models.resnet50(pretrained=True)
    )
    model_dict = resnets.get_prepared_models()
    print(model_dict)


if __name__ == "__main__":
    main()
