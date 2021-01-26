import os

import torch
import torch.onnx
from torch.autograd import Variable
from torchvision import models


def get_pytorch_onnx_model(original_model):
    # define the directory for further converted model save
    onnx_model_path = "models"
    # define the name of further converted model
    onnx_model_name = "resnet50.onnx"

    # create directory for further converted model
    os.makedirs(onnx_model_path, exist_ok=True)

    # get full path to the converted model
    full_model_path = os.path.join(onnx_model_path, onnx_model_name)

    # generate model input
    generated_input = Variable(
        torch.randn(1, 3, 224, 224)
    )

    # model export into ONNX format
    torch.onnx.export(
        original_model,
        generated_input,
        full_model_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )

    return full_model_path


def main():
    # initialize PyTorch ResNet-50 model
    original_model = models.resnet50(pretrained=True)

    # get the path to the converted into ONNX PyTorch model
    full_model_path = get_pytorch_onnx_model(original_model)
    print("PyTorch ResNet-50 model was successfully converted: ", full_model_path)


if __name__ == "__main__":
    main()
