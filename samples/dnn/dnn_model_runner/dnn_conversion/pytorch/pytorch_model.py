import os

import cv2
import torch.onnx
from torch.autograd import Variable

from ..common.abstract_model import AbstractModel, Framework
from ..common.utils import DNN_LIB, get_full_model_path

CURRENT_LIB = "PyTorch"
MODEL_FORMAT = ".onnx"


class PyTorchModelPreparer(AbstractModel):

    def __init__(
            self,
            height,
            width,
            model_name="default",
            original_model=object,
            batch_size=1,
            default_input_name="input",
            default_output_name="output"
    ):
        self._height = height
        self._width = width
        self._model_name = model_name
        self._original_model = original_model
        self._batch_size = batch_size
        self._default_input_name = default_input_name
        self._default_output_name = default_output_name

        self.model_path = self._set_model_path()
        self._dnn_model = self._set_dnn_model()

    def _set_dnn_model(self):
        generated_input = Variable(torch.randn(
            self._batch_size, 3, self._height, self._width)
        )
        os.makedirs(self.model_path["path"], exist_ok=True)
        torch.onnx.export(
            self._original_model,
            generated_input,
            self.model_path["full_path"],
            verbose=True,
            input_names=[self._default_input_name],
            output_names=[self._default_output_name],
            opset_version=11
        )

        return cv2.dnn.readNetFromONNX(self.model_path["full_path"])

    def _set_model_path(self):
        model_to_save = self._model_name + MODEL_FORMAT
        return get_full_model_path(CURRENT_LIB.lower(), model_to_save)

    def get_prepared_models(self):
        return {
            CURRENT_LIB + " " + self._model_name: self._original_model,
            DNN_LIB + " " + self._model_name: self._dnn_model
        }


class PyTorchModelProcessor(Framework):
    def __init__(self, prepared_model, model_name):
        self._prepared_model = prepared_model
        self._name = model_name

    def get_output(self, input_blob):
        tensor = torch.FloatTensor(input_blob)
        self._prepared_model.eval()

        with torch.no_grad():
            model_out = self._prepared_model(tensor)

        # segmentation case
        if len(model_out) == 2:
            model_out = model_out['out']

        out = model_out.detach().numpy()
        return out

    def get_name(self):
        return self._name


class PyTorchDnnModelProcessor(Framework):
    def __init__(self, prepared_dnn_model, model_name):
        self._prepared_dnn_model = prepared_dnn_model
        self._name = model_name

    def get_output(self, input_blob):
        self._prepared_dnn_model.setInput(input_blob, '')
        return self._prepared_dnn_model.forward()

    def get_name(self):
        return self._name
