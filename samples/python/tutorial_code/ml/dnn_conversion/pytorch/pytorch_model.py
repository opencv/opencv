import cv2
import torch.onnx
from torch.autograd import Variable

from ..common.abstract_model import AbstractModel, Framework
from ..common.utils import DNN_LIB, make_dir, get_full_model_path

CURRENT_LIB = "PyTorch"
MODEL_FORMAT = ".onnx"


class PyTorchModelPreparer(AbstractModel):

    def __init__(
            self,
            model_name="default",
            original_model=object,
            batch_size=1,
            default_input_name="input",
            default_output_name="output"
    ):
        self._model_name = model_name
        self._original_model = original_model
        self._batch_size = batch_size
        self._default_input_name = default_input_name
        self._default_output_name = default_output_name

        self._onnx_model_path = self._set_model_path()
        self._dnn_model = self._set_dnn_model()

    def _set_dnn_model(self):
        generated_input = Variable(torch.randn(self._batch_size, 3, 224, 224))
        make_dir(self._onnx_model_path["path"])

        torch.onnx.export(self._original_model,
                          generated_input,
                          self._onnx_model_path["full_path"],
                          verbose=True,
                          input_names=[self._default_input_name],
                          output_names=[self._default_output_name])

        return cv2.dnn.readNetFromONNX(self._onnx_model_path["full_path"])

    def _set_model_path(self):
        model_to_save = self._model_name + MODEL_FORMAT
        return get_full_model_path(CURRENT_LIB.lower(), model_to_save)

    def get_prepared_models(self):
        return {
            CURRENT_LIB + " " + self._model_name: self._original_model,
            DNN_LIB + " " + self._model_name: self._dnn_model
        }


class PyTorchModelProcessor(Framework):
    def __init__(self, prepared_model):
        self._prepared_model = prepared_model

    def get_output(self, input_blob):
        # TBD: in progress, modifications
        pass

    def get_name(self):
        return CURRENT_LIB


class PyTorchDnnModelProcessor(Framework):
    def __init__(self, prepared_dnn_model, in_blob_name="data", out_blob_name="prob"):
        self._prepared_dnn_model = prepared_dnn_model
        self.in_blob_name = in_blob_name
        self.out_blob_name = out_blob_name

    def get_output(self, input_blob):
        self._prepared_dnn_model.setInput(input_blob, self.in_blob_name)
        return self._prepared_dnn_model.forward(self.out_blob_name)

    def get_name(self):
        return DNN_LIB
