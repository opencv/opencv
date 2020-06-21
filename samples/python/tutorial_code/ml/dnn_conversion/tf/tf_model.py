import cv2
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from ..common.abstract_model import AbstractModel, Framework
from ..common.utils import DNN_LIB, get_full_model_path

CURRENT_LIB = "TF"
MODEL_FORMAT = ".pb"


class TFModelPreparer(AbstractModel):
    def __init__(
            self,
            model_name="default",
            original_model=object,
    ):
        self._model_name = model_name
        self._original_model = original_model
        self._model_to_save = ""

        self._pb_model_path = self._set_model_path()
        self._dnn_model = self._set_dnn_model()

    def _set_dnn_model(self):
        # get model TF graph
        tf_model_graph = tf.function(lambda x: self._original_model(x))

        tf_model_graph = tf_model_graph.get_concrete_function(
            tf.TensorSpec(self._original_model.inputs[0].shape, self._original_model.inputs[0].dtype))

        # obtain frozen concrete function
        frozen_tf_func = convert_variables_to_constants_v2(tf_model_graph)
        frozen_tf_func.graph.as_graph_def()

        # save full TF model
        tf.io.write_graph(graph_or_graph_def=frozen_tf_func.graph,
                          logdir=self._pb_model_path["path"],
                          name=self._model_to_save,
                          as_text=False)

        return cv2.dnn.readNetFromTensorflow(self._pb_model_path["full_path"])

    def _set_model_path(self):
        self._model_to_save = self._model_name + MODEL_FORMAT
        return get_full_model_path(CURRENT_LIB.lower(), self._model_to_save)

    def get_prepared_models(self):
        return {
            CURRENT_LIB + " " + self._model_name: self._original_model,
            DNN_LIB + " " + self._model_name: self._dnn_model
        }


class TFModelProcessor(Framework):
    def __init__(self, prepared_model):
        self._prepared_model = prepared_model

    def get_output(self, input_blob):
        # TBD: in progress, modifications
        pass

    def get_name(self):
        return CURRENT_LIB


class TFDnnModelProcessor(Framework):
    def __init__(self, prepared_dnn_model, in_blob_name="data", out_blob_name="prob"):
        self._prepared_dnn_model = prepared_dnn_model
        self.in_blob_name = in_blob_name
        self.out_blob_name = out_blob_name

    def get_output(self, input_blob):
        self._prepared_dnn_model.setInput(input_blob, self.in_blob_name)
        return self._prepared_dnn_model.forward(self.out_blob_name)[..., 1:1001]

    def get_name(self):
        return DNN_LIB
