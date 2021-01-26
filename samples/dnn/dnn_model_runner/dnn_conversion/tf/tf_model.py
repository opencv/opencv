import cv2
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from ..common.abstract_model import AbstractModel, Framework
from ..common.utils import DNN_LIB, get_full_model_path

CURRENT_LIB = "TF"
MODEL_FORMAT = ".pb"


class TFModelPreparer(AbstractModel):
    """ Class for the preparation of the TF models: original and converted OpenCV Net.

    Args:
        model_name: TF model name
        original_model: TF configured model object or session
        is_ready_graph: indicates whether ready .pb file already exists
        tf_model_graph_path: path to the existing frozen TF graph
    """

    def __init__(
            self,
            model_name="default",
            original_model=None,
            is_ready_graph=False,
            tf_model_graph_path=""
    ):
        self._model_name = model_name
        self._original_model = original_model
        self._model_to_save = ""

        self._is_ready_to_transfer_graph = is_ready_graph
        self.model_path = self._set_model_path(tf_model_graph_path)
        self._dnn_model = self._set_dnn_model()

    def _set_dnn_model(self):
        if not self._is_ready_to_transfer_graph:
            # get model TF graph
            tf_model_graph = tf.function(lambda x: self._original_model(x))

            tf_model_graph = tf_model_graph.get_concrete_function(
                tf.TensorSpec(self._original_model.inputs[0].shape, self._original_model.inputs[0].dtype))

            # obtain frozen concrete function
            frozen_tf_func = convert_variables_to_constants_v2(tf_model_graph)
            frozen_tf_func.graph.as_graph_def()

            # save full TF model
            tf.io.write_graph(graph_or_graph_def=frozen_tf_func.graph,
                              logdir=self.model_path["path"],
                              name=self._model_to_save,
                              as_text=False)

        return cv2.dnn.readNetFromTensorflow(self.model_path["full_path"])

    def _set_model_path(self, tf_pb_file_path):
        """ Method for setting model paths.

        Args:
            tf_pb_file_path: path to the existing TF .pb

        Returns:
            dictionary, where full_path key means saved model path and its full name.
        """
        model_paths_dict = {
            "path": "",
            "full_path": tf_pb_file_path
        }

        if not self._is_ready_to_transfer_graph:
            self._model_to_save = self._model_name + MODEL_FORMAT
            model_paths_dict = get_full_model_path(CURRENT_LIB.lower(), self._model_to_save)

        return model_paths_dict

    def get_prepared_models(self):
        original_lib_name = CURRENT_LIB + " " + self._model_name
        configured_model_dict = {
            original_lib_name: self._original_model,
            DNN_LIB + " " + self._model_name: self._dnn_model
        }
        return configured_model_dict


class TFModelProcessor(Framework):
    def __init__(self, prepared_model, model_name):
        self._prepared_model = prepared_model
        self._name = model_name

    def get_output(self, input_blob):
        assert len(input_blob.shape) == 4
        batch_tf = input_blob.transpose(0, 2, 3, 1)
        out = self._prepared_model(batch_tf)
        return out

    def get_name(self):
        return CURRENT_LIB


class TFDnnModelProcessor(Framework):
    def __init__(self, prepared_dnn_model, model_name):
        self._prepared_dnn_model = prepared_dnn_model
        self._name = model_name

    def get_output(self, input_blob):
        self._prepared_dnn_model.setInput(input_blob)
        ret_val = self._prepared_dnn_model.forward()
        return ret_val

    def get_name(self):
        return DNN_LIB
