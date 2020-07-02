import os

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from ..common.utils import MODEL_PATH_ROOT

MODEL_PATH_ROOT_TF = MODEL_PATH_ROOT.format("tf")


def save_tf_model_proto(tf_model, model_name):
    # get model TF graph
    tf_model_graph = tf.function(lambda x: tf_model(x))

    # get concrete function
    tf_model_graph = tf_model_graph.get_concrete_function(
        tf.TensorSpec(tf_model.inputs[0].shape, tf_model.inputs[0].dtype))

    # obtain frozen concrete function
    frozen_tf_func = convert_variables_to_constants_v2(tf_model_graph)
    # get frozen graph
    frozen_tf_func.graph.as_graph_def()

    pb_model_name = model_name + ".pb"

    # save full tf model
    tf.io.write_graph(graph_or_graph_def=frozen_tf_func.graph,
                      logdir=MODEL_PATH_ROOT_TF,
                      name=pb_model_name,
                      as_text=False)

    return os.path.join(MODEL_PATH_ROOT_TF, pb_model_name)
