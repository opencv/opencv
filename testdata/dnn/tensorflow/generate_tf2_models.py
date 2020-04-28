# This script is used to generate test data for OpenCV deep learning module.
import numpy as np
import tensorflow as tf
import shutil

assert(tf.__version__ >= '2.0.0')

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

np.random.seed(2701)

def gen_data(placeholder):
    shape = placeholder.shape.as_list()
    shape[0] = shape[0] if shape[0] else 1  # batch size = 1 instead None
    return np.random.standard_normal(shape).astype(placeholder.dtype.as_numpy_dtype())


def writeBlob(data, name):
    try:
        data = data.numpy()
    except:
        pass

    if data.ndim == 4:
        # NHWC->NCHW
        data = data.transpose(0, 3, 1, 2)
    elif data.ndim == 5:
        # NDHWC->NCDHW
        data = data.transpose(0, 4, 1, 2, 3)

    data = np.ascontiguousarray(data.astype(np.float32))
    np.save(name + '.npy', data)


def save(model, name, **kwargs):
    model.save(name)

    assert(len(kwargs) == 1)

    inputData = gen_data(next(iter(kwargs.values())))
    outputData = model(inputData)

    writeBlob(inputData, name + '_in')
    writeBlob(outputData, name + '_out')

    # Freeze model
    loaded = tf.saved_model.load(name)
    infer = loaded.signatures['serving_default']

    f = tf.function(infer).get_concrete_function(**kwargs)
    f2 = convert_variables_to_constants_v2(f)
    graph_def = f2.graph.as_graph_def()

    # print(graph_def)

    with tf.io.gfile.GFile(name + '_net.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())

    shutil.rmtree(name)


# Test cases ###################################################################
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1, 2, 3)),
  tf.keras.layers.Dense(3, activation='relu'),
])
save(model, 'tf2_dense', flatten_input=tf.TensorSpec(shape=[None, 1, 2, 3], dtype=tf.float32))
################################################################################
model = tf.keras.models.Sequential([
  tf.keras.layers.PReLU(input_shape=(1, 2, 3)),
])
save(model, 'tf2_prelu', p_re_lu_input=tf.TensorSpec(shape=[None, 1, 2, 3], dtype=tf.float32))
################################################################################
model = tf.keras.models.Sequential([
  tf.keras.layers.AveragePooling2D(input_shape=(4, 6, 3), pool_size=(2, 2)),
  tf.keras.layers.Permute((3, 2, 1)),  # NHWC->NCWH
  tf.keras.layers.Flatten()
])
save(model, 'tf2_permute_nhwc_ncwh', average_pooling2d_input=tf.TensorSpec(shape=[None, 4, 6, 3], dtype=tf.float32))
################################################################################

# Uncomment to print the final graph.
# with tf.io.gfile.GFile('tf2_prelu_net.pb', 'rb') as f:
#     graph_def = tf.compat.v1.GraphDef()
#     graph_def.ParseFromString(f.read())
#     print(graph_def)
