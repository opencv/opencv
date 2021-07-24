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


def writeBlob(data, name, nchw = False):
    try:
        data = data.numpy()
    except:
        pass

    if not nchw and data.ndim == 4:
        # NHWC->NCHW
        data = data.transpose(0, 3, 1, 2)
    elif not nchw and data.ndim == 5:
        # NDHWC->NCDHW
        data = data.transpose(0, 4, 1, 2, 3)

    data = np.ascontiguousarray(data.astype(np.float32))
    np.save(name + '.npy', data)


def save(model, name, nchw = False, **kwargs):
    model.save(name)

    assert(len(kwargs) == 1)

    inputData = gen_data(next(iter(kwargs.values())))
    outputData = model(inputData)

    writeBlob(inputData, name + '_in', nchw)
    writeBlob(outputData, name + '_out', nchw)

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

def getGraph(model):
    func = tf.function(lambda x: model(x))
    func = func.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])

    frozen_func = convert_variables_to_constants_v2(func)
    return frozen_func.graph.as_graph_def()

def saveBroken(graph, name):
    tf.io.write_graph(graph_or_graph_def=graph, logdir='.', name=name + '_net.pb', as_text=False)

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
# TF 2.5.0 + python 3.6.13
x_0 = tf.keras.layers.Input(batch_shape = (2, 3, 4))
mid_0 = tf.expand_dims(x_0, axis=0)
x_1 = tf.keras.layers.Input(batch_shape = (2, 3, 4))
mid_1 = tf.reshape(x_1, [1, 2, 3, 4])
out = tf.math.multiply(mid_0, mid_1)
graph = getGraph(tf.keras.Model([x_0, x_1], out))
graph.node[3].op = 'UnknownLayer' # replace ExpandDims op with womething that will never be implemented
saveBroken(graph, 'not_implemented_layer')
################################################################################
# TF 2.5.0 + python 3.6.13
x_0 = tf.keras.layers.Input(batch_shape = (1, 3, 4))
x_1 = tf.keras.layers.Input(batch_shape = (1, 3, 4))
mid = tf.math.multiply(x_0, x_1)
out = tf.math.multiply(mid, x_1)
graph = getGraph(tf.keras.Model([x_0, x_1], out))
graph.node[2].input.pop() # break the connection in the graph
saveBroken(graph, 'broken_layer')
# TF 2.5.0 + python 3.6.13
tf.keras.backend.set_image_data_format('channels_first')
x = tf.keras.layers.Input(batch_shape = (1, 2, 3, 4), name='x')
kernel = np.random.standard_normal((3, 3, 2, 3)).astype(np.float32)
y = tf.nn.conv2d(x, tf.constant(kernel, dtype=tf.float32), data_format = 'NCHW', padding = [[0, 0], [0, 0], [2, 1], [2, 1]], strides = [1, 1, 3, 2])
model = tf.keras.Model(x, y)
save(model, 'conv2d_asymmetric_pads_nchw', True, x=tf.TensorSpec(shape=[1, 2, 3, 4], dtype=tf.float32))
################################################################################
# TF 2.5.0 + python 3.6.13
tf.keras.backend.set_image_data_format('channels_last')
x = tf.keras.layers.Input(batch_shape = (1, 3, 4, 2), name='x')
kernel = np.random.standard_normal((3, 3, 2, 3)).astype(np.float32)
y = tf.nn.conv2d(x, tf.constant(kernel, dtype=tf.float32), data_format = 'NHWC', padding = [[0, 0], [2, 1], [2, 1], [0, 0]], strides = [1, 3, 2, 1])
model = tf.keras.Model(x, y)
save(model, 'conv2d_asymmetric_pads_nhwc', False, x=tf.TensorSpec(shape=[1, 3, 4, 2], dtype=tf.float32))
################################################################################
# TF 2.5.0 + python 3.6.13
tf.keras.backend.set_image_data_format('channels_first')
x = tf.keras.layers.Input(batch_shape = (1, 1, 2, 3), name='x')
y = tf.nn.max_pool(x, ksize=2, data_format = "NCHW", padding = [[0, 0], [0, 0], [1, 0], [1, 1]], strides = [1, 1, 3, 2])
model = tf.keras.Model(x, y)
save(model, 'max_pool2d_asymmetric_pads_nchw', True, x=tf.TensorSpec(shape=(1, 1, 2, 3), dtype=tf.float32))
################################################################################
# TF 2.5.0 + python 3.6.13
tf.keras.backend.set_image_data_format('channels_last')
x = tf.keras.layers.Input(batch_shape = (1, 2, 3, 1), name='x')
y = tf.nn.max_pool(x, ksize=2, data_format = "NHWC", padding = [[0, 0], [1, 0], [1, 1], [0, 0]], strides = [1, 3, 2, 1])
model = tf.keras.Model(x, y)
save(model, 'max_pool2d_asymmetric_pads_nhwc', False, x=tf.TensorSpec(shape=(1, 2, 3, 1), dtype=tf.float32))
################dd################################################################
tf.keras.backend.set_image_data_format('channels_first')
x = tf.keras.layers.Input(batch_shape = (1, 3, 2, 3), name='x')
kernel = np.random.standard_normal((3, 3, 2, 3)).astype(np.float32)
y = tf.compat.v1.nn.conv2d_backprop_input(input_sizes=tf.constant([1, 2, 3, 4]), filter=kernel, out_backprop=x, data_format = "NCHW", padding = [[0, 0], [0, 0], [2, 1], [2, 1]], strides = [1, 1, 3, 2])
model = tf.keras.Model(x, y)
save(model, 'conv2d_backprop_input_asymmetric_pads_nchw', True, x=tf.TensorSpec(shape=(1, 3, 2, 3), dtype=tf.float32))
################################################################################
tf.keras.backend.set_image_data_format('channels_last')
x = tf.keras.layers.Input(batch_shape = (1, 2, 3, 3), name='x')
kernel = np.random.standard_normal((3, 3, 2, 3)).astype(np.float32)
y = tf.compat.v1.nn.conv2d_backprop_input(input_sizes=tf.constant([1, 3, 4, 2]), filter=kernel, out_backprop=x, data_format = "NHWC", padding = [[0, 0], [2, 1], [2, 1], [0, 0]], strides = [1, 3, 2, 1])
model = tf.keras.Model(x, y)
save(model, 'conv2d_backprop_input_asymmetric_pads_nhwc', False, x=tf.TensorSpec(shape=(1, 2, 3, 3), dtype=tf.float32))

# Uncomment to print the final graph.
# with tf.io.gfile.GFile('tf2_prelu_net.pb', 'rb') as f:
#     graph_def = tf.compat.v1.GraphDef()
#     graph_def.ParseFromString(f.read())
#     print(graph_def)
