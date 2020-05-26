# This script is used to generate test data for OpenCV deep learning module.
import numpy as np
import tensorflow as tf
import os
import argparse
import struct
import cv2 as cv

from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

np.random.seed(2701)

def gen_data(placeholder):
    shape = placeholder.shape.as_list()
    shape[0] = shape[0] if shape[0] else 1  # batch size = 1 instead None
    return np.random.standard_normal(shape).astype(placeholder.dtype.as_numpy_dtype())

def prepare_for_dnn(sess, graph_def, in_node, out_node, out_graph, dtype, optimize=True, quantize=False):
    # Freeze graph. Replaces variables to constants.
    graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [out_node])
    if optimize:
        # Optimize graph. Removes training-only ops, unused nodes.
        graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, [in_node], [out_node], dtype.as_datatype_enum)
        # Fuse constant operations.
        transforms = ["fold_constants(ignore_errors=True)"]
        if quantize:
            transforms += ["quantize_weights(minimum_size=0)"]
        transforms += ["sort_by_execution_order"]
        graph_def = TransformGraph(graph_def, [in_node], [out_node], transforms)
    # Serialize
    with tf.gfile.FastGFile(out_graph, 'wb') as f:
        f.write(graph_def.SerializeToString())

tf.reset_default_graph()
tf.Graph().as_default()
tf.set_random_seed(324)
sess = tf.Session()

# Use this variable to switch behavior of layers.
isTraining = tf.placeholder(tf.bool, name='isTraining')

def writeBlob(data, name):
    if data.ndim == 4:
        # NHWC->NCHW
        np.save(name + '.npy', data.transpose(0, 3, 1, 2).astype(np.float32))
    elif data.ndim == 5:
        # NDHWC->NCDHW
        np.save(name + '.npy', data.transpose(0, 4, 1, 2, 3).astype(np.float32))
    else:
        # Save raw data.
        np.save(name + '.npy', data.astype(np.float32))

def runModel(inpName, outName, name):
    with tf.Session(graph=tf.Graph()) as localSession:
        localSession.graph.as_default()

        with tf.gfile.FastGFile(name + '_net.pb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        inputData = gen_data(inp)
        outputData = localSession.run(localSession.graph.get_tensor_by_name(outName),
                                      feed_dict={localSession.graph.get_tensor_by_name(inp.name): inputData})
        writeBlob(inputData, name + '_in')
        writeBlob(outputData, name + '_out')

def save(inp, out, name, quantize=False, optimize=True):
    sess.run(tf.global_variables_initializer())

    inputData = gen_data(inp)
    outputData = sess.run(out, feed_dict={inp: inputData, isTraining: False})
    writeBlob(inputData, name + '_in')
    writeBlob(outputData, name + '_out')

    prepare_for_dnn(sess, sess.graph.as_graph_def(), inp.name[:inp.name.rfind(':')],
                    out.name[:out.name.rfind(':')], name + '_net.pb', inp.dtype,
                    optimize, quantize)

    # By default, float16 weights are stored in repeated tensor's field called
    # `half_val`. It has type int32 with leading zeros for unused bytes.
    # This type is encoded by Varint that means only 7 bits are used for value
    # representation but the last one is indicated the end of encoding. This way
    # float16 might takes 1 or 2 or 3 bytes depends on value. To impove compression,
    # we replace all `half_val` values to `tensor_content` using only 2 bytes for everyone.
    with tf.gfile.FastGFile(name + '_net.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            if 'value' in node.attr:
                halfs = node.attr["value"].tensor.half_val
                if not node.attr["value"].tensor.tensor_content and halfs:
                    node.attr["value"].tensor.tensor_content = struct.pack('H' * len(halfs), *halfs)
                    node.attr["value"].tensor.ClearField('half_val')
        tf.train.write_graph(graph_def, "", name + '_net.pb', as_text=False)

# Test cases ###################################################################
# shape: NHWC
for dtype, prefix in zip([tf.float32, tf.float16], ['', 'fp16_']):
    inp = tf.placeholder(dtype, [1, 6, 5, 3], 'input')
    conv = tf.layers.conv2d(inputs=inp, filters=3, kernel_size=[1, 1],
                            activation=tf.nn.relu,
                            bias_initializer=tf.random_normal_initializer())
    save(inp, conv, prefix + 'single_conv')
################################################################################
    inp = tf.placeholder(dtype, [3, 7, 5, 4], 'input')
    conv = tf.layers.conv2d(inputs=inp, filters=5, kernel_size=[5, 3], padding='SAME',
                            use_bias=False)
    activation_abs = tf.abs(conv)
    save(inp, activation_abs, prefix + 'padding_same')
################################################################################
    inp = tf.placeholder(dtype, [2, 4, 6, 5], 'input')
    conv = tf.layers.conv2d(inputs=inp, filters=4, kernel_size=[3, 5], padding='VALID',
                            activation=tf.nn.elu, bias_initializer=tf.random_normal_initializer())
    save(inp, conv, prefix + 'padding_valid')
################################################################################
    inp = tf.placeholder(dtype, [3, 2, 3, 4], 'input')
    conv = tf.layers.conv2d(inputs=inp, filters=4, kernel_size=[1, 1], activation=tf.nn.tanh,
                            bias_initializer=tf.random_uniform_initializer(0, 1))
    conv2 = tf.layers.conv2d(inputs=inp, filters=4, kernel_size=[1, 1], activation=tf.nn.sigmoid,
                             bias_initializer=None)
    eltwise_add_mul = (inp * 0.31 + 2 * conv) * conv2
    save(inp, eltwise_add_mul, prefix + 'eltwise_add_mul')
################################################################################
    inp = tf.placeholder(dtype, [1, 4, 5, 1], 'input')
    conv = tf.layers.conv2d(inputs=inp, filters=4, kernel_size=[3, 1], padding='VALID')
    padded = tf.pad(conv, [[0, 0], [0, 2], [0, 0], [0, 0]])
    merged = tf.concat([padded, inp], axis=3)
    save(inp, merged, prefix + 'pad_and_concat')
###############################################################################
    inp = tf.placeholder(dtype, [1, 6, 6, 2], 'input')
    conv = tf.layers.conv2d(inputs=inp, filters=3, kernel_size=[3, 3], padding='SAME')
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2)
    save(inp, pool, prefix + 'max_pool_even')
################################################################################
    inp = tf.placeholder(dtype, [1, 7, 7, 2], 'input')
    conv = tf.layers.conv2d(inputs=inp, filters=3, kernel_size=[3, 3], padding='SAME')
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=3, strides=2, padding='VALID')
    save(inp, pool, prefix + 'max_pool_odd_valid')
################################################################################
    inp = tf.placeholder(dtype, [1, 7, 7, 2], 'input')
    conv = tf.layers.conv2d(inputs=inp, filters=3, kernel_size=[3, 3], padding='SAME')
    relu = tf.nn.relu6(conv * 10)
    pool = tf.layers.max_pooling2d(inputs=relu, pool_size=2, strides=2, padding='SAME')
    save(inp, pool, prefix + 'max_pool_odd_same')
################################################################################
    inp = tf.placeholder(dtype, [1, 5, 6, 2], 'input')
    deconv_weights = tf.Variable(tf.random_normal([5, 3, 4, 2], dtype=dtype), name='deconv_weights')
    deconv = tf.nn.conv2d_transpose(value=inp, filter=deconv_weights,
                                    output_shape=[1, 9, 8, 4], strides=[1, 1, 1, 1],
                                    padding='VALID')
    deconv_bias = tf.contrib.layers.bias_add(deconv, activation_fn=tf.nn.relu,
                                             initializer=tf.random_normal_initializer())
    save(inp, deconv_bias, prefix + 'deconvolution')
################################################################################
inp = tf.placeholder(tf.float32, [2, 5, 4, 3], 'input')
bn = tf.contrib.layers.batch_norm(inputs=inp, fused=True, is_training=False,
                                  scale=True, param_initializers={
                                    'beta': tf.random_normal_initializer(),
                                    'gamma': tf.random_normal_initializer(),
                                    'moving_mean': tf.random_uniform_initializer(-2, -1),
                                    'moving_variance': tf.random_uniform_initializer(1, 2)
                                  })
save(inp, bn, 'fused_batch_norm')
################################################################################
inp = tf.placeholder(tf.float32, [2, 5, 6, 3], 'input')
weights = tf.Variable(tf.random_normal([3, 3, 3, 4]), name='weights')
conv = tf.nn.atrous_conv2d(inp, weights, rate=2, padding='VALID')
relu = tf.nn.relu(conv)
save(inp, relu, 'atrous_conv2d_valid')
################################################################################
inp = tf.placeholder(tf.float32, [2, 5, 10, 3], 'input')
weights = tf.Variable(tf.random_normal([3, 5, 3, 4]), name='weights')
conv = tf.nn.atrous_conv2d(inp, weights, rate=2, padding='SAME')
relu = tf.nn.relu(conv)
save(inp, relu, 'atrous_conv2d_same')
################################################################################
inp = tf.placeholder(tf.float32, [2, 5, 4, 3], 'input')
bn = tf.contrib.layers.batch_norm(inputs=inp, is_training=False,
                                  scale=True, param_initializers={
                                    'beta': tf.random_normal_initializer(),
                                    'gamma': tf.random_normal_initializer(),
                                    'moving_mean': tf.random_uniform_initializer(-2, -1),
                                    'moving_variance': tf.random_uniform_initializer(1, 2)
                                  })
save(inp, bn, 'batch_norm')
################################################################################
inp = tf.placeholder(tf.float32, [2, 10, 9, 6], 'input')
weights = tf.Variable(tf.random_normal([5, 3, 6, 4]), name='weights')
conv = tf.nn.depthwise_conv2d(input=inp, filter=weights, strides=[1, 1, 1, 1],
                              padding='SAME')
save(inp, conv, 'depthwise_conv2d')
################################################################################
inp = tf.placeholder(tf.float32, [2, 3], 'input')
biases = tf.Variable(tf.random_normal([4]), name='matmul_biases')
weights = tf.Variable(tf.random_normal([3, 4]), name='matmul_weights')
mm = tf.matmul(inp, weights) + biases
save(inp, mm, 'matmul')
################################################################################
from tensorflow.python.framework import function

@function.Defun(tf.float32, func_name='Dropout')
def my_dropout(x):
    return tf.layers.dropout(x, rate=0.1, training=isTraining)

inp = tf.placeholder(tf.float32, [1, 10, 10, 3], 'input')
conv = tf.layers.conv2d(inp, filters=3, kernel_size=[1, 1])
dropout = my_dropout(conv)
relu = tf.nn.relu(dropout)

save(inp, relu, 'defun_dropout')
################################################################################
# Use not 4 dimensions to save the raw data (see writeBlob function)
inp = tf.placeholder(tf.float32, [2, 3, 4], 'input')
shift = tf.Variable(tf.random_normal([2, 3, 4]), name='shift')
shifted = tf.add(inp, shift, name='shifted')
reshape = tf.reshape(shifted, [4, 3, 2], 'reshaped')
save(inp, reshape, 'shift_reshape_no_reorder')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3], 'input')
reshape = tf.reshape(inp, [3, 1, 2], 'reshaped')
save(inp, reshape, 'reshape_no_reorder')
################################################################################
inp = tf.placeholder(tf.float32, [2, 10, 10, 3], 'input')
pad = tf.pad(inp, [[0, 0], [3, 3], [3, 3], [0, 0]])
conv = tf.layers.conv2d(inp, filters=4, kernel_size=[5, 5], strides=(2, 2),
                        bias_initializer=tf.random_normal_initializer())
save(inp, conv, 'spatial_padding')
################################################################################
inp = tf.placeholder(tf.float32, [1, 10, 10, 3], 'input')
pad = tf.pad(inp, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
save(inp, pad, 'mirror_pad')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3], 'input')
bn = tf.add(inp, tf.Variable(tf.random_normal(inp.shape)))
reshape = tf.reshape(bn, [-1, 3])
save(inp, reshape, 'reshape_reduce')
################################################################################
times = 4  # Sequence length (number of batches in different time stamps)
batch_size = 2
input_size = 5*6*3  # W*H*C
output_size = 10
# Define LSTM blobk.
inp = tf.placeholder(tf.float32, [times, batch_size, input_size], 'input')
lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(output_size, forget_bias=0.9,
                                           cell_clip=0.4, use_peephole=True)
outputs, state = lstm_cell(inp, dtype=tf.float32)
# shape(outputs) is a (times, batch_size, output_size)

# Slice the last time iteration:
last_output = tf.slice(outputs, [times - 1, 0, 0], [1, -1, output_size])
# shape(last_output) is a (1, batch_size, output_size)

# Remove time axis.
last_output = tf.reshape(last_output, [-1, 10])
# shape(last_output) is a (batch_size, output_size)

# Fully-connected
weights = tf.Variable(tf.random_normal([10, 2]))
biases = tf.Variable(tf.random_normal([2]))
sigmoid = tf.nn.sigmoid(tf.matmul(last_output, weights) + biases)
save(inp, sigmoid, 'lstm')
################################################################################
bgr = tf.placeholder(tf.float32, [4, 5, 6, 3], 'input')
b, g, r = tf.split(bgr, num_or_size_splits=3, axis=3)
rgb = tf.concat([r, g, b], axis=3)
alpha, beta = tf.split(rgb, num_or_size_splits=2, axis=0)
res = tf.layers.conv2d(alpha, filters=1, kernel_size=[1, 1]) + \
      tf.layers.conv2d(beta, filters=1, kernel_size=[1, 1])
save(bgr, res, 'split_equals')
################################################################################
inp = tf.placeholder(tf.float32, [2, 10, 11, 3], 'input')
conv = tf.layers.conv2d(inp, filters=7, kernel_size=[1, 1])
scaled = tf.image.resize_nearest_neighbor(conv, size=(15, 8))
scaled = tf.image.resize_nearest_neighbor(scaled, size=(9, 12))
save(inp, scaled, 'resize_nearest_neighbor')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
bn = tf.layers.batch_normalization(inp, training=isTraining, fused=False, name='batch_norm',
                                   beta_initializer=tf.random_normal_initializer(),
                                   gamma_initializer=tf.random_normal_initializer(),
                                   moving_mean_initializer=tf.random_uniform_initializer(-2, 1),
                                   moving_variance_initializer=tf.random_uniform_initializer(0.1, 2),)
save(inp, bn, 'batch_norm_text')
################################################################################
inp = tf.placeholder(tf.float32, [2, 4, 5], 'input')
flatten = tf.contrib.layers.flatten(inp)
save(inp, flatten, 'flatten')
################################################################################
inp = tf.placeholder(tf.float32, [2, 2, 2, 2, 3], 'input')
conv = tf.layers.conv3d(inp, filters=2, kernel_size=[1, 1, 1], padding='SAME')
concat = tf.concat([inp, conv], axis=-1)
save(inp, concat, 'concat_3d')
################################################################################
# Generate test data for MobileNet-SSD object detection model from TensorFlow
# model zoo, http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
# 1. Download and extract an archive
# 2. Place frozen_inference_graph.pb as a ssd_mobilenet_v1_coco.pb nearby this script
with tf.gfile.FastGFile('../ssd_mobilenet_v1_coco.pb') as f:
    # Load the model
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session(graph=tf.Graph()) as localSession:
    # Restore session
    localSession.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Receive output
    inp = cv.imread('../street.png')
    inp = cv.resize(inp, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    out = localSession.run([localSession.graph.get_tensor_by_name('concat:0'),    # All detections
                            localSession.graph.get_tensor_by_name('concat_1:0'),  # Classification
                            localSession.graph.get_tensor_by_name('num_detections:0'),     # Postprocessed output
                            localSession.graph.get_tensor_by_name('detection_scores:0'),   #
                            localSession.graph.get_tensor_by_name('detection_boxes:0'),    #
                            localSession.graph.get_tensor_by_name('detection_classes:0')], #
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    np.save('ssd_mobilenet_v1_coco.concat.npy', out[0])
    np.save('ssd_mobilenet_v1_coco.concat_1.npy', out[1])
    # Pack detections in format [id, class_id, confidence, left, top, right, bottom]
    num_detections = int(out[2][0])
    detections = np.zeros([1, 1, num_detections, 7], np.float32)
    detections[0, 0, :, 0] = 0  # bounding boxes ids
    detections[0, 0, :, 1] = out[5][0]
    detections[0, 0, :, 2] = out[3][0]
    detections[0, 0, :, 3:] = out[4][0][:, [1, 0, 3, 2]]
    # Detections are sorted in descending by confidence order. Group them by classes
    # to make OpenCV test more simple.
    detections = sorted(detections[0, 0, :, :], cmp=lambda x, y: -1 if x[1] < y[1] and x[2] < y[2] else 0)
    np.save('ssd_mobilenet_v1_coco.detection_out.npy', detections)
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
conv = tf.layers.conv2d(inp, filters=5, kernel_size=[1, 1],
                        activation=tf.nn.relu,
                        bias_initializer=tf.random_normal_initializer())
flattened = tf.reshape(conv, [1, -1], 'reshaped')
biases = tf.Variable(tf.random_normal([10]), name='matmul_biases')
weights = tf.Variable(tf.random_normal([2*3*5, 10]), name='matmul_weights')
mm = tf.matmul(flattened, weights) + biases
save(inp, mm, 'nhwc_reshape_matmul')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
conv = tf.layers.conv2d(inp, filters=5, kernel_size=[1, 1],
                        activation=tf.nn.relu,
                        bias_initializer=tf.random_normal_initializer())
transposed = tf.transpose(conv, [0, 1, 2, 3])
flattened = tf.reshape(transposed, [1, -1], 'reshaped')
biases = tf.Variable(tf.random_normal([10]), name='matmul_biases')
weights = tf.Variable(tf.random_normal([2*3*5, 10]), name='matmul_weights')
mm = tf.matmul(flattened, weights) + biases
save(inp, flattened, 'nhwc_transpose_reshape_matmul')
################################################################################
inp = tf.placeholder(tf.float32, [1, 6, 5, 3], 'input')
conv = tf.layers.conv2d(inputs=inp, filters=3, kernel_size=[1, 1],
                        activation=tf.nn.relu,
                        bias_initializer=tf.random_normal_initializer())
save(inp, conv, 'uint8_single_conv', quantize=True)
runModel(inp, conv.name, 'uint8_single_conv')
################################################################################
inp = tf.placeholder(tf.float32, [1, 4, 4, 1], 'input')
conv = tf.layers.conv2d(inp, filters=3, kernel_size=[3, 3], padding='SAME')
pool = tf.layers.average_pooling2d(conv, pool_size=3, strides=1, padding='SAME')
save(inp, pool, 'ave_pool_same')
################################################################################
inp = tf.placeholder(tf.float32, [1, 4, 6, 1], 'input')
conv = tf.layers.conv2d(inp, filters=3, kernel_size=[1, 1], padding='SAME')
sliced = tf.slice(conv, [0, 1, 2, 0], [-1, 3, 4, 1])
save(inp, sliced, 'slice_4d')
################################################################################
inp = tf.placeholder(tf.float32, [1, 4, 4, 1], 'input')
#                                             ky kx out in
deconv_weights = tf.Variable(tf.random_normal([3, 3, 2, 1], dtype=tf.float32))
deconv = tf.nn.conv2d_transpose(inp, deconv_weights,
                                output_shape=[1, 4, 4, 2], strides=[1, 1, 1, 1],
                                padding='SAME')
leakyRelu = tf.nn.leaky_relu(deconv, alpha=0.2)
save(inp, leakyRelu, 'deconvolution_same')
# ################################################################################
inp = tf.placeholder(tf.float32, [1, 3, 3, 1], 'input')
deconv_weights = tf.Variable(tf.random_normal([3, 3, 2, 1], dtype=tf.float32))
deconv = tf.nn.conv2d_transpose(inp, deconv_weights,
                                output_shape=[1, 5, 5, 2], strides=[1, 2, 2, 1],
                                padding='SAME')
save(inp, deconv, 'deconvolution_stride_2_same')
################################################################################
inp = tf.placeholder(tf.float32, [1, 3, 2, 1], 'input')
deconv_weights = tf.Variable(tf.random_normal([3, 3, 2, 1], dtype=tf.float32))
deconv = tf.nn.conv2d_transpose(inp, deconv_weights,
                                output_shape=[1, 8, 6, 2], strides=[1, 2, 2, 1],
                                padding='VALID')
save(inp, deconv, 'deconvolution_adj_pad_valid')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 2, 1], 'input')
deconv_weights = tf.Variable(np.ones([3, 3, 1, 1]), dtype=tf.float32)
deconv = tf.nn.conv2d_transpose(inp, deconv_weights,
                                output_shape=[1, 4, 4, 1], strides=[1, 2, 2, 1],
                                padding='SAME')
save(inp, deconv, 'deconvolution_adj_pad_same')
################################################################################
inp = tf.placeholder(tf.float32, [1, 3, 4, 5], 'input')
gamma = tf.Variable(tf.random_normal([5], dtype=tf.float32))
beta = tf.Variable(tf.random_normal([5], dtype=tf.float32))
bn = tf.nn.fused_batch_norm(inp, gamma, beta, epsilon=1e-5, is_training=True)[0]
save(inp, bn, 'mvn_batch_norm')
################################################################################
inp = tf.placeholder(tf.float32, [1, 1, 1, 5], 'input')
gamma = tf.Variable(tf.random_normal([5], dtype=tf.float32))
beta = tf.Variable(tf.random_normal([5], dtype=tf.float32))
bn = tf.nn.fused_batch_norm(inp, gamma, beta, epsilon=1e-5, is_training=True)[0]
save(inp, bn, 'mvn_batch_norm_1x1')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
bn = tf.layers.batch_normalization(inp, training=False, fused=False, name='unfused_batch_norm',
                                   beta_initializer=tf.random_normal_initializer(),
                                   gamma_initializer=tf.random_normal_initializer(),
                                   moving_mean_initializer=tf.random_uniform_initializer(-2, 1),
                                   moving_variance_initializer=tf.random_uniform_initializer(0.1, 2),)
save(inp, bn, 'unfused_batch_norm', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
bn = tf.layers.batch_normalization(inp, training=False, fused=True, name='fused_batch_norm_no_gamma',
                                   beta_initializer=tf.random_normal_initializer(),
                                   scale=False,
                                   moving_mean_initializer=tf.random_uniform_initializer(-2, 1),
                                   moving_variance_initializer=tf.random_uniform_initializer(0.1, 2),)
save(inp, bn, 'fused_batch_norm_no_gamma', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
bn = tf.layers.batch_normalization(inp, training=False, fused=False, name='unfused_batch_norm_no_gamma',
                                   beta_initializer=tf.random_normal_initializer(),
                                   scale=False,
                                   moving_mean_initializer=tf.random_uniform_initializer(-2, 1),
                                   moving_variance_initializer=tf.random_uniform_initializer(0.1, 2),)
save(inp, bn, 'unfused_batch_norm_no_gamma', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3], 'input')
flatten = tf.contrib.layers.flatten(inp)
save(inp, flatten, 'unfused_flatten', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [None, 2, 3], 'input')
flatten = tf.contrib.layers.flatten(inp)
save(inp, flatten, 'unfused_flatten_unknown_batch', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
relu = tf.maximum(inp, 0.01 * inp, name='leaky_relu') * 2
save(inp, relu, 'leaky_relu_order1', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
relu = tf.maximum(inp, inp * 0.01, name='leaky_relu') * 2
save(inp, relu, 'leaky_relu_order2', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
relu = tf.maximum(0.01 * inp, inp, name='leaky_relu') * 2
save(inp, relu, 'leaky_relu_order3', optimize=False)
################################################################################
from tensorflow import keras as K
model = K.models.Sequential()
model.add(K.layers.Softmax(name='keras_softmax', input_shape=(2, 3, 4)))
sess = K.backend.get_session()
sess.as_default()
save(sess.graph.get_tensor_by_name('keras_softmax_input:0'),
     sess.graph.get_tensor_by_name('keras_softmax/truediv:0'), 'keras_softmax', optimize=False)
################################################################################
model = K.models.Sequential()
model.add(K.layers.Conv2D(filters=4, kernel_size=1, data_format='channels_last',
                          name='keras_mobilenet_head_conv', input_shape=(2, 3, 4)))
model.add(K.layers.GlobalAveragePooling2D(name='keras_mobilenet_head_pool'))
model.add(K.layers.Reshape((1, 1, 4), name='keras_mobilenet_head_reshape'))
sess = K.backend.get_session()
sess.as_default()
save(sess.graph.get_tensor_by_name('keras_mobilenet_head_conv_input:0'),
     sess.graph.get_tensor_by_name('keras_mobilenet_head_reshape/Reshape:0'),
     'keras_mobilenet_head', optimize=False)
################################################################################
def keras_relu6(x):
    return K.activations.relu(x, max_value=6)

inp = K.Input(shape=(2, 3, 4), name='keras_relu6_input')
relu = K.layers.Activation(keras_relu6, name='keras_relu6')(inp)
model = K.Model(inp, relu)
sess = K.backend.get_session()
sess.as_default()
save(sess.graph.get_tensor_by_name('keras_relu6_input:0'),
     sess.graph.get_tensor_by_name('keras_relu6/clip_by_value:0'), 'keras_relu6', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [2, 3], 'input')
max_node = tf.clip_by_value(inp, clip_value_min=0, clip_value_max=1)
save(inp, max_node, 'clip_by_value')
################################################################################
inp = tf.placeholder(tf.float32, [2, 3, 4, 5], 'input')
reduced = tf.reduce_mean(inp, axis=[1, 2], keepdims=True)
save(inp, reduced, 'reduce_mean')
################################################################################
inp = tf.placeholder(tf.float32, [2, 3, 4, 5], 'input')
pool = tf.layers.average_pooling2d(inp, pool_size=1, strides=1, padding='SAME')
l2norm = tf.nn.l2_normalize(pool, axis=-1)
l2norm = tf.nn.l2_normalize(l2norm, axis=[2, 3, 1])
save(inp, l2norm, 'l2_normalize')
################################################################################
inp = tf.placeholder(tf.float32, [2, 3, 4], 'input')
l2norm = tf.nn.l2_normalize(inp, axis=1)
l2norm = tf.nn.l2_normalize(l2norm, axis=-1)
l2norm = tf.nn.l2_normalize(l2norm, axis=[0, 1])
save(inp, l2norm, 'l2_normalize_3d')
################################################################################
model = K.models.Sequential()
model.add(K.layers.Conv2DTranspose(filters=4, kernel_size=3, strides=(2, 2),
                                   data_format='channels_last', name='keras_deconv_valid',
                                   input_shape=(4, 5, 2)))
sess = K.backend.get_session()
sess.as_default()
save(sess.graph.get_tensor_by_name('keras_deconv_valid_input:0'),
     sess.graph.get_tensor_by_name('keras_deconv_valid/BiasAdd:0'),
     'keras_deconv_valid', optimize=True)
################################################################################
model = K.models.Sequential()
model.add(K.layers.Conv2DTranspose(filters=4, kernel_size=3, strides=(2, 2),
                                   data_format='channels_last', name='keras_deconv_same',
                                   input_shape=(4, 5, 2), padding='same'))
sess = K.backend.get_session()
sess.as_default()
save(sess.graph.get_tensor_by_name('keras_deconv_same_input:0'),
     sess.graph.get_tensor_by_name('keras_deconv_same/BiasAdd:0'),
     'keras_deconv_same', optimize=True)
################################################################################
inp = tf.placeholder(tf.float32, [2, 3, 4, 5], 'input')
resized = tf.image.resize_bilinear(inp, size=[9, 8], name='resize_bilinear')
save(inp, resized, 'resize_bilinear')
################################################################################
inp = tf.placeholder(tf.float32, [None, 3, 4, 5], 'input')
resized = tf.image.resize_bilinear(inp, size=[tf.shape(inp)[1]*2, tf.shape(inp)[2]*3],
                                   name='resize_bilinear_factor')
sub_add = resized - 0.3 + 0.3
save(inp, sub_add, 'resize_bilinear_factor', optimize=False)
################################################################################
model = K.models.Sequential()
model.add(K.layers.SeparableConv2D(filters=4, kernel_size=3, strides=(1, 1),
                                   dilation_rate=(2, 3), name='keras_atrous_conv2d_same',
                                   input_shape=(11, 12, 2), padding='same'))
sess = K.backend.get_session()
sess.as_default()
save(sess.graph.get_tensor_by_name('keras_atrous_conv2d_same_input:0'),
     sess.graph.get_tensor_by_name('keras_atrous_conv2d_same/BiasAdd:0'),
     'keras_atrous_conv2d_same', optimize=True)
################################################################################
# Generate test data for Faster-RCNN object detection model from TensorFlow
# model zoo, http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
for name in ['faster_rcnn_inception_v2_coco_2018_01_28', 'faster_rcnn_resnet50_coco_2018_01_28']:
    with tf.gfile.FastGFile(os.path.join('..', name + '.pb')) as f:
        # Load the model
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session(graph=tf.Graph()) as localSession:
        # Restore session
        localSession.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Receive output
        inp = cv.imread('../dog416.png')
        inp = cv.resize(inp, (800, 600))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        out = localSession.run([localSession.graph.get_tensor_by_name('num_detections:0'),   #
                                localSession.graph.get_tensor_by_name('detection_scores:0'),   #
                                localSession.graph.get_tensor_by_name('detection_boxes:0'),    #
                                localSession.graph.get_tensor_by_name('detection_classes:0')], #
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        # Pack detections in format [id, class_id, confidence, left, top, right, bottom]
        num_detections = int(out[0][0])
        detections = np.zeros([1, 1, num_detections, 7], np.float32)
        detections[0, 0, :, 1] = out[3][0, :num_detections] - 1
        detections[0, 0, :, 2] = out[1][0, :num_detections]
        detections[0, 0, :, 3:] = out[2][:, :num_detections, [1, 0, 3, 2]]
        np.save(name + '.detection_out.npy', detections)
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
conv1 = tf.layers.conv2d(inp, filters=4, kernel_size=[1, 1])
conv2 = tf.layers.conv2d(inp, filters=4, kernel_size=[1, 1])
flatten1 = tf.contrib.layers.flatten(conv1)
flatten2 = tf.contrib.layers.flatten(conv2)
concat = tf.concat([flatten1, flatten2], axis=1)
bias = tf.contrib.layers.bias_add(concat)  # Add zeros (it has NHWC data format flag)
save(inp, bias, 'concat_axis_1')
################################################################################
inp = tf.placeholder(tf.float32, [1, 3, 5, 8], 'input')  # NCHW input
conv = tf.layers.conv2d(inp, filters=4, kernel_size=[2, 3], data_format='channels_first')
pool = tf.layers.max_pooling2d(conv, pool_size=2, strides=2, data_format='channels_first')
save(inp, pool, 'conv_pool_nchw')
# Input and output have been transposed (see writeBlob)
for name in ['conv_pool_nchw_in.npy', 'conv_pool_nchw_out.npy']:
    np.save(name, np.load(name).transpose(0, 2, 3, 1))
################################################################################
model = K.models.Sequential()

model.add(K.layers.UpSampling2D(size=(3, 2), data_format='channels_last',
                          name='keras_upsampling2d', input_shape=(2, 3, 4)))
sess = K.backend.get_session()
sess.as_default()
save(sess.graph.get_tensor_by_name('keras_upsampling2d_input:0'),
     sess.graph.get_tensor_by_name('keras_upsampling2d/ResizeNearestNeighbor:0'),
     'keras_upsampling2d')
################################################################################
# Generate test data for MobileNet-SSD object detection model from TensorFlow
# model zoo, http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz
with tf.gfile.FastGFile('../ssd_mobilenet_v1_ppn_coco.pb') as f:
    # Load the model
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session(graph=tf.Graph()) as localSession:
    # Restore session
    localSession.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Receive output
    img = cv.imread('../dog416.png')
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    out = localSession.run([localSession.graph.get_tensor_by_name('num_detections:0'),   #
                            localSession.graph.get_tensor_by_name('detection_scores:0'),   #
                            localSession.graph.get_tensor_by_name('detection_boxes:0'),    #
                            localSession.graph.get_tensor_by_name('detection_classes:0')], #
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    # Pack detections in format [id, class_id, confidence, left, top, right, bottom]
    num_detections = int(out[0][0])
    detections = np.zeros([1, 1, num_detections, 7], np.float32)
    detections[0, 0, :, 1] = out[3][0, :num_detections]
    detections[0, 0, :, 2] = out[1][0, :num_detections]
    detections[0, 0, :, 3:] = out[2][:, :num_detections, [1, 0, 3, 2]]
    np.save('ssd_mobilenet_v1_ppn_coco.detection_out.npy', detections)
################################################################################
inp = tf.placeholder(tf.float32, [None, 2, 3], 'input')
flatten = tf.reshape(inp, [-1, 2*3], 'planar')
reshaped = tf.reshape(flatten, tf.shape(inp), 'reshape')
save(inp, reshaped, 'reshape_as_shape', optimize=False)
################################################################################
with tf.gfile.FastGFile('../mask_rcnn_inception_v2_coco_2018_01_28.pb') as f:
    # Load the model
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session(graph=tf.Graph()) as localSession:
    # Restore session
    localSession.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Receive output
    img = cv.imread('../street.png')
    inp = cv.resize(img, (800, 800))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    out = localSession.run([localSession.graph.get_tensor_by_name('num_detections:0'),
                            localSession.graph.get_tensor_by_name('detection_scores:0'),
                            localSession.graph.get_tensor_by_name('detection_boxes:0'),
                            localSession.graph.get_tensor_by_name('detection_classes:0'),
                            localSession.graph.get_tensor_by_name('detection_masks:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    # Pack detections in format [id, class_id, confidence, left, top, right, bottom]
    num_detections = int(out[0][0])
    detections = np.zeros([1, 1, num_detections, 7], np.float32)
    detections[0, 0, :, 1] = out[3][0, :num_detections] - 1
    detections[0, 0, :, 2] = out[1][0, :num_detections]
    detections[0, 0, :, 3:] = out[2][:, :num_detections, [1, 0, 3, 2]]
    np.save('mask_rcnn_inception_v2_coco_2018_01_28.detection_out.npy', detections)
    np.save('mask_rcnn_inception_v2_coco_2018_01_28.detection_masks.npy', out[4])
################################################################################
inp = K.Input(shape=(2, 3, 4), name='keras_pad_concat_input', batch_size=1)
conv = K.layers.Conv2D(filters=4, kernel_size=1, data_format='channels_last',
                       name='keras_pad_concat_conv', input_shape=(2, 3, 4))(inp)

def pad_depth(x, desired_channels):
    y = K.backend.random_uniform_variable(x.shape.as_list()[:-1] + [desired_channels], low=0, high=1)
    return K.layers.concatenate([x, y])

pad = K.layers.Lambda(pad_depth, arguments={'desired_channels': 5}, name='keras_pad_concat')(conv)

sess = K.backend.get_session()
sess.as_default()
save(sess.graph.get_tensor_by_name('keras_pad_concat_input:0'),
     sess.graph.get_tensor_by_name('keras_pad_concat/concatenate/concat:0'),
     'keras_pad_concat', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [2, 3, 4, 5], 'input')
conv = tf.layers.conv2d(inp, filters=5, kernel_size=[1, 1],
                        bias_initializer=tf.random_normal_initializer())
sub = conv - inp
save(inp, sub, 'eltwise_sub')
################################################################################
inp = tf.placeholder(tf.float32, [None, 2, 3, 4], 'input')
conv = tf.layers.conv2d(inp, filters=3, kernel_size=[1, 1])
softmax = tf.contrib.slim.softmax(conv)
save(inp, softmax, 'slim_softmax')
################################################################################
# issue https://github.com/opencv/opencv/issues/14224
inp_node = 'img_inputs'
out_node = 'MobileFaceNet/MobileFaceNet/Conv2d_0/add'
with tf.Session(graph=tf.Graph()) as localSession:
    localSession.graph.as_default()

    with tf.gfile.FastGFile('frozen_model.pb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            if node.name == inp_node:
                del node.attr['shape']

    tf.import_graph_def(graph_def, name='')

    inputData = gen_data(tf.placeholder(tf.float32, [1, 4, 5, 3], inp_node))
    outputData = localSession.run(localSession.graph.get_tensor_by_name(out_node + ':0'),
                                  feed_dict={inp_node + ':0': inputData})
    writeBlob(inputData, 'slim_batch_norm_in')
    writeBlob(outputData, 'slim_batch_norm_out')

    graph_def = TransformGraph(graph_def, [inp_node], [out_node], ['fold_constants', 'strip_unused_nodes'])
    with tf.gfile.FastGFile('slim_batch_norm_net.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())

################################################################################
# issue https://github.com/opencv/opencv/issues/13839
inp_node = 'PNet/conv3/add'
out_node = 'PNet/cls_prob'
with tf.Session(graph=tf.Graph()) as localSession:
    localSession.graph.as_default()

    with tf.gfile.FastGFile('PNet_pnet.pb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_def = TransformGraph(graph_def, [inp_node], [out_node], ['strip_unused_nodes'])

    tf.import_graph_def(graph_def, name='')

    inputData = gen_data(tf.placeholder(tf.float32, [1, 4, 5, 16], inp_node))
    outputData = localSession.run(localSession.graph.get_tensor_by_name(out_node + ':0'),
                                  feed_dict={inp_node + ':0': inputData})
    writeBlob(inputData, 'slim_softmax_v2_in')
    writeBlob(outputData, 'slim_softmax_v2_out')

    with tf.gfile.FastGFile('slim_softmax_v2_net.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())
################################################################################
inp = tf.placeholder(tf.float32, [1, 4, 6, 5, 3], 'input') # NDHWC format
conv3d = tf.layers.conv3d(inputs=inp, filters=2, kernel_size=[2, 3, 4], use_bias=True, padding='same')
save(inp, conv3d, 'conv3d')
################################################################################
inp = tf.placeholder(tf.float32, [1, 4, 6, 5, 3], 'input') # NDHWC format
maxpool3d = tf.layers.max_pooling3d(inputs=inp, pool_size=(3, 2, 3), strides=(1, 2, 1), padding='same')
save(inp, maxpool3d, 'max_pool3d')
################################################################################
inp = tf.placeholder(tf.float32, [1, 5, 4, 5, 2], 'input') # NDHWC format
avepool3d = tf.layers.average_pooling3d(inputs=inp, pool_size=(3, 3, 2), strides=(2, 1, 1), padding='valid')
save(inp, avepool3d, 'ave_pool3d')
################################################################################
# issue https://github.com/opencv/opencv/issues/13494
inp_node = 'input_image'
out_node = 'SUBPIXEL/SUBPIXEL/subpixel_image/Identity'
with tf.Session(graph=tf.Graph()) as localSession:
    localSession.graph.as_default()

    with tf.gfile.FastGFile('simple_subpixel.optimized.pb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def, name='')

    inputData = gen_data(tf.placeholder(tf.float32, [1, 1, 1, 4], inp_node))
    outputData = localSession.run(localSession.graph.get_tensor_by_name(out_node + ':0'),
                                  feed_dict={inp_node + ':0': inputData})
    writeBlob(inputData, 'subpixel_in')
    writeBlob(outputData, 'subpixel_out')

    for node in graph_def.node:
        if node.op == 'Placeholder':
            node.attr["data_format"].s = "NHWC"

    with tf.gfile.FastGFile('subpixel_net.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
conv = tf.layers.conv2d(inp, filters=4, kernel_size=[1, 1])
strided_slice = conv[:, 1:, :2, 2:3]
save(inp, strided_slice, 'strided_slice')
################################################################################
inp = tf.placeholder(tf.float32, [1, 4, 5, 3], 'input')
conv_out = tf.keras.layers.Conv2D(filters=3, kernel_size=[3, 3], padding='same')(inp)
crop2d = tf.keras.layers.Cropping2D(((1, 1), (1, 1)))(conv_out)
save(inp, crop2d, 'crop2d')
################################################################################

inp = tf.placeholder(tf.float32, [1, 2, 3, 4, 2], 'input')
bn = tf.layers.batch_normalization(inp, training=False, fused=False, name='batch_norm3d',
                                   beta_initializer=tf.random_normal_initializer(),
                                   gamma_initializer=tf.random_normal_initializer(),
                                   moving_mean_initializer=tf.random_uniform_initializer(-2, 1),
                                   moving_variance_initializer=tf.random_uniform_initializer(0.1, 2),)
save(inp, bn, 'batch_norm3d', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [1, 4, 6, 64], 'activation_8/Elu')
runModel(inp, 'batch_normalization_1/cond/FusedBatchNorm:0', 'switch_identity')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 64], 'Relu_8')
runModel(inp, 'conv2d_transpose_1:0', 'keras_deconv_same_v2')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 4, 3], 'ContentImage')
runModel(inp, 'Relu:0', 'keras_batch_norm_training')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 2, 4], 'Split')
features1 = tf.split(inp, num_or_size_splits=2, axis=3)[0]
features2 = tf.split(inp, num_or_size_splits=2, axis=3)[1]
merged = tf.concat([features1, features2], axis=3)
save(inp, merged, 'split')
################################################################################
from tensorflow.python.ops.nn_grad import _MaxPoolGrad as MaxUnPooling2D

inp = tf.placeholder(tf.float32, [1, 7, 7, 3], 'input')
pool = tf.layers.max_pooling2d(inp, pool_size=(2, 2), strides=(2, 2))
conv = tf.layers.conv2d(inputs=pool, filters=3, kernel_size=[1, 1], padding='VALID')
unpool = MaxUnPooling2D(pool.op, conv)
save(inp, unpool, 'max_pool_grad')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
conv = tf.layers.conv2d(inp, filters=5, kernel_size=[1, 1])
flatten = tf.contrib.layers.flatten(conv)
weights = tf.Variable(tf.random_normal([2*3*5, 4]), name='matmul_weights')
mm = tf.matmul(flatten, weights)
reshape = tf.reshape(mm, [-1, 1, 1, 4], 'reshaped')  # NHWC
save(inp, reshape, 'matmul_layout')
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 2, 4], 'ReduceMean')
out = tf.reduce_mean([inp, inp * 2], axis=0)
save(inp, out, 'global_pool_by_axis')
################################################################################
# issue https://github.com/opencv/opencv/issues/15141
inp_node = 'mobilenetv2_1.00_96_input'
out_node = 'mobilenetv2_1.00_96/Conv1_relu/Relu6'
with tf.Session(graph=tf.Graph()) as localSession:
    localSession.graph.as_default()

    with tf.gfile.FastGFile('normal_and_abnormal_mnet_v2_96_96_Flatten.pb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, [inp_node], [out_node], tf.float32.as_datatype_enum)

    tf.import_graph_def(graph_def, name='')

    inputData = gen_data(tf.placeholder(tf.float32, [1, 4, 5, 3], inp_node))
    outputData = localSession.run(localSession.graph.get_tensor_by_name(out_node + ':0'),
                                  feed_dict={inp_node + ':0': inputData})
    writeBlob(inputData, 'keras_learning_phase_in')
    writeBlob(outputData, 'keras_learning_phase_out')

    with tf.gfile.FastGFile('keras_learning_phase_net.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())
################################################################################
inp = tf.placeholder(tf.float32, [None, 2, 3, 4], 'input')
pool = tf.layers.average_pooling2d(inp, pool_size=(2, 3), strides=1, padding='VALID')
out = inp * pool
save(inp, out, 'channel_broadcast', optimize=False)
################################################################################
inp = tf.placeholder(tf.float32, [1, 2, 3, 4], 'input')
resized = tf.image.resize_bilinear(inp, size=[3, 5], name='resize_bilinear', align_corners=True)
conv = tf.layers.conv2d(resized, filters=4, kernel_size=[1, 1])
save(inp, conv, 'fused_resize_conv')
################################################################################
# Uncomment to save model with dynamic shapes
# inp = tf.placeholder(tf.float32, [1, None, None, 2], 'input')
inp = tf.placeholder(tf.float32, [1, 9, 6, 2], 'input')
conv = tf.layers.conv2d(inp, filters=2, kernel_size=[1, 1])
shape_input = tf.shape(inp)
hi = shape_input[1] / 3
wi = shape_input[2] / 2
input_down = tf.image.resize(conv, size=[hi,wi], method=0, name='resize_down')
save(inp, input_down, 'resize_bilinear_down')
################################################################################

# Uncomment to print the final graph.
# with tf.gfile.FastGFile('fused_batch_norm_net.pb', 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     print(graph_def)
