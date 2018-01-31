import numpy as np
import tensorflow as tf
import os
import argparse
import struct
import cv2 as cv
from prepare_for_dnn import prepare_for_dnn

parser = argparse.ArgumentParser(description='Script for OpenCV\'s DNN module '
                                             'test data generation')
parser.add_argument('-f', dest='freeze_graph_tool', required=True,
                    help='Path to freeze_graph.py tool')
parser.add_argument('-o', dest='optimizer_tool', required=True,
                    help='Path to optimize_for_inference.py tool')
parser.add_argument('-t', dest='transform_graph_tool', required=True,
                    help='Path to transform_graph tool')
args = parser.parse_args()

np.random.seed(2701)

def gen_data(placeholder):
    return np.random.standard_normal(placeholder.shape).astype(placeholder.dtype.as_numpy_dtype())

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
    else:
        # Save raw data.
        np.save(name + '.npy', data.astype(np.float32))

def runModel(inp, out, name):
    with tf.Session(graph=tf.Graph()) as localSession:
        localSession.graph.as_default()

        with tf.gfile.FastGFile(name + '_net.pb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        inputData = gen_data(inp)
        outputData = localSession.run(localSession.graph.get_tensor_by_name(out.name),
                                      feed_dict={localSession.graph.get_tensor_by_name(inp.name): inputData})
        writeBlob(inputData, name + '_in')
        writeBlob(outputData, name + '_out')

def save(inp, out, name, quantize=False):
    sess.run(tf.global_variables_initializer())

    inputData = gen_data(inp)
    outputData = sess.run(out, feed_dict={inp: inputData, isTraining: False})
    writeBlob(inputData, name + '_in')
    writeBlob(outputData, name + '_out')

    saver = tf.train.Saver()
    saver.save(sess, os.path.join('.', 'tmp.ckpt'))
    tf.train.write_graph(sess.graph.as_graph_def(), "", "graph.pb")
    prepare_for_dnn('graph.pb', 'tmp.ckpt', args.freeze_graph_tool,
                    args.optimizer_tool, args.transform_graph_tool,
                    inp.name[:inp.name.rfind(':')], out.name[:out.name.rfind(':')],
                    name + '_net.pb', inp.dtype, quantize=quantize)

    # By default, float16 weights are stored in repeated tensor's field called
    # `half_val`. It has type int32 with leading zeros for unused bytes.
    # This type is encoded by Varint that means only 7 bits are used for value
    # representation but the last one is indicated the end of encoding. This way
    # float16 might takes 1 or 2 or 3 bytes depends on value. To impove compression,
    # we replace all `half_val` values to `tensor_content` using only 2 bytes for everyone.
    with tf.gfile.FastGFile(name + '_net.pb') as f:
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
inp = tf.placeholder(tf.float32, [2, 10, 10, 3], 'input')
pad = tf.pad(inp, [[0, 0], [3, 3], [3, 3], [0, 0]])
conv = tf.layers.conv2d(inp, filters=4, kernel_size=[5, 5], strides=(2, 2),
                        bias_initializer=tf.random_normal_initializer())
save(inp, conv, 'spatial_padding')
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
# Override the graph by a frozen one.
os.rename('frozen_graph.pb', 'batch_norm_text_net.pb')
################################################################################
inp = tf.placeholder(tf.float32, [2, 4, 5], 'input')
flatten = tf.contrib.layers.flatten(inp)
save(inp, flatten, 'flatten')
################################################################################
# Generate test data for MobileNet-SSD object detection model from TensorFlow
# model zoo, http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
# 1. Download and extract an archive
# 2. Place frozen_inference_graph.pb as a ssd_mobilenet_v1_coco.pb nearby this script
with tf.gfile.FastGFile('../ssd_mobilenet_v1_coco.pb') as f:
    # Load the model
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as localSession:
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
runModel(inp, conv, 'uint8_single_conv')
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

# Uncomment to print the final graph.
# with tf.gfile.FastGFile('fused_batch_norm_net.pb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     print graph_def

def rm(f):
    if os.path.exists(f):
        os.remove(f)

rm('checkpoint')
rm('graph.pb')
rm('frozen_graph.pb')
rm('tmp.ckpt.data-00000-of-00001')
rm('tmp.ckpt.index')
rm('tmp.ckpt.meta')
