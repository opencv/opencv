import numpy as np
import tensorflow as tf
import os
import argparse
import struct
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

def save(inp, out, name):
    sess.run(tf.global_variables_initializer())

    inputData = gen_data(inp)
    outputData = sess.run(out, feed_dict={inp: inputData, isTraining: False})
    writeBlob(inputData, name + '_in')
    writeBlob(outputData, name + '_out')

    saver = tf.train.Saver()
    saver.save(sess, 'tmp.ckpt')
    tf.train.write_graph(sess.graph.as_graph_def(), "", "graph.pb")
    prepare_for_dnn('graph.pb', 'tmp.ckpt', args.freeze_graph_tool,
                    args.optimizer_tool, args.transform_graph_tool,
                    inp.name[:inp.name.rfind(':')], out.name[:out.name.rfind(':')],
                    name + '_net.pb', inp.dtype)

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

# Uncomment to print the final graph.
# with tf.gfile.FastGFile('fused_batch_norm_net.pb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     print graph_def

os.remove('checkpoint')
os.remove('graph.pb')
os.remove('frozen_graph.pb')
os.remove('tmp.ckpt.data-00000-of-00001')
os.remove('tmp.ckpt.index')
os.remove('tmp.ckpt.meta')
