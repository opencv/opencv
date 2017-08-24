import numpy as np
import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser(description='Script for OpenCV\'s DNN module '
                                             'test data generation')
parser.add_argument('-f', dest='freeze_graph_tool', required=True,
                    help='Path to freeze_graph.py tool')
parser.add_argument('-o', dest='optimizer_tool', required=True,
                    help='Path to optimize_for_inference.py tool')
args = parser.parse_args()

np.random.seed(2701)

def gen_data(placeholder):
    return np.random.standard_normal(placeholder.shape).astype(np.float32)

tf.reset_default_graph()
tf.Graph().as_default()
tf.set_random_seed(324)
sess = tf.Session()

def writeBlob(data, name):
    # NHWC->NCHW
    np.save(name + '.npy', data.transpose(0, 3, 1, 2))

def save(inp, out, name):
    sess.run(tf.global_variables_initializer())

    inputData = gen_data(inp)
    outputData = sess.run(out, feed_dict={inp: inputData})
    writeBlob(inputData, name + '_in')
    writeBlob(outputData, name + '_out')

    saver = tf.train.Saver()
    saver.save(sess, 'tmp.ckpt')
    tf.train.write_graph(sess.graph.as_graph_def(), "", "graph.pb")
    os.system('python ' + args.freeze_graph_tool +
              ' --input_graph graph.pb '
              '--input_checkpoint tmp.ckpt '
              '--output_graph frozen_graph.pb '
              '--output_node_names ' + out.name[:out.name.rfind(':')])
    os.system('python ' + args.optimizer_tool +
              ' --input frozen_graph.pb '
              '--output ' + name + '_net.pb '
              '--frozen_graph True '
              '--input_names ' + inp.name[:inp.name.rfind(':')] +
              ' --output_names ' + out.name[:out.name.rfind(':')])

# Test cases ###################################################################
# shape: NHWC
inp = tf.placeholder(tf.float32, [1, 6, 5, 3], 'input')
conv = tf.layers.conv2d(inputs=inp, filters=3, kernel_size=[1, 1],
                        activation=tf.nn.relu,
                        bias_initializer=tf.random_normal_initializer())
save(inp, conv, 'single_conv')
################################################################################
inp = tf.placeholder(tf.float32, [3, 7, 5, 4], 'input')
conv = tf.layers.conv2d(inputs=inp, filters=5, kernel_size=[5, 3], padding='SAME',
                        use_bias=False)
activation_abs = tf.abs(conv)
save(inp, activation_abs, 'padding_same')
################################################################################
inp = tf.placeholder(tf.float32, [2, 4, 6, 5], 'input')
conv = tf.layers.conv2d(inputs=inp, filters=4, kernel_size=[3, 5], padding='VALID',
                        activation=tf.nn.elu, bias_initializer=tf.random_normal_initializer())
save(inp, conv, 'padding_valid')
################################################################################
inp = tf.placeholder(tf.float32, [3, 2, 3, 4], 'input')
conv = tf.layers.conv2d(inputs=inp, filters=4, kernel_size=[1, 1], activation=tf.nn.tanh,
                        bias_initializer=tf.random_uniform_initializer(0, 1))
conv2 = tf.layers.conv2d(inputs=inp, filters=4, kernel_size=[1, 1], activation=tf.nn.sigmoid,
                         bias_initializer=None)
eltwise_add_mul = (inp * 0.31 + 2 * conv) * conv2
save(inp, eltwise_add_mul, 'eltwise_add_mul')
################################################################################
inp = tf.placeholder(tf.float32, [1, 4, 5, 1], 'input')
conv = tf.layers.conv2d(inputs=inp, filters=4, kernel_size=[3, 1], padding='VALID')
padded = tf.pad(conv, [[0, 0], [0, 2], [0, 0], [0, 0]])
merged = tf.concat([padded, inp], axis=3)
save(inp, merged, 'pad_and_concat')
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
inp = tf.placeholder(tf.float32, [1, 6, 6, 2], 'input')
conv = tf.layers.conv2d(inputs=inp, filters=3, kernel_size=[3, 3], padding='SAME')
pool = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2)
save(inp, pool, 'max_pool_even')
################################################################################
inp = tf.placeholder(tf.float32, [1, 7, 7, 2], 'input')
conv = tf.layers.conv2d(inputs=inp, filters=3, kernel_size=[3, 3], padding='SAME')
pool = tf.layers.max_pooling2d(inputs=conv, pool_size=3, strides=2, padding='VALID')
save(inp, pool, 'max_pool_odd_valid')
################################################################################
inp = tf.placeholder(tf.float32, [1, 7, 7, 2], 'input')
conv = tf.layers.conv2d(inputs=inp, filters=3, kernel_size=[3, 3], padding='SAME')
pool = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, padding='SAME')
save(inp, pool, 'max_pool_odd_same')
################################################################################
inp = tf.placeholder(tf.float32, [1, 5, 6, 2], 'input')
deconv_weights = tf.Variable(tf.random_normal([5, 3, 4, 2]), name='deconv_weights')
deconv = tf.nn.conv2d_transpose(value=inp, filter=deconv_weights,
                                output_shape=[1, 9, 8, 4], strides=[1, 1, 1, 1],
                                padding='VALID')
deconv_bias = tf.contrib.layers.bias_add(deconv, activation_fn=tf.nn.relu,
                                         initializer=tf.random_normal_initializer())
save(inp, deconv_bias, 'deconvolution')

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
