from __future__ import print_function
import sys
import argparse
import cv2 as cv
import tensorflow as tf
import numpy as np
import struct

if sys.version_info > (3,):
    long = int

from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.core.framework.node_def_pb2 import NodeDef
from google.protobuf import text_format

parser = argparse.ArgumentParser(description="Use this script to create TensorFlow graph "
                                             "with weights from OpenCV's face detection network. "
                                             "Only backbone part of SSD model is converted this way. "
                                             "Look for .pbtxt configuration file at "
                                             "https://github.com/opencv/opencv_extra/tree/4.x/testdata/dnn/opencv_face_detector.pbtxt")
parser.add_argument('--model', help='Path to .caffemodel weights', required=True)
parser.add_argument('--proto', help='Path to .prototxt Caffe model definition', required=True)
parser.add_argument('--pb', help='Path to output .pb TensorFlow model', required=True)
parser.add_argument('--pbtxt', help='Path to output .pbxt TensorFlow graph', required=True)
parser.add_argument('--quantize', help='Quantize weights to uint8', action='store_true')
parser.add_argument('--fp16', help='Convert weights to half precision floats', action='store_true')
args = parser.parse_args()

assert(not args.quantize or not args.fp16)

dtype = tf.float16 if args.fp16 else tf.float32

################################################################################
cvNet = cv.dnn.readNetFromCaffe(args.proto, args.model)

def dnnLayer(name):
    return cvNet.getLayer(long(cvNet.getLayerId(name)))

def scale(x, name):
    with tf.variable_scope(name):
        layer = dnnLayer(name)
        w = tf.Variable(layer.blobs[0].flatten(), dtype=dtype, name='mul')
        if len(layer.blobs) > 1:
            b = tf.Variable(layer.blobs[1].flatten(), dtype=dtype, name='add')
            return tf.nn.bias_add(tf.multiply(x, w), b)
        else:
            return tf.multiply(x, w, name)

def conv(x, name, stride=1, pad='SAME', dilation=1, activ=None):
    with tf.variable_scope(name):
        layer = dnnLayer(name)
        w = tf.Variable(layer.blobs[0].transpose(2, 3, 1, 0), dtype=dtype, name='weights')
        if dilation == 1:
            conv = tf.nn.conv2d(x, filter=w, strides=(1, stride, stride, 1), padding=pad)
        else:
            assert(stride == 1)
            conv = tf.nn.atrous_conv2d(x, w, rate=dilation, padding=pad)

        if len(layer.blobs) > 1:
            b = tf.Variable(layer.blobs[1].flatten(), dtype=dtype, name='bias')
            conv = tf.nn.bias_add(conv, b)
        return activ(conv) if activ else conv

def batch_norm(x, name):
    with tf.variable_scope(name):
        # Unfortunately, TensorFlow's batch normalization layer doesn't work with fp16 input.
        # Here we do a cast to fp32 but remove it in the frozen graph.
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)

        layer = dnnLayer(name)
        assert(len(layer.blobs) >= 3)

        mean = layer.blobs[0].flatten()
        std = layer.blobs[1].flatten()
        scale = layer.blobs[2].flatten()

        eps = 1e-5
        hasBias = len(layer.blobs) > 3
        hasWeights = scale.shape != (1,)

        if not hasWeights and not hasBias:
            mean /= scale[0]
            std /= scale[0]

        mean = tf.Variable(mean, dtype=tf.float32, name='mean')
        std = tf.Variable(std, dtype=tf.float32, name='std')
        gamma = tf.Variable(scale if hasWeights else np.ones(mean.shape), dtype=tf.float32, name='gamma')
        beta = tf.Variable(layer.blobs[3].flatten() if hasBias else np.zeros(mean.shape), dtype=tf.float32, name='beta')
        bn = tf.nn.fused_batch_norm(x, gamma, beta, mean, std, eps,
                                    is_training=False)[0]
        if bn.dtype != dtype:
            bn = tf.cast(bn, dtype)
        return bn

def l2norm(x, name):
    with tf.variable_scope(name):
        layer = dnnLayer(name)
        w = tf.Variable(layer.blobs[0].flatten(), dtype=dtype, name='mul')
        return tf.nn.l2_normalize(x, 3, epsilon=1e-10) * w

### Graph definition ###########################################################
inp = tf.placeholder(dtype, [1, 300, 300, 3], 'data')
data_bn = batch_norm(inp, 'data_bn')
data_scale = scale(data_bn, 'data_scale')

# Instead of tf.pad we use tf.space_to_batch_nd layers which override convolution's padding strategy to explicit numbers
# data_scale = tf.pad(data_scale, [[0, 0], [3, 3], [3, 3], [0, 0]])
data_scale = tf.space_to_batch_nd(data_scale, [1, 1], [[3, 3], [3, 3]], name='Pad')
conv1_h = conv(data_scale, stride=2, pad='VALID', name='conv1_h')

conv1_bn_h = batch_norm(conv1_h, 'conv1_bn_h')
conv1_scale_h = scale(conv1_bn_h, 'conv1_scale_h')
conv1_relu = tf.nn.relu(conv1_scale_h)
conv1_pool = tf.layers.max_pooling2d(conv1_relu, pool_size=(3, 3), strides=(2, 2),
                                     padding='SAME', name='conv1_pool')

layer_64_1_conv1_h = conv(conv1_pool, 'layer_64_1_conv1_h')
layer_64_1_bn2_h = batch_norm(layer_64_1_conv1_h, 'layer_64_1_bn2_h')
layer_64_1_scale2_h = scale(layer_64_1_bn2_h, 'layer_64_1_scale2_h')
layer_64_1_relu2 = tf.nn.relu(layer_64_1_scale2_h)
layer_64_1_conv2_h = conv(layer_64_1_relu2, 'layer_64_1_conv2_h')
layer_64_1_sum = layer_64_1_conv2_h + conv1_pool

layer_128_1_bn1_h = batch_norm(layer_64_1_sum, 'layer_128_1_bn1_h')
layer_128_1_scale1_h = scale(layer_128_1_bn1_h, 'layer_128_1_scale1_h')
layer_128_1_relu1 = tf.nn.relu(layer_128_1_scale1_h)
layer_128_1_conv1_h = conv(layer_128_1_relu1, stride=2, name='layer_128_1_conv1_h')
layer_128_1_bn2 = batch_norm(layer_128_1_conv1_h, 'layer_128_1_bn2')
layer_128_1_scale2 = scale(layer_128_1_bn2, 'layer_128_1_scale2')
layer_128_1_relu2 = tf.nn.relu(layer_128_1_scale2)
layer_128_1_conv2 = conv(layer_128_1_relu2, 'layer_128_1_conv2')
layer_128_1_conv_expand_h = conv(layer_128_1_relu1, stride=2, name='layer_128_1_conv_expand_h')
layer_128_1_sum = layer_128_1_conv2 + layer_128_1_conv_expand_h

layer_256_1_bn1 = batch_norm(layer_128_1_sum, 'layer_256_1_bn1')
layer_256_1_scale1 = scale(layer_256_1_bn1, 'layer_256_1_scale1')
layer_256_1_relu1 = tf.nn.relu(layer_256_1_scale1)

# layer_256_1_conv1 = tf.pad(layer_256_1_relu1, [[0, 0], [1, 1], [1, 1], [0, 0]])
layer_256_1_conv1 = tf.space_to_batch_nd(layer_256_1_relu1, [1, 1], [[1, 1], [1, 1]], name='Pad_1')
layer_256_1_conv1 = conv(layer_256_1_conv1, stride=2, pad='VALID', name='layer_256_1_conv1')

layer_256_1_bn2 = batch_norm(layer_256_1_conv1, 'layer_256_1_bn2')
layer_256_1_scale2 = scale(layer_256_1_bn2, 'layer_256_1_scale2')
layer_256_1_relu2 = tf.nn.relu(layer_256_1_scale2)
layer_256_1_conv2 = conv(layer_256_1_relu2, 'layer_256_1_conv2')
layer_256_1_conv_expand = conv(layer_256_1_relu1, stride=2, name='layer_256_1_conv_expand')
layer_256_1_sum = layer_256_1_conv2 + layer_256_1_conv_expand

layer_512_1_bn1 = batch_norm(layer_256_1_sum, 'layer_512_1_bn1')
layer_512_1_scale1 = scale(layer_512_1_bn1, 'layer_512_1_scale1')
layer_512_1_relu1 = tf.nn.relu(layer_512_1_scale1)
layer_512_1_conv1_h = conv(layer_512_1_relu1, 'layer_512_1_conv1_h')
layer_512_1_bn2_h = batch_norm(layer_512_1_conv1_h, 'layer_512_1_bn2_h')
layer_512_1_scale2_h = scale(layer_512_1_bn2_h, 'layer_512_1_scale2_h')
layer_512_1_relu2 = tf.nn.relu(layer_512_1_scale2_h)
layer_512_1_conv2_h = conv(layer_512_1_relu2, dilation=2, name='layer_512_1_conv2_h')
layer_512_1_conv_expand_h = conv(layer_512_1_relu1, 'layer_512_1_conv_expand_h')
layer_512_1_sum = layer_512_1_conv2_h + layer_512_1_conv_expand_h

last_bn_h = batch_norm(layer_512_1_sum, 'last_bn_h')
last_scale_h = scale(last_bn_h, 'last_scale_h')
fc7 = tf.nn.relu(last_scale_h, name='last_relu')

conv6_1_h = conv(fc7, 'conv6_1_h', activ=tf.nn.relu)
conv6_2_h = conv(conv6_1_h, stride=2, name='conv6_2_h', activ=tf.nn.relu)
conv7_1_h = conv(conv6_2_h, 'conv7_1_h', activ=tf.nn.relu)

# conv7_2_h = tf.pad(conv7_1_h, [[0, 0], [1, 1], [1, 1], [0, 0]])
conv7_2_h = tf.space_to_batch_nd(conv7_1_h, [1, 1], [[1, 1], [1, 1]], name='Pad_2')
conv7_2_h = conv(conv7_2_h, stride=2, pad='VALID', name='conv7_2_h', activ=tf.nn.relu)

conv8_1_h = conv(conv7_2_h, pad='SAME', name='conv8_1_h', activ=tf.nn.relu)
conv8_2_h = conv(conv8_1_h, pad='VALID', name='conv8_2_h', activ=tf.nn.relu)
conv9_1_h = conv(conv8_2_h, 'conv9_1_h', activ=tf.nn.relu)
conv9_2_h = conv(conv9_1_h, pad='VALID', name='conv9_2_h', activ=tf.nn.relu)

conv4_3_norm = l2norm(layer_256_1_relu1, 'conv4_3_norm')

### Locations and confidences ##################################################
locations = []
confidences = []
flattenLayersNames = []  # Collect all reshape layers names that should be replaced to flattens.
for top, suffix in zip([locations, confidences], ['_mbox_loc', '_mbox_conf']):
    for bottom, name in zip([conv4_3_norm, fc7, conv6_2_h, conv7_2_h, conv8_2_h, conv9_2_h],
                            ['conv4_3_norm', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']):
        name += suffix
        flat = tf.layers.flatten(conv(bottom, name))
        flattenLayersNames.append(flat.name[:flat.name.find(':')])
        top.append(flat)

mbox_loc = tf.concat(locations, axis=-1, name='mbox_loc')
mbox_conf = tf.concat(confidences, axis=-1, name='mbox_conf')

total = int(np.prod(mbox_conf.shape[1:]))
mbox_conf_reshape = tf.reshape(mbox_conf, [-1, 2], name='mbox_conf_reshape')
mbox_conf_softmax = tf.nn.softmax(mbox_conf_reshape, name='mbox_conf_softmax')
mbox_conf_flatten = tf.reshape(mbox_conf_softmax, [-1, total], name='mbox_conf_flatten')
flattenLayersNames.append('mbox_conf_flatten')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ### Check correctness ######################################################
    out_nodes = ['mbox_loc', 'mbox_conf_flatten']
    inp_nodes = [inp.name[:inp.name.find(':')]]

    np.random.seed(2701)
    inputData = np.random.standard_normal([1, 3, 300, 300]).astype(np.float32)

    cvNet.setInput(inputData)
    cvNet.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    outDNN = cvNet.forward(out_nodes)

    outTF = sess.run([mbox_loc, mbox_conf_flatten], feed_dict={inp: inputData.transpose(0, 2, 3, 1)})
    print('Max diff @ locations:  %e' % np.max(np.abs(outDNN[0] - outTF[0])))
    print('Max diff @ confidence: %e' % np.max(np.abs(outDNN[1] - outTF[1])))

    # Save a graph
    graph_def = sess.graph.as_graph_def()

    # Freeze graph. Replaces variables to constants.
    graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, out_nodes)
    # Optimize graph. Removes training-only ops, unused nodes.
    graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, inp_nodes, out_nodes, dtype.as_datatype_enum)
    # Fuse constant operations.
    transforms = ["fold_constants(ignore_errors=True)"]
    if args.quantize:
        transforms += ["quantize_weights(minimum_size=0)"]
    transforms += ["sort_by_execution_order"]
    graph_def = TransformGraph(graph_def, inp_nodes, out_nodes, transforms)

    # By default, float16 weights are stored in repeated tensor's field called
    # `half_val`. It has type int32 with leading zeros for unused bytes.
    # This type is encoded by Variant that means only 7 bits are used for value
    # representation but the last one is indicated the end of encoding. This way
    # float16 might takes 1 or 2 or 3 bytes depends on value. To improve compression,
    # we replace all `half_val` values to `tensor_content` using only 2 bytes for everyone.
    for node in graph_def.node:
        if 'value' in node.attr:
            halfs = node.attr["value"].tensor.half_val
            if not node.attr["value"].tensor.tensor_content and halfs:
                node.attr["value"].tensor.tensor_content = struct.pack('H' * len(halfs), *halfs)
                node.attr["value"].tensor.ClearField('half_val')

    # Serialize
    with tf.gfile.FastGFile(args.pb, 'wb') as f:
            f.write(graph_def.SerializeToString())


################################################################################
# Write a text graph representation
################################################################################
def tensorMsg(values):
    msg = 'tensor { dtype: DT_FLOAT tensor_shape { dim { size: %d } }' % len(values)
    for value in values:
        msg += 'float_val: %f ' % value
    return msg + '}'

# Remove Const nodes and unused attributes.
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op in ['Const', 'Dequantize']:
        del graph_def.node[i]
    for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
                 'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
                 'Tpaddings', 'Tblock_shape', 'Tcrops']:
        if attr in graph_def.node[i].attr:
            del graph_def.node[i].attr[attr]

# Append prior box generators
min_sizes = [30, 60, 111, 162, 213, 264]
max_sizes = [60, 111, 162, 213, 264, 315]
steps = [8, 16, 32, 64, 100, 300]
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
layers = [conv4_3_norm, fc7, conv6_2_h, conv7_2_h, conv8_2_h, conv9_2_h]
for i in range(6):
    priorBox = NodeDef()
    priorBox.name = 'PriorBox_%d' % i
    priorBox.op = 'PriorBox'
    priorBox.input.append(layers[i].name[:layers[i].name.find(':')])
    priorBox.input.append(inp_nodes[0])  # data

    text_format.Merge('i: %d' % min_sizes[i], priorBox.attr["min_size"])
    text_format.Merge('i: %d' % max_sizes[i], priorBox.attr["max_size"])
    text_format.Merge('b: true', priorBox.attr["flip"])
    text_format.Merge('b: false', priorBox.attr["clip"])
    text_format.Merge(tensorMsg(aspect_ratios[i]), priorBox.attr["aspect_ratio"])
    text_format.Merge(tensorMsg([0.1, 0.1, 0.2, 0.2]), priorBox.attr["variance"])
    text_format.Merge('f: %f' % steps[i], priorBox.attr["step"])
    text_format.Merge('f: 0.5', priorBox.attr["offset"])
    graph_def.node.extend([priorBox])

# Concatenate prior boxes
concat = NodeDef()
concat.name = 'mbox_priorbox'
concat.op = 'ConcatV2'
for i in range(6):
    concat.input.append('PriorBox_%d' % i)
concat.input.append('mbox_loc/axis')
graph_def.node.extend([concat])

# DetectionOutput layer
detectionOut = NodeDef()
detectionOut.name = 'detection_out'
detectionOut.op = 'DetectionOutput'

detectionOut.input.append('mbox_loc')
detectionOut.input.append('mbox_conf_flatten')
detectionOut.input.append('mbox_priorbox')

text_format.Merge('i: 2', detectionOut.attr['num_classes'])
text_format.Merge('b: true', detectionOut.attr['share_location'])
text_format.Merge('i: 0', detectionOut.attr['background_label_id'])
text_format.Merge('f: 0.45', detectionOut.attr['nms_threshold'])
text_format.Merge('i: 400', detectionOut.attr['top_k'])
text_format.Merge('s: "CENTER_SIZE"', detectionOut.attr['code_type'])
text_format.Merge('i: 200', detectionOut.attr['keep_top_k'])
text_format.Merge('f: 0.01', detectionOut.attr['confidence_threshold'])

graph_def.node.extend([detectionOut])

# Replace L2Normalization subgraph onto a single node.
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].name in ['conv4_3_norm/l2_normalize/Square',
                                  'conv4_3_norm/l2_normalize/Sum',
                                  'conv4_3_norm/l2_normalize/Maximum',
                                  'conv4_3_norm/l2_normalize/Rsqrt']:
        del graph_def.node[i]
for node in graph_def.node:
    if node.name == 'conv4_3_norm/l2_normalize':
        node.op = 'L2Normalize'
        node.input.pop()
        node.input.pop()
        node.input.append(layer_256_1_relu1.name)
        node.input.append('conv4_3_norm/l2_normalize/Sum/reduction_indices')
        break

softmaxShape = NodeDef()
softmaxShape.name = 'reshape_before_softmax'
softmaxShape.op = 'Const'
text_format.Merge(
'tensor {'
'  dtype: DT_INT32'
'  tensor_shape { dim { size: 3 } }'
'  int_val: 0'
'  int_val: -1'
'  int_val: 2'
'}', softmaxShape.attr["value"])
graph_def.node.extend([softmaxShape])

for node in graph_def.node:
    if node.name == 'mbox_conf_reshape':
        node.input[1] = softmaxShape.name
    elif node.name == 'mbox_conf_softmax':
        text_format.Merge('i: 2', node.attr['axis'])
    elif node.name in flattenLayersNames:
        node.op = 'Flatten'
        inpName = node.input[0]
        node.input.pop()
        node.input.pop()
        node.input.append(inpName)

tf.train.write_graph(graph_def, "", args.pbtxt, as_text=True)
