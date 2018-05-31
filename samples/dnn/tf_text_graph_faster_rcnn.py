import argparse
import numpy as np
import tensorflow as tf

from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.tools.graph_transforms import TransformGraph
from google.protobuf import text_format

parser = argparse.ArgumentParser(description='Run this script to get a text graph of '
                                             'SSD model from TensorFlow Object Detection API. '
                                             'Then pass it with .pb file to cv::dnn::readNetFromTensorflow function.')
parser.add_argument('--input', required=True, help='Path to frozen TensorFlow graph.')
parser.add_argument('--output', required=True, help='Path to output text graph.')
parser.add_argument('--num_classes', default=90, type=int, help='Number of trained classes.')
parser.add_argument('--scales', default=[0.25, 0.5, 1.0, 2.0], type=float, nargs='+',
                    help='Hyper-parameter of grid_anchor_generator from a config file.')
parser.add_argument('--aspect_ratios', default=[0.5, 1.0, 2.0], type=float, nargs='+',
                    help='Hyper-parameter of grid_anchor_generator from a config file.')
parser.add_argument('--features_stride', default=16, type=float, nargs='+',
                    help='Hyper-parameter from a config file.')
args = parser.parse_args()

scopesToKeep = ('FirstStageFeatureExtractor', 'Conv',
                'FirstStageBoxPredictor/BoxEncodingPredictor',
                'FirstStageBoxPredictor/ClassPredictor',
                'CropAndResize',
                'MaxPool2D',
                'SecondStageFeatureExtractor',
                'SecondStageBoxPredictor',
                'image_tensor')

scopesToIgnore = ('FirstStageFeatureExtractor/Assert',
                  'FirstStageFeatureExtractor/Shape',
                  'FirstStageFeatureExtractor/strided_slice',
                  'FirstStageFeatureExtractor/GreaterEqual',
                  'FirstStageFeatureExtractor/LogicalAnd')

unusedAttrs = ['T', 'Tshape', 'N', 'Tidx', 'Tdim', 'use_cudnn_on_gpu',
               'Index', 'Tperm', 'is_training', 'Tpaddings']

# Read the graph.
with tf.gfile.FastGFile(args.input, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Removes Identity nodes
def removeIdentity():
    identities = {}
    for node in graph_def.node:
        if node.op == 'Identity':
            identities[node.name] = node.input[0]
            graph_def.node.remove(node)

    for node in graph_def.node:
        for i in range(len(node.input)):
            if node.input[i] in identities:
                node.input[i] = identities[node.input[i]]

removeIdentity()

removedNodes = []

for i in reversed(range(len(graph_def.node))):
    op = graph_def.node[i].op
    name = graph_def.node[i].name

    if op == 'Const' or name.startswith(scopesToIgnore) or not name.startswith(scopesToKeep):
        if op != 'Const':
            removedNodes.append(name)

        del graph_def.node[i]
    else:
        for attr in unusedAttrs:
            if attr in graph_def.node[i].attr:
                del graph_def.node[i].attr[attr]

# Remove references to removed nodes except Const nodes.
for node in graph_def.node:
    for i in reversed(range(len(node.input))):
        if node.input[i] in removedNodes:
            del node.input[i]


# Connect input node to the first layer
assert(graph_def.node[0].op == 'Placeholder')
graph_def.node[1].input.insert(0, graph_def.node[0].name)

# Temporarily remove top nodes.
topNodes = []
while True:
    node = graph_def.node.pop()
    topNodes.append(node)
    if node.op == 'CropAndResize':
        break

def tensorMsg(values):
    if all([isinstance(v, float) for v in values]):
        dtype = 'DT_FLOAT'
        field = 'float_val'
    elif all([isinstance(v, int) for v in values]):
        dtype = 'DT_INT32'
        field = 'int_val'
    else:
        raise Exception('Wrong values types')

    msg = 'tensor { dtype: ' + dtype + ' tensor_shape { dim { size: %d } }' % len(values)
    for value in values:
        msg += '%s: %s ' % (field, str(value))
    return msg + '}'

def addSlice(inp, out, begins, sizes):
    beginsNode = NodeDef()
    beginsNode.name = out + '/begins'
    beginsNode.op = 'Const'
    text_format.Merge(tensorMsg(begins), beginsNode.attr["value"])
    graph_def.node.extend([beginsNode])

    sizesNode = NodeDef()
    sizesNode.name = out + '/sizes'
    sizesNode.op = 'Const'
    text_format.Merge(tensorMsg(sizes), sizesNode.attr["value"])
    graph_def.node.extend([sizesNode])

    sliced = NodeDef()
    sliced.name = out
    sliced.op = 'Slice'
    sliced.input.append(inp)
    sliced.input.append(beginsNode.name)
    sliced.input.append(sizesNode.name)
    graph_def.node.extend([sliced])

def addReshape(inp, out, shape):
    shapeNode = NodeDef()
    shapeNode.name = out + '/shape'
    shapeNode.op = 'Const'
    text_format.Merge(tensorMsg(shape), shapeNode.attr["value"])
    graph_def.node.extend([shapeNode])

    reshape = NodeDef()
    reshape.name = out
    reshape.op = 'Reshape'
    reshape.input.append(inp)
    reshape.input.append(shapeNode.name)
    graph_def.node.extend([reshape])

def addSoftMax(inp, out):
    softmax = NodeDef()
    softmax.name = out
    softmax.op = 'Softmax'
    text_format.Merge('i: -1', softmax.attr['axis'])
    softmax.input.append(inp)
    graph_def.node.extend([softmax])

addReshape('FirstStageBoxPredictor/ClassPredictor/BiasAdd',
           'FirstStageBoxPredictor/ClassPredictor/reshape_1', [0, -1, 2])

addSoftMax('FirstStageBoxPredictor/ClassPredictor/reshape_1',
           'FirstStageBoxPredictor/ClassPredictor/softmax')  # Compare with Reshape_4

flatten = NodeDef()
flatten.name = 'FirstStageBoxPredictor/BoxEncodingPredictor/flatten'  # Compare with FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd
flatten.op = 'Flatten'
flatten.input.append('FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd')
graph_def.node.extend([flatten])

proposals = NodeDef()
proposals.name = 'proposals'  # Compare with ClipToWindow/Gather/Gather (NOTE: normalized)
proposals.op = 'PriorBox'
proposals.input.append('FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd')
proposals.input.append(graph_def.node[0].name)  # image_tensor

text_format.Merge('b: false', proposals.attr["flip"])
text_format.Merge('b: true', proposals.attr["clip"])
text_format.Merge('f: %f' % args.features_stride, proposals.attr["step"])
text_format.Merge('f: 0.0', proposals.attr["offset"])
text_format.Merge(tensorMsg([0.1, 0.1, 0.2, 0.2]), proposals.attr["variance"])

widths = []
heights = []
for a in args.aspect_ratios:
    for s in args.scales:
        ar = np.sqrt(a)
        heights.append((args.features_stride**2) * s / ar)
        widths.append((args.features_stride**2) * s * ar)

text_format.Merge(tensorMsg(widths), proposals.attr["width"])
text_format.Merge(tensorMsg(heights), proposals.attr["height"])

graph_def.node.extend([proposals])

# Compare with Reshape_5
detectionOut = NodeDef()
detectionOut.name = 'detection_out'
detectionOut.op = 'DetectionOutput'

detectionOut.input.append('FirstStageBoxPredictor/BoxEncodingPredictor/flatten')
detectionOut.input.append('FirstStageBoxPredictor/ClassPredictor/softmax')
detectionOut.input.append('proposals')

text_format.Merge('i: 2', detectionOut.attr['num_classes'])
text_format.Merge('b: true', detectionOut.attr['share_location'])
text_format.Merge('i: 0', detectionOut.attr['background_label_id'])
text_format.Merge('f: 0.7', detectionOut.attr['nms_threshold'])
text_format.Merge('i: 6000', detectionOut.attr['top_k'])
text_format.Merge('s: "CENTER_SIZE"', detectionOut.attr['code_type'])
text_format.Merge('i: 100', detectionOut.attr['keep_top_k'])
text_format.Merge('b: true', detectionOut.attr['clip'])
text_format.Merge('b: true', detectionOut.attr['loc_pred_transposed'])

graph_def.node.extend([detectionOut])

# Save as text.
for node in reversed(topNodes):
    graph_def.node.extend([node])

addSoftMax('SecondStageBoxPredictor/Reshape_1', 'SecondStageBoxPredictor/Reshape_1/softmax')

addSlice('SecondStageBoxPredictor/Reshape_1/softmax',
         'SecondStageBoxPredictor/Reshape_1/slice',
         [0, 0, 1], [-1, -1, -1])

addReshape('SecondStageBoxPredictor/Reshape_1/slice',
          'SecondStageBoxPredictor/Reshape_1/Reshape', [1, -1])

# Replace Flatten subgraph onto a single node.
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'CropAndResize':
        graph_def.node[i].input.insert(1, 'detection_out')

    if graph_def.node[i].name == 'SecondStageBoxPredictor/Reshape':
        shapeNode = NodeDef()
        shapeNode.name = 'SecondStageBoxPredictor/Reshape/shape2'
        shapeNode.op = 'Const'
        text_format.Merge(tensorMsg([1, -1, 4]), shapeNode.attr["value"])
        graph_def.node.extend([shapeNode])

        graph_def.node[i].input.pop()
        graph_def.node[i].input.append(shapeNode.name)

    if graph_def.node[i].name in ['SecondStageBoxPredictor/Flatten/flatten/Shape',
                                  'SecondStageBoxPredictor/Flatten/flatten/strided_slice',
                                  'SecondStageBoxPredictor/Flatten/flatten/Reshape/shape']:
        del graph_def.node[i]

for node in graph_def.node:
    if node.name == 'SecondStageBoxPredictor/Flatten/flatten/Reshape':
        node.op = 'Flatten'
        node.input.pop()
        break

################################################################################
### Postprocessing
################################################################################
addSlice('detection_out', 'detection_out/slice', [0, 0, 0, 3], [-1, -1, -1, 4])

variance = NodeDef()
variance.name = 'proposals/variance'
variance.op = 'Const'
text_format.Merge(tensorMsg([0.1, 0.1, 0.2, 0.2]), variance.attr["value"])
graph_def.node.extend([variance])

varianceEncoder = NodeDef()
varianceEncoder.name = 'variance_encoded'
varianceEncoder.op = 'Mul'
varianceEncoder.input.append('SecondStageBoxPredictor/Reshape')
varianceEncoder.input.append(variance.name)
text_format.Merge('i: 2', varianceEncoder.attr["axis"])
graph_def.node.extend([varianceEncoder])

addReshape('detection_out/slice', 'detection_out/slice/reshape', [1, 1, -1])

detectionOut = NodeDef()
detectionOut.name = 'detection_out_final'
detectionOut.op = 'DetectionOutput'

detectionOut.input.append('variance_encoded')
detectionOut.input.append('SecondStageBoxPredictor/Reshape_1/Reshape')
detectionOut.input.append('detection_out/slice/reshape')

text_format.Merge('i: %d' % args.num_classes, detectionOut.attr['num_classes'])
text_format.Merge('b: false', detectionOut.attr['share_location'])
text_format.Merge('i: %d' % (args.num_classes + 1), detectionOut.attr['background_label_id'])
text_format.Merge('f: 0.6', detectionOut.attr['nms_threshold'])
text_format.Merge('s: "CENTER_SIZE"', detectionOut.attr['code_type'])
text_format.Merge('i: 100', detectionOut.attr['keep_top_k'])
text_format.Merge('b: true', detectionOut.attr['loc_pred_transposed'])
text_format.Merge('b: true', detectionOut.attr['clip'])
text_format.Merge('b: true', detectionOut.attr['variance_encoded_in_target'])
graph_def.node.extend([detectionOut])

tf.train.write_graph(graph_def, "", args.output, as_text=True)
