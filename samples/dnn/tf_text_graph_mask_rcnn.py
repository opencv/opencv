import argparse
import numpy as np
import tensorflow as tf

from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.tools.graph_transforms import TransformGraph
from google.protobuf import text_format

from tf_text_graph_common import *

parser = argparse.ArgumentParser(description='Run this script to get a text graph of '
                                             'Mask-RCNN model from TensorFlow Object Detection API. '
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
                'Preprocessor/sub',
                'Preprocessor/mul',
                'image_tensor')

scopesToIgnore = ('FirstStageFeatureExtractor/Assert',
                  'FirstStageFeatureExtractor/Shape',
                  'FirstStageFeatureExtractor/strided_slice',
                  'FirstStageFeatureExtractor/GreaterEqual',
                  'FirstStageFeatureExtractor/LogicalAnd')


# Read the graph.
with tf.gfile.FastGFile(args.input, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

removeIdentity(graph_def)

def to_remove(name, op):
    return name.startswith(scopesToIgnore) or not name.startswith(scopesToKeep)

removeUnusedNodesAndAttrs(to_remove, graph_def)


# Connect input node to the first layer
assert(graph_def.node[0].op == 'Placeholder')
graph_def.node[1].input.insert(0, graph_def.node[0].name)

# Temporarily remove top nodes.
topNodes = []
numCropAndResize = 0
while True:
    node = graph_def.node.pop()
    topNodes.append(node)
    if node.op == 'CropAndResize':
        numCropAndResize += 1
        if numCropAndResize == 2:
            break

addReshape('FirstStageBoxPredictor/ClassPredictor/BiasAdd',
           'FirstStageBoxPredictor/ClassPredictor/reshape_1', [0, -1, 2], graph_def)

addSoftMax('FirstStageBoxPredictor/ClassPredictor/reshape_1',
           'FirstStageBoxPredictor/ClassPredictor/softmax', graph_def)  # Compare with Reshape_4

addFlatten('FirstStageBoxPredictor/ClassPredictor/softmax',
           'FirstStageBoxPredictor/ClassPredictor/softmax/flatten', graph_def)

# Compare with FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd
addFlatten('FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd',
           'FirstStageBoxPredictor/BoxEncodingPredictor/flatten', graph_def)

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
detectionOut.input.append('FirstStageBoxPredictor/ClassPredictor/softmax/flatten')
detectionOut.input.append('proposals')

text_format.Merge('i: 2', detectionOut.attr['num_classes'])
text_format.Merge('b: true', detectionOut.attr['share_location'])
text_format.Merge('i: 0', detectionOut.attr['background_label_id'])
text_format.Merge('f: 0.7', detectionOut.attr['nms_threshold'])
text_format.Merge('i: 6000', detectionOut.attr['top_k'])
text_format.Merge('s: "CENTER_SIZE"', detectionOut.attr['code_type'])
text_format.Merge('i: 100', detectionOut.attr['keep_top_k'])
text_format.Merge('b: true', detectionOut.attr['clip'])

graph_def.node.extend([detectionOut])

# Save as text.
for node in reversed(topNodes):
    if node.op != 'CropAndResize':
        graph_def.node.extend([node])
        topNodes.pop()
    else:
        if numCropAndResize == 1:
            break
        else:
            graph_def.node.extend([node])
            topNodes.pop()
            numCropAndResize -= 1

addSoftMax('SecondStageBoxPredictor/Reshape_1', 'SecondStageBoxPredictor/Reshape_1/softmax', graph_def)

addSlice('SecondStageBoxPredictor/Reshape_1/softmax',
         'SecondStageBoxPredictor/Reshape_1/slice',
         [0, 0, 1], [-1, -1, -1], graph_def)

addReshape('SecondStageBoxPredictor/Reshape_1/slice',
          'SecondStageBoxPredictor/Reshape_1/Reshape', [1, -1], graph_def)

# Replace Flatten subgraph onto a single node.
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'CropAndResize':
        graph_def.node[i].input.insert(1, 'detection_out')

    if graph_def.node[i].name == 'SecondStageBoxPredictor/Reshape':
        addConstNode('SecondStageBoxPredictor/Reshape/shape2', [1, -1, 4], graph_def)

        graph_def.node[i].input.pop()
        graph_def.node[i].input.append('SecondStageBoxPredictor/Reshape/shape2')

    if graph_def.node[i].name in ['SecondStageBoxPredictor/Flatten/flatten/Shape',
                                  'SecondStageBoxPredictor/Flatten/flatten/strided_slice',
                                  'SecondStageBoxPredictor/Flatten/flatten/Reshape/shape']:
        del graph_def.node[i]

for node in graph_def.node:
    if node.name == 'SecondStageBoxPredictor/Flatten/flatten/Reshape':
        node.op = 'Flatten'
        node.input.pop()

    if node.name in ['FirstStageBoxPredictor/BoxEncodingPredictor/Conv2D',
                     'SecondStageBoxPredictor/BoxEncodingPredictor/MatMul']:
        text_format.Merge('b: true', node.attr["loc_pred_transposed"])

################################################################################
### Postprocessing
################################################################################
addSlice('detection_out', 'detection_out/slice', [0, 0, 0, 3], [-1, -1, -1, 4], graph_def)

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

addReshape('detection_out/slice', 'detection_out/slice/reshape', [1, 1, -1], graph_def)
addFlatten('variance_encoded', 'variance_encoded/flatten', graph_def)

detectionOut = NodeDef()
detectionOut.name = 'detection_out_final'
detectionOut.op = 'DetectionOutput'

detectionOut.input.append('variance_encoded/flatten')
detectionOut.input.append('SecondStageBoxPredictor/Reshape_1/Reshape')
detectionOut.input.append('detection_out/slice/reshape')

text_format.Merge('i: %d' % args.num_classes, detectionOut.attr['num_classes'])
text_format.Merge('b: false', detectionOut.attr['share_location'])
text_format.Merge('i: %d' % (args.num_classes + 1), detectionOut.attr['background_label_id'])
text_format.Merge('f: 0.6', detectionOut.attr['nms_threshold'])
text_format.Merge('s: "CENTER_SIZE"', detectionOut.attr['code_type'])
text_format.Merge('i: 100', detectionOut.attr['keep_top_k'])
text_format.Merge('b: true', detectionOut.attr['clip'])
text_format.Merge('b: true', detectionOut.attr['variance_encoded_in_target'])
text_format.Merge('f: 0.3', detectionOut.attr['confidence_threshold'])
text_format.Merge('b: false', detectionOut.attr['group_by_classes'])
graph_def.node.extend([detectionOut])

for node in reversed(topNodes):
    graph_def.node.extend([node])

for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'CropAndResize':
        graph_def.node[i].input.insert(1, 'detection_out_final')
        break

graph_def.node[-1].name = 'detection_masks'
graph_def.node[-1].op = 'Sigmoid'
graph_def.node[-1].input.pop()

tf.train.write_graph(graph_def, "", args.output, as_text=True)
