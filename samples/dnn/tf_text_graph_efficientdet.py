# This file is a part of OpenCV project.
# It is a subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
#
# Copyright (C) 2020, Intel Corporation, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# Use this script to get the text graph representation (.pbtxt) of EfficientDet
# deep learning network trained in https://github.com/google/automl.
# Then you can import it with a binary frozen graph (.pb) using readNetFromTensorflow() function.
# See details and examples on the following wiki page: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
import argparse
import re
from math import sqrt
from tf_text_graph_common import *


class AnchorGenerator:
    def __init__(self, min_level, aspect_ratios, num_scales, anchor_scale):
        self.min_level = min_level
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        self.scales = [2**(float(s) / num_scales) for s in range(num_scales)]

    def get(self, layer_id):
        widths = []
        heights = []
        for s in self.scales:
            for a in self.aspect_ratios:
                base_anchor_size = 2**(self.min_level + layer_id) * self.anchor_scale
                heights.append(base_anchor_size * s * a[1])
                widths.append(base_anchor_size * s * a[0])
        return widths, heights


def createGraph(modelPath, outputPath, min_level, aspect_ratios, num_scales,
                anchor_scale, num_classes, image_width, image_height):
    print('Min level: %d' % min_level)
    print('Anchor scale: %f' % anchor_scale)
    print('Num scales: %d' % num_scales)
    print('Aspect ratios: %s' % str(aspect_ratios))
    print('Number of classes: %d' % num_classes)
    print('Input image size: %dx%d' % (image_width, image_height))

    # Read the graph.
    _inpNames = ['image_arrays']
    outNames = ['detections']

    writeTextGraph(modelPath, outputPath, outNames)
    graph_def = parseTextGraph(outputPath)

    def getUnconnectedNodes():
        unconnected = []
        for node in graph_def.node:
            if node.op == 'Const':
                continue
            unconnected.append(node.name)
            for inp in node.input:
                if inp in unconnected:
                    unconnected.remove(inp)
        return unconnected


    nodesToKeep = ['truediv']  # Keep preprocessing nodes

    removeIdentity(graph_def)

    scopesToKeep = ('image_arrays', 'efficientnet', 'resample_p6', 'resample_p7',
                    'fpn_cells', 'class_net', 'box_net', 'Reshape', 'concat')

    addConstNode('scale_w', [2.0], graph_def)
    addConstNode('scale_h', [2.0], graph_def)
    nodesToKeep += ['scale_w', 'scale_h']

    for node in graph_def.node:
        if re.match('efficientnet-(.*)/blocks_\d+/se/mul_1', node.name):
            node.input[0], node.input[1] = node.input[1], node.input[0]

        if re.match('fpn_cells/cell_\d+/fnode\d+/resample(.*)/nearest_upsampling/Reshape_1$', node.name):
            node.op = 'ResizeNearestNeighbor'
            node.input[1] = 'scale_w'
            node.input.append('scale_h')

            for inpNode in graph_def.node:
                if inpNode.name == node.name[:node.name.rfind('_')]:
                    node.input[0] = inpNode.input[0]

        if re.match('box_net/box-predict(_\d)*/separable_conv2d$', node.name):
            node.addAttr('loc_pred_transposed', True)

        # Replace RealDiv to Mul with inversed scale for compatibility
        if node.op == 'RealDiv':
            for inpNode in graph_def.node:
                if inpNode.name != node.input[1] or not 'value' in inpNode.attr:
                    continue

                tensor = inpNode.attr['value']['tensor'][0]
                if not 'float_val' in tensor:
                    continue
                scale = float(inpNode.attr['value']['tensor'][0]['float_val'][0])

                addConstNode(inpNode.name + '/inv', [1.0 / scale], graph_def)
                nodesToKeep.append(inpNode.name + '/inv')
                node.input[1] = inpNode.name + '/inv'
                node.op = 'Mul'
                break


    def to_remove(name, op):
        if name in nodesToKeep:
            return False
        return op == 'Const' or not name.startswith(scopesToKeep)

    removeUnusedNodesAndAttrs(to_remove, graph_def)

    # Attach unconnected preprocessing
    assert(graph_def.node[1].name == 'truediv' and graph_def.node[1].op == 'RealDiv')
    graph_def.node[1].input.insert(0, 'image_arrays')
    graph_def.node[2].input.insert(0, 'truediv')

    priors_generator = AnchorGenerator(min_level, aspect_ratios, num_scales, anchor_scale)
    priorBoxes = []
    for i in range(5):
        inpName = ''
        for node in graph_def.node:
            if node.name == 'Reshape_%d' % (i * 2 + 1):
                inpName = node.input[0]
                break

        priorBox = NodeDef()
        priorBox.name = 'PriorBox_%d' % i
        priorBox.op = 'PriorBox'
        priorBox.input.append(inpName)
        priorBox.input.append(graph_def.node[0].name)  # image_tensor

        priorBox.addAttr('flip', False)
        priorBox.addAttr('clip', False)

        widths, heights = priors_generator.get(i)

        priorBox.addAttr('width', widths)
        priorBox.addAttr('height', heights)
        priorBox.addAttr('variance', [1.0, 1.0, 1.0, 1.0])

        graph_def.node.extend([priorBox])
        priorBoxes.append(priorBox.name)

    addConstNode('concat/axis_flatten', [-1], graph_def)

    def addConcatNode(name, inputs, axisNodeName):
        concat = NodeDef()
        concat.name = name
        concat.op = 'ConcatV2'
        for inp in inputs:
            concat.input.append(inp)
        concat.input.append(axisNodeName)
        graph_def.node.extend([concat])

    addConcatNode('PriorBox/concat', priorBoxes, 'concat/axis_flatten')

    sigmoid = NodeDef()
    sigmoid.name = 'concat/sigmoid'
    sigmoid.op = 'Sigmoid'
    sigmoid.input.append('concat')
    graph_def.node.extend([sigmoid])

    addFlatten(sigmoid.name, sigmoid.name + '/Flatten', graph_def)
    addFlatten('concat_1', 'concat_1/Flatten', graph_def)

    detectionOut = NodeDef()
    detectionOut.name = 'detection_out'
    detectionOut.op = 'DetectionOutput'

    detectionOut.input.append('concat_1/Flatten')
    detectionOut.input.append(sigmoid.name + '/Flatten')
    detectionOut.input.append('PriorBox/concat')

    detectionOut.addAttr('num_classes', num_classes)
    detectionOut.addAttr('share_location', True)
    detectionOut.addAttr('background_label_id', num_classes + 1)
    detectionOut.addAttr('nms_threshold', 0.6)
    detectionOut.addAttr('confidence_threshold', 0.2)
    detectionOut.addAttr('top_k', 100)
    detectionOut.addAttr('keep_top_k', 100)
    detectionOut.addAttr('code_type', "CENTER_SIZE")
    graph_def.node.extend([detectionOut])

    graph_def.node[0].attr['shape'] =  {
            'shape': {
                'dim': [
                    {'size': -1},
                    {'size': image_height},
                    {'size': image_width},
                    {'size': 3}
                ]
            }
        }

    while True:
        unconnectedNodes = getUnconnectedNodes()
        unconnectedNodes.remove(detectionOut.name)
        if not unconnectedNodes:
            break

        for name in unconnectedNodes:
            for i in range(len(graph_def.node)):
                if graph_def.node[i].name == name:
                    del graph_def.node[i]
                    break

    # Save as text
    graph_def.save(outputPath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run this script to get a text graph of '
                                                 'SSD model from TensorFlow Object Detection API. '
                                                 'Then pass it with .pb file to cv::dnn::readNetFromTensorflow function.')
    parser.add_argument('--input', required=True, help='Path to frozen TensorFlow graph.')
    parser.add_argument('--output', required=True, help='Path to output text graph.')
    parser.add_argument('--min_level', default=3, type=int, help='Parameter from training config')
    parser.add_argument('--num_scales', default=3, type=int, help='Parameter from training config')
    parser.add_argument('--anchor_scale', default=4.0, type=float, help='Parameter from training config')
    parser.add_argument('--aspect_ratios', default=[1.0, 1.0, 1.4, 0.7, 0.7, 1.4],
                        nargs='+', type=float, help='Parameter from training config')
    parser.add_argument('--num_classes', default=90, type=int, help='Number of classes to detect')
    parser.add_argument('--width', default=512, type=int, help='Network input width')
    parser.add_argument('--height', default=512, type=int, help='Network input height')
    args = parser.parse_args()

    ar = args.aspect_ratios
    assert(len(ar) % 2 == 0)
    ar = list(zip(ar[::2], ar[1::2]))

    createGraph(args.input, args.output, args.min_level, ar, args.num_scales,
                args.anchor_scale, args.num_classes, args.width, args.height)
