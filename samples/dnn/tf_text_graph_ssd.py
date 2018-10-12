# This file is a part of OpenCV project.
# It is a subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
#
# Copyright (C) 2018, Intel Corporation, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# Use this script to get the text graph representation (.pbtxt) of SSD-based
# deep learning network trained in TensorFlow Object Detection API.
# Then you can import it with a binary frozen graph (.pb) using readNetFromTensorflow() function.
# See details and examples on the following wiki page: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
import argparse
from math import sqrt
from tf_text_graph_common import *

def createSSDGraph(modelPath, configPath, outputPath):
    # Nodes that should be kept.
    keepOps = ['Conv2D', 'BiasAdd', 'Add', 'Relu6', 'Placeholder', 'FusedBatchNorm',
               'DepthwiseConv2dNative', 'ConcatV2', 'Mul', 'MaxPool', 'AvgPool', 'Identity',
               'Sub']

    # Node with which prefixes should be removed
    prefixesToRemove = ('MultipleGridAnchorGenerator/', 'Postprocessor/', 'Preprocessor/map')

    # Load a config file.
    config = readTextMessage(configPath)
    config = config['model'][0]['ssd'][0]
    num_classes = int(config['num_classes'][0])

    ssd_anchor_generator = config['anchor_generator'][0]['ssd_anchor_generator'][0]
    min_scale = float(ssd_anchor_generator['min_scale'][0])
    max_scale = float(ssd_anchor_generator['max_scale'][0])
    num_layers = int(ssd_anchor_generator['num_layers'][0])
    aspect_ratios = [float(ar) for ar in ssd_anchor_generator['aspect_ratios']]
    reduce_boxes_in_lowest_layer = True
    if 'reduce_boxes_in_lowest_layer' in ssd_anchor_generator:
        reduce_boxes_in_lowest_layer = ssd_anchor_generator['reduce_boxes_in_lowest_layer'][0] == 'true'

    fixed_shape_resizer = config['image_resizer'][0]['fixed_shape_resizer'][0]
    image_width = int(fixed_shape_resizer['width'][0])
    image_height = int(fixed_shape_resizer['height'][0])

    box_predictor = 'convolutional' if 'convolutional_box_predictor' in config['box_predictor'][0] else 'weight_shared_convolutional'

    print('Number of classes: %d' % num_classes)
    print('Number of layers: %d' % num_layers)
    print('Scale: [%f-%f]' % (min_scale, max_scale))
    print('Aspect ratios: %s' % str(aspect_ratios))
    print('Reduce boxes in the lowest layer: %s' % str(reduce_boxes_in_lowest_layer))
    print('box predictor: %s' % box_predictor)
    print('Input image size: %dx%d' % (image_width, image_height))

    # Read the graph.
    inpNames = ['image_tensor']
    outNames = ['num_detections', 'detection_scores', 'detection_boxes', 'detection_classes']

    writeTextGraph(modelPath, outputPath, outNames)
    graph_def = parseTextGraph(outputPath)

    def getUnconnectedNodes():
        unconnected = []
        for node in graph_def.node:
            unconnected.append(node.name)
            for inp in node.input:
                if inp in unconnected:
                    unconnected.remove(inp)
        return unconnected


    # Detect unfused batch normalization nodes and fuse them.
    def fuse_batch_normalization():
        # Add_0 <-- moving_variance, add_y
        # Rsqrt <-- Add_0
        # Mul_0 <-- Rsqrt, gamma
        # Mul_1 <-- input, Mul_0
        # Mul_2 <-- moving_mean, Mul_0
        # Sub_0 <-- beta, Mul_2
        # Add_1 <-- Mul_1, Sub_0
        nodesMap = {node.name: node for node in graph_def.node}
        subgraph = ['Add',
            ['Mul', 'input', ['Mul', ['Rsqrt', ['Add', 'moving_variance', 'add_y']], 'gamma']],
            ['Sub', 'beta', ['Mul', 'moving_mean', 'Mul_0']]]
        def checkSubgraph(node, targetNode, inputs, fusedNodes):
            op = targetNode[0]
            if node.op == op and (len(node.input) >= len(targetNode) - 1):
                fusedNodes.append(node)
                for i, inpOp in enumerate(targetNode[1:]):
                    if isinstance(inpOp, list):
                        if not node.input[i] in nodesMap or \
                           not checkSubgraph(nodesMap[node.input[i]], inpOp, inputs, fusedNodes):
                            return False
                    else:
                        inputs[inpOp] = node.input[i]

                return True
            else:
                return False

        nodesToRemove = []
        for node in graph_def.node:
            inputs = {}
            fusedNodes = []
            if checkSubgraph(node, subgraph, inputs, fusedNodes):
                name = node.name
                node.Clear()
                node.name = name
                node.op = 'FusedBatchNorm'
                node.input.append(inputs['input'])
                node.input.append(inputs['gamma'])
                node.input.append(inputs['beta'])
                node.input.append(inputs['moving_mean'])
                node.input.append(inputs['moving_variance'])
                node.addAttr('epsilon', 0.001)
                nodesToRemove += fusedNodes[1:]
        for node in nodesToRemove:
            graph_def.node.remove(node)

    fuse_batch_normalization()

    removeIdentity(graph_def)

    def to_remove(name, op):
        return (not op in keepOps) or name.startswith(prefixesToRemove)

    removeUnusedNodesAndAttrs(to_remove, graph_def)


    # Connect input node to the first layer
    assert(graph_def.node[0].op == 'Placeholder')
    # assert(graph_def.node[1].op == 'Conv2D')
    weights = graph_def.node[1].input[0]
    for i in range(len(graph_def.node[1].input)):
        graph_def.node[1].input.pop()
    graph_def.node[1].input.append(graph_def.node[0].name)
    graph_def.node[1].input.append(weights)

    # Create SSD postprocessing head ###############################################

    # Concatenate predictions of classes, predictions of bounding boxes and proposals.
    def addConcatNode(name, inputs, axisNodeName):
        concat = NodeDef()
        concat.name = name
        concat.op = 'ConcatV2'
        for inp in inputs:
            concat.input.append(inp)
        concat.input.append(axisNodeName)
        graph_def.node.extend([concat])

    addConstNode('concat/axis_flatten', [-1], graph_def)
    addConstNode('PriorBox/concat/axis', [-2], graph_def)

    for label in ['ClassPredictor', 'BoxEncodingPredictor' if box_predictor is 'convolutional' else 'BoxPredictor']:
        concatInputs = []
        for i in range(num_layers):
            # Flatten predictions
            flatten = NodeDef()
            if box_predictor is 'convolutional':
                inpName = 'BoxPredictor_%d/%s/BiasAdd' % (i, label)
            else:
                if i == 0:
                    inpName = 'WeightSharedConvolutionalBoxPredictor/%s/BiasAdd' % label
                else:
                    inpName = 'WeightSharedConvolutionalBoxPredictor_%d/%s/BiasAdd' % (i, label)
            flatten.input.append(inpName)
            flatten.name = inpName + '/Flatten'
            flatten.op = 'Flatten'

            concatInputs.append(flatten.name)
            graph_def.node.extend([flatten])
        addConcatNode('%s/concat' % label, concatInputs, 'concat/axis_flatten')

    idx = 0
    for node in graph_def.node:
        if node.name == ('BoxPredictor_%d/BoxEncodingPredictor/Conv2D' % idx) or \
           node.name == ('WeightSharedConvolutionalBoxPredictor_%d/BoxPredictor/Conv2D' % idx) or \
           node.name == 'WeightSharedConvolutionalBoxPredictor/BoxPredictor/Conv2D':
            node.addAttr('loc_pred_transposed', True)
            idx += 1
    assert(idx == num_layers)

    # Add layers that generate anchors (bounding boxes proposals).
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)] + [1.0]

    priorBoxes = []
    for i in range(num_layers):
        priorBox = NodeDef()
        priorBox.name = 'PriorBox_%d' % i
        priorBox.op = 'PriorBox'
        if box_predictor is 'convolutional':
            priorBox.input.append('BoxPredictor_%d/BoxEncodingPredictor/BiasAdd' % i)
        else:
            if i == 0:
                priorBox.input.append('WeightSharedConvolutionalBoxPredictor/BoxPredictor/Conv2D')
            else:
                priorBox.input.append('WeightSharedConvolutionalBoxPredictor_%d/BoxPredictor/BiasAdd' % i)
        priorBox.input.append(graph_def.node[0].name)  # image_tensor

        priorBox.addAttr('flip', False)
        priorBox.addAttr('clip', False)

        if i == 0 and reduce_boxes_in_lowest_layer:
            widths = [0.1, min_scale * sqrt(2.0), min_scale * sqrt(0.5)]
            heights = [0.1, min_scale / sqrt(2.0), min_scale / sqrt(0.5)]
        else:
            widths = [scales[i] * sqrt(ar) for ar in aspect_ratios]
            heights = [scales[i] / sqrt(ar) for ar in aspect_ratios]

            widths += [sqrt(scales[i] * scales[i + 1])]
            heights += [sqrt(scales[i] * scales[i + 1])]
        widths = [w * image_width for w in widths]
        heights = [h * image_height for h in heights]
        priorBox.addAttr('width', widths)
        priorBox.addAttr('height', heights)
        priorBox.addAttr('variance', [0.1, 0.1, 0.2, 0.2])

        graph_def.node.extend([priorBox])
        priorBoxes.append(priorBox.name)

    addConcatNode('PriorBox/concat', priorBoxes, 'concat/axis_flatten')

    # Sigmoid for classes predictions and DetectionOutput layer
    sigmoid = NodeDef()
    sigmoid.name = 'ClassPredictor/concat/sigmoid'
    sigmoid.op = 'Sigmoid'
    sigmoid.input.append('ClassPredictor/concat')
    graph_def.node.extend([sigmoid])

    detectionOut = NodeDef()
    detectionOut.name = 'detection_out'
    detectionOut.op = 'DetectionOutput'

    if box_predictor == 'convolutional':
        detectionOut.input.append('BoxEncodingPredictor/concat')
    else:
        detectionOut.input.append('BoxPredictor/concat')
    detectionOut.input.append(sigmoid.name)
    detectionOut.input.append('PriorBox/concat')

    detectionOut.addAttr('num_classes', num_classes + 1)
    detectionOut.addAttr('share_location', True)
    detectionOut.addAttr('background_label_id', 0)
    detectionOut.addAttr('nms_threshold', 0.6)
    detectionOut.addAttr('top_k', 100)
    detectionOut.addAttr('code_type', "CENTER_SIZE")
    detectionOut.addAttr('keep_top_k', 100)
    detectionOut.addAttr('confidence_threshold', 0.01)

    graph_def.node.extend([detectionOut])

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

    # Save as text.
    graph_def.save(outputPath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run this script to get a text graph of '
                                                 'SSD model from TensorFlow Object Detection API. '
                                                 'Then pass it with .pb file to cv::dnn::readNetFromTensorflow function.')
    parser.add_argument('--input', required=True, help='Path to frozen TensorFlow graph.')
    parser.add_argument('--output', required=True, help='Path to output text graph.')
    parser.add_argument('--config', required=True, help='Path to a *.config file is used for training.')
    args = parser.parse_args()

    createSSDGraph(args.input, args.config, args.output)
