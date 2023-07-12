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
import re
from math import sqrt
from tf_text_graph_common import *

class SSDAnchorGenerator:
    def __init__(self, min_scale, max_scale, num_layers, aspect_ratios,
                 reduce_boxes_in_lowest_layer, image_width, image_height):
        self.min_scale = min_scale
        self.aspect_ratios = aspect_ratios
        self.reduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer
        self.image_width = image_width
        self.image_height = image_height
        self.scales =  [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
                            for i in range(num_layers)] + [1.0]

    def get(self, layer_id):
        if layer_id == 0 and self.reduce_boxes_in_lowest_layer:
            widths = [0.1, self.min_scale * sqrt(2.0), self.min_scale * sqrt(0.5)]
            heights = [0.1, self.min_scale / sqrt(2.0), self.min_scale / sqrt(0.5)]
        else:
            widths = [self.scales[layer_id] * sqrt(ar) for ar in self.aspect_ratios]
            heights = [self.scales[layer_id] / sqrt(ar) for ar in self.aspect_ratios]

            widths += [sqrt(self.scales[layer_id] * self.scales[layer_id + 1])]
            heights += [sqrt(self.scales[layer_id] * self.scales[layer_id + 1])]
        min_size = min(self.image_width, self.image_height)
        widths = [w * min_size for w in widths]
        heights = [h * min_size for h in heights]
        return widths, heights


class MultiscaleAnchorGenerator:
    def __init__(self, min_level, aspect_ratios, scales_per_octave, anchor_scale):
        self.min_level = min_level
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        self.scales = [2**(float(s) / scales_per_octave) for s in range(scales_per_octave)]

    def get(self, layer_id):
        widths = []
        heights = []
        for a in self.aspect_ratios:
            for s in self.scales:
                base_anchor_size = 2**(self.min_level + layer_id) * self.anchor_scale
                ar = sqrt(a)
                heights.append(base_anchor_size * s / ar)
                widths.append(base_anchor_size * s * ar)
        return widths, heights


def createSSDGraph(modelPath, configPath, outputPath):
    # Nodes that should be kept.
    keepOps = ['Conv2D', 'BiasAdd', 'Add', 'AddV2', 'Relu', 'Relu6', 'Placeholder', 'FusedBatchNorm',
               'DepthwiseConv2dNative', 'ConcatV2', 'Mul', 'MaxPool', 'AvgPool', 'Identity',
               'Sub', 'ResizeNearestNeighbor', 'Pad', 'FusedBatchNormV3', 'Mean']

    # Node with which prefixes should be removed
    prefixesToRemove = ('MultipleGridAnchorGenerator/', 'Concatenate/', 'Postprocessor/', 'Preprocessor/map')

    # Load a config file.
    config = readTextMessage(configPath)
    config = config['model'][0]['ssd'][0]
    num_classes = int(config['num_classes'][0])

    fixed_shape_resizer = config['image_resizer'][0]['fixed_shape_resizer'][0]
    image_width = int(fixed_shape_resizer['width'][0])
    image_height = int(fixed_shape_resizer['height'][0])

    box_predictor = 'convolutional' if 'convolutional_box_predictor' in config['box_predictor'][0] else 'weight_shared_convolutional'

    anchor_generator = config['anchor_generator'][0]
    if 'ssd_anchor_generator' in anchor_generator:
        ssd_anchor_generator = anchor_generator['ssd_anchor_generator'][0]
        min_scale = float(ssd_anchor_generator['min_scale'][0])
        max_scale = float(ssd_anchor_generator['max_scale'][0])
        num_layers = int(ssd_anchor_generator['num_layers'][0])
        aspect_ratios = [float(ar) for ar in ssd_anchor_generator['aspect_ratios']]
        reduce_boxes_in_lowest_layer = True
        if 'reduce_boxes_in_lowest_layer' in ssd_anchor_generator:
            reduce_boxes_in_lowest_layer = ssd_anchor_generator['reduce_boxes_in_lowest_layer'][0] == 'true'
        priors_generator = SSDAnchorGenerator(min_scale, max_scale, num_layers,
                                              aspect_ratios, reduce_boxes_in_lowest_layer,
                                              image_width, image_height)


        print('Scale: [%f-%f]' % (min_scale, max_scale))
        print('Aspect ratios: %s' % str(aspect_ratios))
        print('Reduce boxes in the lowest layer: %s' % str(reduce_boxes_in_lowest_layer))
    elif 'multiscale_anchor_generator' in anchor_generator:
        multiscale_anchor_generator = anchor_generator['multiscale_anchor_generator'][0]
        min_level = int(multiscale_anchor_generator['min_level'][0])
        max_level = int(multiscale_anchor_generator['max_level'][0])
        anchor_scale = float(multiscale_anchor_generator['anchor_scale'][0])
        aspect_ratios = [float(ar) for ar in multiscale_anchor_generator['aspect_ratios']]
        scales_per_octave = int(multiscale_anchor_generator['scales_per_octave'][0])
        num_layers = max_level - min_level + 1
        priors_generator = MultiscaleAnchorGenerator(min_level, aspect_ratios,
                                                     scales_per_octave, anchor_scale)
        print('Levels: [%d-%d]' % (min_level, max_level))
        print('Anchor scale: %f' % anchor_scale)
        print('Scales per octave: %d' % scales_per_octave)
        print('Aspect ratios: %s' % str(aspect_ratios))
    else:
        print('Unknown anchor_generator')
        exit(0)

    print('Number of classes: %d' % num_classes)
    print('Number of layers: %d' % num_layers)
    print('box predictor: %s' % box_predictor)
    print('Input image size: %dx%d' % (image_width, image_height))

    # Read the graph.
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


    def fuse_nodes(nodesToKeep):
        # Detect unfused batch normalization nodes and fuse them.
        # Add_0 <-- moving_variance, add_y
        # Rsqrt <-- Add_0
        # Mul_0 <-- Rsqrt, gamma
        # Mul_1 <-- input, Mul_0
        # Mul_2 <-- moving_mean, Mul_0
        # Sub_0 <-- beta, Mul_2
        # Add_1 <-- Mul_1, Sub_0
        nodesMap = {node.name: node for node in graph_def.node}
        subgraphBatchNorm = ['Add',
            ['Mul', 'input', ['Mul', ['Rsqrt', ['Add', 'moving_variance', 'add_y']], 'gamma']],
            ['Sub', 'beta', ['Mul', 'moving_mean', 'Mul_0']]]
        subgraphBatchNormV2 = ['AddV2',
            ['Mul', 'input', ['Mul', ['Rsqrt', ['AddV2', 'moving_variance', 'add_y']], 'gamma']],
            ['Sub', 'beta', ['Mul', 'moving_mean', 'Mul_0']]]
        # Detect unfused nearest neighbor resize.
        subgraphResizeNN = ['Reshape',
            ['Mul', ['Reshape', 'input', ['Pack', 'shape_1', 'shape_2', 'shape_3', 'shape_4', 'shape_5']],
                    'ones'],
            ['Pack', ['StridedSlice', ['Shape', 'input'], 'stack', 'stack_1', 'stack_2'],
                     'out_height', 'out_width', 'out_channels']]
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
            if checkSubgraph(node, subgraphBatchNorm, inputs, fusedNodes) or \
               checkSubgraph(node, subgraphBatchNormV2, inputs, fusedNodes):
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

            inputs = {}
            fusedNodes = []
            if checkSubgraph(node, subgraphResizeNN, inputs, fusedNodes):
                name = node.name
                node.Clear()
                node.name = name
                node.op = 'ResizeNearestNeighbor'
                node.input.append(inputs['input'])
                node.input.append(name + '/output_shape')

                out_height_node = nodesMap[inputs['out_height']]
                out_width_node = nodesMap[inputs['out_width']]
                out_height = int(out_height_node.attr['value']['tensor'][0]['int_val'][0])
                out_width = int(out_width_node.attr['value']['tensor'][0]['int_val'][0])

                shapeNode = NodeDef()
                shapeNode.name = name + '/output_shape'
                shapeNode.op = 'Const'
                shapeNode.addAttr('value', [out_height, out_width])
                graph_def.node.insert(graph_def.node.index(node), shapeNode)
                nodesToKeep.append(shapeNode.name)

                nodesToRemove += fusedNodes[1:]
        for node in nodesToRemove:
            graph_def.node.remove(node)

    nodesToKeep = []
    fuse_nodes(nodesToKeep)

    removeIdentity(graph_def)

    def to_remove(name, op):
        return (not name in nodesToKeep) and \
               (op == 'Const' or (not op in keepOps) or name.startswith(prefixesToRemove))

    removeUnusedNodesAndAttrs(to_remove, graph_def)


    # Connect input node to the first layer
    assert(graph_def.node[0].op == 'Placeholder')
    try:
        input_shape = graph_def.node[0].attr['shape']['shape'][0]['dim']
        input_shape[1]['size'] = image_height
        input_shape[2]['size'] = image_width
    except:
        print("Input shapes are undefined")
    # assert(graph_def.node[1].op == 'Conv2D')
    weights = graph_def.node[1].input[-1]
    for i in range(len(graph_def.node[1].input)):
        graph_def.node[1].input.pop()
    graph_def.node[1].input.append(graph_def.node[0].name)
    graph_def.node[1].input.append(weights)

    # check and correct the case when preprocessing block is after input
    preproc_id = "Preprocessor/"
    if graph_def.node[2].name.startswith(preproc_id) and \
        graph_def.node[2].input[0].startswith(preproc_id):

        if not any(preproc_id in inp for inp in graph_def.node[3].input):
            graph_def.node[3].input.insert(0, graph_def.node[2].name)


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

    for label in ['ClassPredictor', 'BoxEncodingPredictor' if box_predictor == 'convolutional' else 'BoxPredictor']:
        concatInputs = []
        for i in range(num_layers):
            # Flatten predictions
            flatten = NodeDef()
            if box_predictor == 'convolutional':
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

    num_matched_layers = 0
    for node in graph_def.node:
        if re.match('BoxPredictor_\d/BoxEncodingPredictor/convolution', node.name) or \
           re.match('BoxPredictor_\d/BoxEncodingPredictor/Conv2D', node.name) or \
           re.match('WeightSharedConvolutionalBoxPredictor(_\d)*/BoxPredictor/Conv2D', node.name):
            node.addAttr('loc_pred_transposed', True)
            num_matched_layers += 1
    assert(num_matched_layers == num_layers)

    # Add layers that generate anchors (bounding boxes proposals).
    priorBoxes = []
    boxCoder = config['box_coder'][0]
    fasterRcnnBoxCoder = boxCoder['faster_rcnn_box_coder'][0]
    boxCoderVariance = [1.0/float(fasterRcnnBoxCoder['x_scale'][0]), 1.0/float(fasterRcnnBoxCoder['y_scale'][0]), 1.0/float(fasterRcnnBoxCoder['width_scale'][0]), 1.0/float(fasterRcnnBoxCoder['height_scale'][0])]
    for i in range(num_layers):
        priorBox = NodeDef()
        priorBox.name = 'PriorBox_%d' % i
        priorBox.op = 'PriorBox'
        if box_predictor == 'convolutional':
            priorBox.input.append('BoxPredictor_%d/BoxEncodingPredictor/BiasAdd' % i)
        else:
            if i == 0:
                priorBox.input.append('WeightSharedConvolutionalBoxPredictor/BoxPredictor/Conv2D')
            else:
                priorBox.input.append('WeightSharedConvolutionalBoxPredictor_%d/BoxPredictor/BiasAdd' % i)
        priorBox.input.append(graph_def.node[0].name)  # image_tensor

        priorBox.addAttr('flip', False)
        priorBox.addAttr('clip', False)

        widths, heights = priors_generator.get(i)

        priorBox.addAttr('width', widths)
        priorBox.addAttr('height', heights)
        priorBox.addAttr('variance', boxCoderVariance)

        graph_def.node.extend([priorBox])
        priorBoxes.append(priorBox.name)

    # Compare this layer's output with Postprocessor/Reshape
    addConcatNode('PriorBox/concat', priorBoxes, 'concat/axis_flatten')

    # Sigmoid for classes predictions and DetectionOutput layer
    addReshape('ClassPredictor/concat', 'ClassPredictor/concat3d', [0, -1, num_classes + 1], graph_def)

    sigmoid = NodeDef()
    sigmoid.name = 'ClassPredictor/concat/sigmoid'
    sigmoid.op = 'Sigmoid'
    sigmoid.input.append('ClassPredictor/concat3d')
    graph_def.node.extend([sigmoid])

    addFlatten(sigmoid.name, sigmoid.name + '/Flatten', graph_def)

    detectionOut = NodeDef()
    detectionOut.name = 'detection_out'
    detectionOut.op = 'DetectionOutput'

    if box_predictor == 'convolutional':
        detectionOut.input.append('BoxEncodingPredictor/concat')
    else:
        detectionOut.input.append('BoxPredictor/concat')
    detectionOut.input.append(sigmoid.name + '/Flatten')
    detectionOut.input.append('PriorBox/concat')

    detectionOut.addAttr('num_classes', num_classes + 1)
    detectionOut.addAttr('share_location', True)
    detectionOut.addAttr('background_label_id', 0)

    postProcessing = config['post_processing'][0]
    batchNMS = postProcessing['batch_non_max_suppression'][0]

    if 'iou_threshold' in batchNMS:
        detectionOut.addAttr('nms_threshold', float(batchNMS['iou_threshold'][0]))
    else:
        detectionOut.addAttr('nms_threshold', 0.6)

    if 'score_threshold' in batchNMS:
        detectionOut.addAttr('confidence_threshold', float(batchNMS['score_threshold'][0]))
    else:
        detectionOut.addAttr('confidence_threshold', 0.01)

    if 'max_detections_per_class' in batchNMS:
        detectionOut.addAttr('top_k', int(batchNMS['max_detections_per_class'][0]))
    else:
        detectionOut.addAttr('top_k', 100)

    if 'max_total_detections' in batchNMS:
        detectionOut.addAttr('keep_top_k', int(batchNMS['max_total_detections'][0]))
    else:
        detectionOut.addAttr('keep_top_k', 100)

    detectionOut.addAttr('code_type', "CENTER_SIZE")

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
