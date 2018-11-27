import argparse
import numpy as np
from tf_text_graph_common import *


def createFasterRCNNGraph(modelPath, configPath, outputPath):
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

    # Load a config file.
    config = readTextMessage(configPath)
    config = config['model'][0]['faster_rcnn'][0]
    num_classes = int(config['num_classes'][0])

    grid_anchor_generator = config['first_stage_anchor_generator'][0]['grid_anchor_generator'][0]
    scales = [float(s) for s in grid_anchor_generator['scales']]
    aspect_ratios = [float(ar) for ar in grid_anchor_generator['aspect_ratios']]
    width_stride = float(grid_anchor_generator['width_stride'][0])
    height_stride = float(grid_anchor_generator['height_stride'][0])
    features_stride = float(config['feature_extractor'][0]['first_stage_features_stride'][0])
    first_stage_nms_iou_threshold = float(config['first_stage_nms_iou_threshold'][0])
    first_stage_max_proposals = int(config['first_stage_max_proposals'][0])

    print('Number of classes: %d' % num_classes)
    print('Scales:            %s' % str(scales))
    print('Aspect ratios:     %s' % str(aspect_ratios))
    print('Width stride:      %f' % width_stride)
    print('Height stride:     %f' % height_stride)
    print('Features stride:   %f' % features_stride)

    # Read the graph.
    writeTextGraph(modelPath, outputPath, ['num_detections', 'detection_scores', 'detection_boxes', 'detection_classes'])
    graph_def = parseTextGraph(outputPath)

    removeIdentity(graph_def)

    def to_remove(name, op):
        return name.startswith(scopesToIgnore) or not name.startswith(scopesToKeep) or \
               (name.startswith('CropAndResize') and op != 'CropAndResize')

    removeUnusedNodesAndAttrs(to_remove, graph_def)


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

    proposals.addAttr('flip', False)
    proposals.addAttr('clip', True)
    proposals.addAttr('step', features_stride)
    proposals.addAttr('offset', 0.0)
    proposals.addAttr('variance', [0.1, 0.1, 0.2, 0.2])

    widths = []
    heights = []
    for a in aspect_ratios:
        for s in scales:
            ar = np.sqrt(a)
            heights.append((height_stride**2) * s / ar)
            widths.append((width_stride**2) * s * ar)

    proposals.addAttr('width', widths)
    proposals.addAttr('height', heights)

    graph_def.node.extend([proposals])

    # Compare with Reshape_5
    detectionOut = NodeDef()
    detectionOut.name = 'detection_out'
    detectionOut.op = 'DetectionOutput'

    detectionOut.input.append('FirstStageBoxPredictor/BoxEncodingPredictor/flatten')
    detectionOut.input.append('FirstStageBoxPredictor/ClassPredictor/softmax/flatten')
    detectionOut.input.append('proposals')

    detectionOut.addAttr('num_classes', 2)
    detectionOut.addAttr('share_location', True)
    detectionOut.addAttr('background_label_id', 0)
    detectionOut.addAttr('nms_threshold', first_stage_nms_iou_threshold)
    detectionOut.addAttr('top_k', 6000)
    detectionOut.addAttr('code_type', "CENTER_SIZE")
    detectionOut.addAttr('keep_top_k', first_stage_max_proposals)
    detectionOut.addAttr('clip', False)

    graph_def.node.extend([detectionOut])

    addConstNode('clip_by_value/lower', [0.0], graph_def)
    addConstNode('clip_by_value/upper', [1.0], graph_def)

    clipByValueNode = NodeDef()
    clipByValueNode.name = 'detection_out/clip_by_value'
    clipByValueNode.op = 'ClipByValue'
    clipByValueNode.input.append('detection_out')
    clipByValueNode.input.append('clip_by_value/lower')
    clipByValueNode.input.append('clip_by_value/upper')
    graph_def.node.extend([clipByValueNode])

    # Save as text.
    for node in reversed(topNodes):
        graph_def.node.extend([node])

    addSoftMax('SecondStageBoxPredictor/Reshape_1', 'SecondStageBoxPredictor/Reshape_1/softmax', graph_def)

    addSlice('SecondStageBoxPredictor/Reshape_1/softmax',
             'SecondStageBoxPredictor/Reshape_1/slice',
             [0, 0, 1], [-1, -1, -1], graph_def)

    addReshape('SecondStageBoxPredictor/Reshape_1/slice',
              'SecondStageBoxPredictor/Reshape_1/Reshape', [1, -1], graph_def)

    # Replace Flatten subgraph onto a single node.
    cropAndResizeNodeName = ''
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op == 'CropAndResize':
            graph_def.node[i].input.insert(1, 'detection_out/clip_by_value')
            cropAndResizeNodeName = graph_def.node[i].name

        if graph_def.node[i].name == 'SecondStageBoxPredictor/Reshape':
            addConstNode('SecondStageBoxPredictor/Reshape/shape2', [1, -1, 4], graph_def)

            graph_def.node[i].input.pop()
            graph_def.node[i].input.append('SecondStageBoxPredictor/Reshape/shape2')

        if graph_def.node[i].name in ['SecondStageBoxPredictor/Flatten/flatten/Shape',
                                      'SecondStageBoxPredictor/Flatten/flatten/strided_slice',
                                      'SecondStageBoxPredictor/Flatten/flatten/Reshape/shape',
                                      'SecondStageBoxPredictor/Flatten_1/flatten/Shape',
                                      'SecondStageBoxPredictor/Flatten_1/flatten/strided_slice',
                                      'SecondStageBoxPredictor/Flatten_1/flatten/Reshape/shape']:
            del graph_def.node[i]

    for node in graph_def.node:
        if node.name == 'SecondStageBoxPredictor/Flatten/flatten/Reshape' or \
           node.name == 'SecondStageBoxPredictor/Flatten_1/flatten/Reshape':
            node.op = 'Flatten'
            node.input.pop()

        if node.name in ['FirstStageBoxPredictor/BoxEncodingPredictor/Conv2D',
                         'SecondStageBoxPredictor/BoxEncodingPredictor/MatMul']:
            node.addAttr('loc_pred_transposed', True)

        if node.name.startswith('MaxPool2D'):
            assert(node.op == 'MaxPool')
            assert(cropAndResizeNodeName)
            node.input = [cropAndResizeNodeName]

    ################################################################################
    ### Postprocessing
    ################################################################################
    addSlice('detection_out/clip_by_value', 'detection_out/slice', [0, 0, 0, 3], [-1, -1, -1, 4], graph_def)

    variance = NodeDef()
    variance.name = 'proposals/variance'
    variance.op = 'Const'
    variance.addAttr('value', [0.1, 0.1, 0.2, 0.2])
    graph_def.node.extend([variance])

    varianceEncoder = NodeDef()
    varianceEncoder.name = 'variance_encoded'
    varianceEncoder.op = 'Mul'
    varianceEncoder.input.append('SecondStageBoxPredictor/Reshape')
    varianceEncoder.input.append(variance.name)
    varianceEncoder.addAttr('axis', 2)
    graph_def.node.extend([varianceEncoder])

    addReshape('detection_out/slice', 'detection_out/slice/reshape', [1, 1, -1], graph_def)
    addFlatten('variance_encoded', 'variance_encoded/flatten', graph_def)

    detectionOut = NodeDef()
    detectionOut.name = 'detection_out_final'
    detectionOut.op = 'DetectionOutput'

    detectionOut.input.append('variance_encoded/flatten')
    detectionOut.input.append('SecondStageBoxPredictor/Reshape_1/Reshape')
    detectionOut.input.append('detection_out/slice/reshape')

    detectionOut.addAttr('num_classes', num_classes)
    detectionOut.addAttr('share_location', False)
    detectionOut.addAttr('background_label_id', num_classes + 1)
    detectionOut.addAttr('nms_threshold', 0.6)
    detectionOut.addAttr('code_type', "CENTER_SIZE")
    detectionOut.addAttr('keep_top_k', 100)
    detectionOut.addAttr('clip', True)
    detectionOut.addAttr('variance_encoded_in_target', True)
    graph_def.node.extend([detectionOut])

    # Save as text.
    graph_def.save(outputPath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run this script to get a text graph of '
                                                 'Faster-RCNN model from TensorFlow Object Detection API. '
                                                 'Then pass it with .pb file to cv::dnn::readNetFromTensorflow function.')
    parser.add_argument('--input', required=True, help='Path to frozen TensorFlow graph.')
    parser.add_argument('--output', required=True, help='Path to output text graph.')
    parser.add_argument('--config', required=True, help='Path to a *.config file is used for training.')
    args = parser.parse_args()

    createFasterRCNNGraph(args.input, args.config, args.output)
