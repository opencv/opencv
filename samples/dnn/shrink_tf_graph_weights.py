# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
#
# Copyright (C) 2017, Intel Corporation, all rights reserved.
# Third party copyrights are property of their respective owners.
import tensorflow as tf
import struct
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Convert weights of a frozen TensorFlow graph to fp16.')
parser.add_argument('--input', required=True, help='Path to frozen graph.')
parser.add_argument('--output', required=True, help='Path to output graph.')
parser.add_argument('--ops', default=['Conv2D', 'MatMul'], nargs='+',
                    help='List of ops which weights are converted.')
args = parser.parse_args()

DT_FLOAT = 1
DT_HALF = 19

# For the frozen graphs, an every node that uses weights connected to Const nodes
# through an Identity node. Usually they're called in the same way with '/read' suffix.
# We'll replace all of them to Cast nodes.

# Load the model
with tf.gfile.FastGFile(args.input) as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Set of all inputs from desired nodes.
inputs = []
for node in graph_def.node:
    if node.op in args.ops:
        inputs += node.input

weightsNodes = []
for node in graph_def.node:
    # From the whole inputs we need to keep only an Identity nodes.
    if node.name in inputs and node.op == 'Identity' and node.attr['T'].type == DT_FLOAT:
        weightsNodes.append(node.input[0])

        # Replace Identity to Cast.
        node.op = 'Cast'
        node.attr['DstT'].type = DT_FLOAT
        node.attr['SrcT'].type = DT_HALF
        del node.attr['T']
        del node.attr['_class']

# Convert weights to halfs.
for node in graph_def.node:
    if node.name in weightsNodes:
        node.attr['dtype'].type = DT_HALF
        node.attr['value'].tensor.dtype = DT_HALF

        floats = node.attr['value'].tensor.tensor_content

        floats = struct.unpack('f' * (len(floats) / 4), floats)
        halfs = np.array(floats).astype(np.float16).view(np.uint16)
        node.attr['value'].tensor.tensor_content = struct.pack('H' * len(halfs), *halfs)

tf.train.write_graph(graph_def, "", args.output, as_text=False)
