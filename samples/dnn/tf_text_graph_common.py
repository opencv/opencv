import tensorflow as tf
from tensorflow.core.framework.node_def_pb2 import NodeDef
from google.protobuf import text_format

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

def addConstNode(name, values, graph_def):
    node = NodeDef()
    node.name = name
    node.op = 'Const'
    text_format.Merge(tensorMsg(values), node.attr["value"])
    graph_def.node.extend([node])


def addSlice(inp, out, begins, sizes, graph_def):
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


def addReshape(inp, out, shape, graph_def):
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


def addSoftMax(inp, out, graph_def):
    softmax = NodeDef()
    softmax.name = out
    softmax.op = 'Softmax'
    text_format.Merge('i: -1', softmax.attr['axis'])
    softmax.input.append(inp)
    graph_def.node.extend([softmax])


def addFlatten(inp, out, graph_def):
    flatten = NodeDef()
    flatten.name = out
    flatten.op = 'Flatten'
    flatten.input.append(inp)
    graph_def.node.extend([flatten])


# Removes Identity nodes
def removeIdentity(graph_def):
    identities = {}
    for node in graph_def.node:
        if node.op == 'Identity':
            identities[node.name] = node.input[0]
            graph_def.node.remove(node)

    for node in graph_def.node:
        for i in range(len(node.input)):
            if node.input[i] in identities:
                node.input[i] = identities[node.input[i]]


def removeUnusedNodesAndAttrs(to_remove, graph_def):
    unusedAttrs = ['T', 'Tshape', 'N', 'Tidx', 'Tdim', 'use_cudnn_on_gpu',
                   'Index', 'Tperm', 'is_training', 'Tpaddings']

    removedNodes = []

    for i in reversed(range(len(graph_def.node))):
        op = graph_def.node[i].op
        name = graph_def.node[i].name

        if op == 'Const' or to_remove(name, op):
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
