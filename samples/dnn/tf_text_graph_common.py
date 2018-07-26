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
