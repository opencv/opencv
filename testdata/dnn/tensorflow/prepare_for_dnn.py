import argparse
import tensorflow as tf
import os

# Freeze graph. Replaces variables to constants. Removes training-only ops.
def freeze_graph(net, ckpt, tool, out_node, out_graph='frozen_graph.pb'):
    os.system('python ' + tool +
              ' --input_graph ' + net +
              ' --input_checkpoint ' + ckpt +
              ' --output_graph ' + out_graph +
              ' --output_node_names ' + out_node)

def optimize_for_inference(dtype, tool, in_node, out_node, out_graph='optimized_graph.pb',
                           frozen_graph='frozen_graph.pb'):
    os.system('python ' + tool +
              ' --input ' + frozen_graph +
              ' --output ' + out_graph +
              ' --frozen_graph True ' +
              ' --input_names ' + in_node +
              ' --output_names ' + out_node +
              ' --placeholder_type_enum ' + str(dtype.as_datatype_enum))

def fuse_constants(tool, in_graph, out_graph, in_node, out_node):
    os.system(tool + ' --in_graph=' + out_graph + \
                     ' --out_graph=' + out_graph + \
                     ' --inputs=' + in_node + \
                     ' --outputs=' + out_node + \
                     ' --transforms="fold_constants(ignore_errors=True) sort_by_execution_order"')

# WARNING: If there is LSTM this procedure creates the graph that won't work in TensorFlow anymore.
def simplify_lstm(tool, in_graph, out_graph, in_node, out_node):
    # Get all BlockLSTM nodes names.
    lstm_names = []
    with tf.gfile.FastGFile(in_graph) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            if node.op == 'BlockLSTM':
                lstm_names.append(node.name)

        # BlockLSTM returns tuple of tensors (i, cs, f, o, ci, co, h)
        # where output is h (6th tensor). So we replace an every reference to
        # an lstm_node_name:6 onto just an lstm_node_name.
        for node in graph_def.node:
            for i in range(len(node.input)):
                for name in lstm_names:
                    if node.input[i] == name + ':6':
                        node.input[i] = name
        tf.train.write_graph(graph_def, "", out_graph, as_text=False)

    if lstm_names:  # There is at least one lstm block.
        os.system(tool + ' --in_graph=' + out_graph + \
                         ' --out_graph=' + out_graph + \
                         ' --inputs=' + in_node + \
                         ' --outputs=' + out_node + \
                         ' --transforms="remove_nodes(op=Cast) '
                         'rename_op(old_op_name=Fill, new_op_name=Const)"')

def prepare_for_dnn(net, ckpt, freeze_graph_tool, optimizer_tool, transform_graph_tool,
                    input_node_name, output_node_name, out_graph, dtype):
    freeze_graph(net, ckpt, freeze_graph_tool, output_node_name)
    optimize_for_inference(dtype, optimizer_tool, input_node_name, output_node_name, out_graph)
    fuse_constants(transform_graph_tool, out_graph, out_graph, input_node_name, output_node_name)
    simplify_lstm(transform_graph_tool, out_graph, out_graph, input_node_name, output_node_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for preparing serialized '
                                                 'TensorFlow graph to import into DNN. '
                                                 'Modified graph still may be used in TensorFlow.')
    parser.add_argument('-frz', dest='freeze_graph_tool', required=True,
                        help='Path to freeze_graph.py tool')
    parser.add_argument('-opt', dest='optimizer_tool', required=True,
                        help='Path to optimize_for_inference.py tool')
    parser.add_argument('-tr', dest='transform_graph_tool', required=True,
                        help='Path to transform_graph tool')
    parser.add_argument('-net', dest='net', required=True,
                        help='Path serialized graph by tf.train.write_graph()')
    parser.add_argument('-ckpt', dest='ckpt', required=True,
                        help='Path saved checkpoint by saver.save()')
    parser.add_argument('-in_node', dest='input_name', required=True,
                        help='Input op name')
    parser.add_argument('-out_node', dest='output_name', required=True,
                        help='Output op name')
    parser.add_argument('-o', dest='output', required=True,
                        help='Output graph name')
    args = parser.parse_args()

    prepare_for_dnn(args.net, args.ckpt, args.freeze_graph_tool,
                    args.optimizer_tool, args.transform_graph_tool,
                    args.input_name, args.output_name, args.output)
