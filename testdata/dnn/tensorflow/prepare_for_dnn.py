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

def prepare_for_dnn(net, ckpt, freeze_graph_tool, optimizer_tool, transform_graph_tool,
                    input_node_name, output_node_name, out_graph, dtype):
    freeze_graph(net, ckpt, freeze_graph_tool, output_node_name)
    optimize_for_inference(dtype, optimizer_tool, input_node_name, output_node_name, out_graph)
    fuse_constants(transform_graph_tool, out_graph, out_graph, input_node_name, output_node_name)

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
