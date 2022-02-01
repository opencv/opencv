import os
import subprocess

pipeline_modeling_tool = os.getenv('PIPELINE_MODELING_TOOL')

def get_output(exec_str):
    try:
        out = subprocess.check_output(exec_str,
                                      stderr=subprocess.STDOUT,
                                      shell=True).strip().decode()
    except subprocess.CalledProcessError as exc:
        out = exc.output.strip().decode()
    return out


def test_error_no_config_specified():
    out = get_output(pipeline_modeling_tool)
    assert out.startswith('Config must be specified via --cfg option')


def test_error_no_config_exists():
    cfg_file = 'not_existing_cfg.yml'

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert 'Failed to open config file: not_existing_cfg.yml' in out


def test_error_no_work_time():
    cfg_file = """\"%YAML:1.0\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Config must contain field: work_time')


def test_error_work_time_not_positive():
    cfg_file = """\"%YAML:1.0
work_time: -1\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('work_time must be positive')


def test_error_no_pipelines():
    cfg_file = """\"%YAML:1.0
work_time: 1000\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Config must contain field: Pipelines')


def test_error_pipelines_node_not_map():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Pipelines field must be a map')


def test_error_config_not_contain_pl():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:\" """

    exec_str = '{} --cfg={} --exec_list=PL2'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Pipelines must contain field: PL2')


def test_error_no_source():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('PL1 must contain field: source')


def test_error_source_no_name():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('source must contain field: name')


def test_error_source_no_latency():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('source must contain field: latency')


def test_error_source_no_output():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('source must contain field: output')


def test_error_source_output_no_dims():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('output must contain field: dims')


def test_error_source_output_no_precision():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('output must contain field: precision')


def test_error_no_nodes():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('PL1 must contain field: nodes')


def test_error_nodes_not_sequence():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('nodes in PL1 must be a sequence')


def test_error_node_no_name():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      -\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('node must contain field: name')


def test_error_node_no_type():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('node must contain field: type')


def test_error_node_unknown_type():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Unknown'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Unsupported node type: Unknown')


def test_error_node_dummy_no_time():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Node0 must contain field: time')


def test_error_node_dummy_not_positive_time():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: -0.2\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Node0 time must be positive')


def test_error_node_dummy_no_output():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Node0 must contain field: output')


def test_error_node_infer_no_model_path():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Infer'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    error_msg = """Path to OpenVINO model must be specified in either of two formats:
1.
  xml: path to *.xml
  bin: path to *.bin
2.
  blob: path to *.blob"""
    assert out.startswith(error_msg)


def test_error_node_infer_no_input_layers():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Infer'
        blob: model.blob
        device: 'CPU'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('Node0 must contain field: input_layers')


def test_error_node_infer_input_layers_are_empty():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Infer'
        blob: model.blob
        device: 'CPU'
        input_layers:
            \" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('input_layers in Node0 must be a sequence')


def test_error_node_infer_no_output_layers():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Infer'
        blob: model.blob
        device: 'CPU'
        input_layers:
          - 'layer_name'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('Node0 must contain field: output_layers')


def test_error_node_infer_output_layers_are_empty():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Infer'
        blob: model.blob
        device: 'CPU'
        input_layers:
          - 'layer_name'
        output_layers:\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('output_layers in Node0 must be a sequence')


def test_error_no_edges():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('PL1 must contain field: edges')


def test_error_edges_not_sequence():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('edges in PL1 must be a sequence')


def test_error_edges_no_from():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      -\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('edge must contain field: from')


def test_error_edges_no_to():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Node0'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('edge must contain field: to')


def test_error_edges_from_not_exists():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Node1'
        to: 'Node2'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('Failed to find node: Node1')


def test_error_edges_from_port_not_exists():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Node0:10'
        to: 'Node2'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('Failed to access node: Node0 by out port: 10')


def test_error_edges_to_not_exists():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Src'
        to: 'Node2'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('Failed to find node: Node2')


def test_error_edges_to_port_not_exists():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Src'
        to: 'Node0:3'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('Failed to access node: Node0 by in port: 3')


def test_error_connect_to_source():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Node0'
        to: 'Src'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('Failed to access node: Src by in port: 0')


def test_error_double_edge():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Src'
        to: 'Node0'
      - from: 'Src'
        to: 'Node0'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('Node: Node0 already connected by in port: 0')


def test_error_double_edge():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Src'
        to: 'Node0'
      - from: 'Src'
        to: 'Node0'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('Node: Node0 already connected by in port: 0')


def test_node_has_dangling_input():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'

      - name: 'Node1'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Node0'
        to: 'Node1'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)

    assert out.startswith('Node: Node0 in Pipeline: PL1 has dangling input by in port: 0')


def test_error_has_cycle_0():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node'
        type: 'Infer'
        blob: 'model.blob'
        device: 'CPU'
        input_layers:
          - 'in_layer_name_0'
          - 'in_layer_name_1'
        output_layers:
          - 'out_layer_name'
    edges:
      - from: 'Src'
        to: 'Node:0'
      - from: 'Node:0'
        to: 'Node:1'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Pipeline: PL1 has cyclic dependencies')


def test_error_has_cycle_0():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Infer'
        blob: 'model.blob'
        device: 'CPU'
        input_layers:
          - 'in_layer_name_0'
          - 'in_layer_name_1'
        output_layers:
          - 'out_layer_name'

      - name: 'Node1'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Src'
        to: 'Node0:0'
      - from: 'Node0:0'
        to: 'Node1:0'
      - from: 'Node1'
        to: 'Node0:1'\" """

    exec_str = '{} --cfg={}'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Pipeline: PL1 has cyclic dependencies')


def test_error_no_load_config_exists():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Src'
        to: 'Node0'\" """

    exec_str = '{} --cfg={} --load_config=not_existing.yml'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert 'Failed to load config: not_existing.yml' in out


def test_error_invalid_app_mode():
    cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
  PL1:
    source:
      name: 'Src'
      latency: 20
      output:
        dims: [1,2,3,4]
        precision: 'U8'
    nodes:
      - name: 'Node0'
        type: 'Dummy'
        time: 0.2
        output:
          dims: [1,2,3,4]
          precision: 'U8'
    edges:
      - from: 'Src'
        to: 'Node0'\" """

    exec_str = '{} --cfg={} --pl_mode=unknown'.format(pipeline_modeling_tool, cfg_file)
    out = get_output(exec_str)
    assert out.startswith('Unsupported PLMode: unknown\n'
                          'Please chose between: streaming and regular')


def test_error_invalid_pl_mode():
  cfg_file = """\"%YAML:1.0
work_time: 1000
Pipelines:
PL1:
  source:
    name: 'Src'
    latency: 20
    output:
      dims: [1,2,3,4]
      precision: 'U8'
  nodes:
    - name: 'Node0'
      type: 'Dummy'
      time: 0.2
      output:
        dims: [1,2,3,4]
        precision: 'U8'
  edges:
    - from: 'Src'
      to: 'Node0'\" """

  exec_str = '{} --cfg={} --app_mode=unknown'.format(pipeline_modeling_tool, cfg_file)
  out = get_output(exec_str)
  assert out.startswith('Unsupported AppMode: unknown\n'
                        'Please chose between: realtime and benchmark')
