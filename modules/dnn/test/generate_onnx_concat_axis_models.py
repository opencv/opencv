#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""Regenerate inline Concat importer test models without dependencies.

Run ``python3 modules/dnn/test/generate_onnx_concat_axis_models.py`` to update
the include beside this script. ``--models DIR`` also writes individual ONNX
files for inspection. The encoder writes only the protobuf fields these small
models need; field numbers follow ONNX's ModelProto, GraphProto, NodeProto,
TensorProto, and ValueInfoProto definitions.
"""

import argparse
import hashlib
import math
import struct
from pathlib import Path


HEADER = '''// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Minimal ONNX models for importer bounds regression tests.
// Regenerate with modules/dnn/test/generate_onnx_concat_axis_models.py.
'''


def varint(value):
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while value > 127:
        out.append((value & 0x7f) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def vint(number, value):
    return varint(number << 3) + varint(value)


def data(number, value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    return varint((number << 3) | 2) + varint(len(value)) + value


def tensor(name, dims, element_type, payload):
    # TensorProto: dims=1, data_type=2, name=8, raw_data=9.
    return (b''.join(vint(1, dimension) for dimension in dims)
            + vint(2, element_type) + data(8, name) + data(9, payload))


def value_info(name, element_type, dims):
    # ValueInfoProto -> TypeProto -> TypeProto.Tensor -> TensorShapeProto.
    shape = b''.join(data(1, vint(1, dimension)) for dimension in dims)
    tensor_type = vint(1, element_type) + data(2, shape)
    return data(1, name) + data(2, data(1, tensor_type))


def make_model(graph_name, op_type, domain, axis, names, initializers,
               output_type, output_dims):
    # AttributeProto: name=1, i=3, type=20 (INT=2).
    axis_attribute = data(1, 'axis') + vint(3, axis) + vint(20, 2)
    # NodeProto: input=1, output=2, op_type=4, attribute=5, domain=7.
    node = (b''.join(data(1, name) for name in names)
            + data(2, 'out') + data(4, op_type) + data(5, axis_attribute))
    if domain:
        node += data(7, domain)
    # GraphProto: node=1, name=2, initializer=5, output=12.
    graph = (data(1, node) + data(2, graph_name)
             + b''.join(data(5, item) for item in initializers)
             + data(12, value_info('out', output_type, output_dims)))
    # ModelProto: ir_version=1, graph=7, opset_import=8.
    default_opset = (data(1, '') if domain else b'') + vint(2, 13)
    model = vint(1, 8) + data(7, graph) + data(8, default_opset)
    if domain:
        model += data(8, data(1, domain) + vint(2, 1))
    return model


def qconcat(axis, graph_name, shapes, values, output_dims):
    # FLOAT=1 and INT8=3. Scalar quantization parameters have no dims.
    f32_one = struct.pack('<f', 1.0)
    initializers = [tensor('out_scale', [], 1, f32_one),
                    tensor('out_zp', [], 3, b'\x00')]
    names = ['out_scale', 'out_zp']
    for index, (shape, value) in enumerate(zip(shapes, values)):
        dname, sname, zname = 'd%d' % index, 's%d' % index, 'z%d' % index
        initializers.extend([tensor(dname, shape, 3, bytes([value]) * math.prod(shape)),
                             tensor(sname, [], 1, f32_one),
                             tensor(zname, [], 3, b'\x00')])
        names.extend([dname, sname, zname])
    return make_model(graph_name, 'QLinearConcat', 'com.microsoft', axis,
                      names, initializers, 3, output_dims)


def concat(axis, first, second, output_dims):
    initializers = [tensor('d0', first, 1, struct.pack('<f', 1.0)),
                    tensor('d1', second, 1, struct.pack('<f', 2.0))]
    return make_model('plain_concat_bounds', 'Concat', '', axis,
                      ['d0', 'd1'], initializers, 1, output_dims)


def padding_concat(second_dims):
    # Graph inputs take the variable-input path into ConcatLayer::getMemoryShapes.
    axis = data(1, 'axis') + vint(3, 1) + vint(20, 2)
    padding = data(1, 'padding') + vint(3, 1) + vint(20, 2)
    node = (data(1, 'x0') + data(1, 'x1') + data(2, 'out')
            + data(4, 'Concat') + data(5, axis) + data(5, padding))
    graph = (data(1, node) + data(2, 'concat_padding')
             + data(11, value_info('x0', 1, [1, 1, 1, 1]))
             + data(11, value_info('x1', 1, second_dims))
             + data(12, value_info('out', 1, [1, 2, 1, 1])))
    return vint(1, 8) + data(7, graph) + data(8, vint(2, 13))


def cases():
    rank8 = [1] * 8
    rank16 = [1] * 16
    rank2 = [1, 4]
    qwide_output = [2] + [1] * 15
    qshort_output = [1, 8]
    plain_rank2 = [1, 1]
    return [
        ('onnxMixedRankControl', 'control-mixed-rank.onnx',
         'QLinearConcat axis 0, INT8 ranks 8 and 16, both 1-byte payloads.',
         qconcat(0, 'qconcat_mixed_rank_control', [rank8, rank16], [0x31, 0x32], qwide_output)),
        ('onnxNegativeAxisControl', 'control-negative-axis.onnx',
         'QLinearConcat axis -1, INT8 ranks 2 and 2, each 4-byte payload.',
         qconcat(-1, 'qconcat_negative_axis_control', [rank2, rank2], [0x11, 0x12], qshort_output)),
        ('onnxMixedRankInvalid', 'invalid-source-rank.onnx',
         'QLinearConcat axis -2, INT8 ranks 8 and 16; axis 14 misses rank 8.',
         qconcat(-2, 'qconcat_invalid_source_rank', [rank8, rank16], [0x31, 0x32], qwide_output)),
        ('onnxOutOfRangeAxis', 'invalid-axis-range.onnx',
         'QLinearConcat axis 9999, INT8 ranks 2 and 2.',
         qconcat(9999, 'qconcat_invalid_axis_range', [rank2, rank2], [0x11, 0x12], qshort_output)),
        ('onnxPlainMixedRankControl', 'plain-control-mixed-rank-axis0.onnx',
         'Concat axis 0, FLOAT32 ranks 16 and 2, 4 bytes per initializer.',
         concat(0, rank16, plain_rank2, [2] + [1] * 15)),
        ('onnxPlainSameRankControl', 'plain-control-same-rank-axis15.onnx',
         'Concat axis 15, FLOAT32 ranks 16 and 16, 4 bytes per initializer.',
         concat(15, rank16, rank16, [1] * 15 + [2])),
        ('onnxPlainMixedRankInvalid', 'plain-invalid-mixed-rank-axis15.onnx',
         'Concat axis 15, FLOAT32 ranks 16 and 2; axis misses rank 2.',
         concat(15, rank16, plain_rank2, [1] * 15 + [2])),
        ('onnxPaddingSameRankControl', 'padding-control-same-rank.onnx',
         'Concat graph inputs of rank 4 and 4, axis 1, padding enabled.',
         padding_concat([1, 1, 1, 1])),
        ('onnxPaddingMixedRankInvalid', 'padding-crafted-mixed-rank.onnx',
         'Concat graph inputs of rank 4 and 2, axis 1, padding enabled.',
         padding_concat([1, 1])),
    ]


def render_include(models):
    sections = [HEADER]
    for symbol, _, comment, blob in models:
        lines = []
        for offset in range(0, len(blob), 12):
            lines.append('    ' + ', '.join('0x%02x' % byte for byte in blob[offset:offset + 12]) + ',')
        sections.append('// ' + comment + '\nstatic const unsigned char ' + symbol
                        + '[] = {\n' + '\n'.join(lines) + '\n};\n')
    return '\n'.join(sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('onnx_qconcat_axis_models.inc'))
    parser.add_argument('--models', type=Path)
    args = parser.parse_args()
    models = cases()
    args.out.write_text(render_include(models), encoding='utf-8')
    if args.models:
        args.models.mkdir(parents=True, exist_ok=True)
        for _, filename, _, blob in models:
            (args.models / filename).write_bytes(blob)
    for symbol, _, _, blob in models:
        print('%s  %s' % (hashlib.sha256(blob).hexdigest(), symbol))


if __name__ == '__main__':
    main()
