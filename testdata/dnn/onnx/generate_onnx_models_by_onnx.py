from __future__ import print_function
import io
import numpy as np
import os.path
import onnx
import google.protobuf.text_format

from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from typing import Optional

def assertExpected(s):
    if not (isinstance(s, str) or (sys.version_info[0] == 2 and isinstance(s, unicode))):
        raise TypeError("assertExpected is strings only")

def assertONNXExpected(binary_pb):
    model_def = onnx.ModelProto.FromString(binary_pb)
    onnx.checker.check_model(model_def)
    # doc_string contains stack trace in it, strip it
    onnx.helper.strip_doc_string(model_def)
    assertExpected(google.protobuf.text_format.MessageToString(model_def, float_format='.15g'))
    return model_def

def save_data_and_model(name, input_np, output_np, onnx_model):
    print(name + " input has sizes",  input_np.shape)
    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, input_np.data)

    print(name + " output has sizes", output_np.shape)
    print()
    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output_np.data))

    models_files = os.path.join("models", name + ".onnx")

    onnx_model_pb = onnx._serialize(onnx_model)
    model_def = assertONNXExpected(onnx_model_pb)
    with open(models_files, 'wb') as file:
        file.write(model_def.SerializeToString())

np.random.seed(0)

# ########################## GEMM ##########################
# The original code is : https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/gemm.py
def gemm_reference_implementation(A: np.ndarray, B: np.ndarray, C: Optional[np.ndarray] = None, alpha: float = 1., beta: float = 1., transA: int = 0,
                                  transB: int = 0) -> np.ndarray:
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)

    Y = alpha * np.dot(A, B) + beta * C

    return Y

## gemm without transB
input_np = np.random.rand(2, 10).astype("float32")
inputs = [helper.make_tensor_value_info("input1", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_np.dtype], shape=input_np.shape)]

weight_np = np.random.rand(10, 3).astype("float32")
weight_tensor = helper.make_tensor('weight_tensor', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight_np.dtype], dims=weight_np.shape, vals=weight_np)
weight_node = helper.make_node("Constant", [], ["weight_node_out"], "input22", value=weight_tensor)

outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape=(2, 3))]

nodes = [weight_node, helper.make_node("Gemm", ["input1", "weight_node_out"], ["output"])]

graph = helper.make_graph(nodes,
                            "gemm_test",
                            inputs,
                            outputs)
gemm_model = helper.make_model(graph)
output_np = gemm_reference_implementation(input_np, weight_np)

save_data_and_model("gemm_no_transB", input_np, output_np, gemm_model)

## gemm with transB = 0

nodes2 = [weight_node, helper.make_node("Gemm", ["input1", "weight_node_out"], ["output"], transB=0)]
graph2 = helper.make_graph(nodes2,
                            "gemm_test",
                            inputs,
                            outputs)
gemm_model2 = helper.make_model(graph2)
output_np = gemm_reference_implementation(input_np, weight_np)

save_data_and_model("gemm_transB_0", input_np, output_np, gemm_model2)