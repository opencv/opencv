import os
import numpy as np
import onnx
import onnxscript as ost
from onnxscript import opset19 as op # opset19 is the lastest by 202309

np.random.seed(0)

def make_model_and_data(model, *args, **kwargs):
    name = model._name

    # TODO: support multiple outputs
    output = model(*args) # eager mode

    # Save model
    model_proto = model.to_model_proto()
    try:
        onnx.checker.check_model(model_proto)
    except onnx.checker.ValidationError as e:
        print(f"Model {name} is invalid: {e}. Skipping ...")
        return False
    else:
        save_path = "./models/{}.onnx".format(name)
        print(f"Model {name} is valid! Saved to {save_path}")
        model_proto_ = onnx.shape_inference.infer_shapes(model_proto)
        onnx.save(model_proto_, save_path)

    # Save inputs and output
    inputs = args
    if "force_saving_input_as_dtype_float32" in kwargs and kwargs["force_saving_input_as_dtype_float32"]:
        inputs = []
        for input in args:
            inputs.append(input.astype(np.float32))
    if len(args) == 1:
        input_file = os.path.join("data", "input_" + name)
        np.save(input_file, inputs[0])
    else:
        for idx, input in enumerate(inputs, start=0):
            input_files = os.path.join("data", "input_" + name + "_" + str(index))
            np.save(input_files, input)
    if "force_saving_output_as_dtype_float32" in kwargs and kwargs["force_saving_output_as_dtype_float32"]:
        output = output.astype(np.float32)
    output_files = os.path.join("data", "output_" + name)
    np.save(output_files, output)

'''
    It builds a model with two Gather ops sharing a single same indices:

    [Input] -> Gather(indices=0) -> Gather(indices=0) -> [Output]

    , where the two indices constants have the same name.
'''
@ost.script()
def gather_shared_indices(x: ost.FLOAT[2, 1, 3, 4]) -> ost.FLOAT[3, 4]:
    indices = op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [], np.array([0], dtype=np.int64)))
    y0 = op.Gather(x, indices, axis=0)
    y1 = op.Gather(y0, indices, axis=0)
    return y1
make_model_and_data(gather_shared_indices, np.random.rand(2, 1, 3, 4).astype(np.float32))

'''
    [Input] -> Greater(B=61) -> [Output]
                        \
                        dtype=np.int64
'''
@ost.script()
def greater_input_dtype_int64(x: ost.FLOAT[27, 9]) ->ost.BOOL[27, 9]:
    y = op.Greater(x, op.Constant(value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [], np.array([61], dtype=np.int64))))
    return y
make_model_and_data(greater_input_dtype_int64, np.random.randint(0, 100, size=[27, 9], dtype=np.int64), force_saving_input_as_dtype_float32=True, force_saving_output_as_dtype_float32=True)
