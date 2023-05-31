from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import onnx # version >= 1.12.0
import onnxruntime as rt
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

class DataReader(CalibrationDataReader):
    def __init__(self, model_path, batchsize=5):
        sess = rt.InferenceSession(model_path, None)
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        calib_data = np.random.uniform(-1, 1, size=[batchsize] + input_shape[1:]).astype("float32")
        self.enum_data_dicts = iter([{input_name: np.expand_dims(x, axis=0)} for x in calib_data])

    def get_next(self):
        return next(self.enum_data_dicts, None)

def quantize_and_save_model(name, input, model, act_type="uint8", wt_type="uint8", per_channel=False, ops_version = 13, quanFormat=QuantFormat.QOperator):
    float_model_path = os.path.join("models", "dummy.onnx")
    quantized_model_path = os.path.join("models", name + ".onnx")
    type_dict = {"uint8" : QuantType.QUInt8, "int8" : QuantType.QInt8}

    model.eval()
    torch.onnx.export(model, input, float_model_path, export_params=True, opset_version=ops_version)

    dr = DataReader(float_model_path)
    quantize_static(float_model_path, quantized_model_path, dr, quant_format=quanFormat, per_channel=per_channel,
                    activation_type=type_dict[act_type], weight_type=type_dict[wt_type])

    os.remove(float_model_path)
    # os.remove(os.path.join("models", "dummy-opt.onnx"))
    # os.remove("augmented_model.onnx")
    
    sess = rt.InferenceSession(quantized_model_path, None)
    input = np.random.uniform(-1, 1, sess.get_inputs()[0].shape).astype("float32")
    output = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name : input})[0]

    print(name + " input has sizes",  input.shape)
    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, input.data)

    print(name + " output has sizes", output.shape)
    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output.data))

torch.manual_seed(0)
np.random.seed(0)

input = Variable(torch.randn(1, 3, 10, 10))
conv = nn.Conv2d(3, 5, kernel_size=3, stride=2, padding=1)
# generate QOperator qunatized model
quantize_and_save_model("quantized_conv_uint8_weights", input, conv)
quantize_and_save_model("quantized_conv_int8_weights", input, conv, wt_type="int8")
quantize_and_save_model("quantized_conv_per_channel_weights", input, conv, per_channel=True)

# generate QDQ qunatized model
quantize_and_save_model("quantized_conv_uint8_weights_qdq", input, conv, quanFormat=QuantFormat.QDQ)
quantize_and_save_model("quantized_conv_int8_weights_qdq", input, conv, wt_type="int8", quanFormat=QuantFormat.QDQ)
quantize_and_save_model("quantized_conv_per_channel_weights_qdq", input, conv, per_channel=True, quanFormat=QuantFormat.QDQ)

input = Variable(torch.randn(1, 3))
linear = nn.Linear(3, 4, bias=True)
quantize_and_save_model("quantized_matmul_uint8_weights", input, linear)
quantize_and_save_model("quantized_matmul_int8_weights", input, linear, wt_type="int8")
quantize_and_save_model("quantized_matmul_per_channel_weights", input, linear, wt_type="int8", per_channel=True)

class MatMul(nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, x):
      y = torch.t(x)
      return x @ y

model = MatMul()
input = Variable(torch.randn(1, 4))
quantize_and_save_model("quantized_matmul_variable_inputs", input, model, wt_type="int8")

class Eltwise(nn.Module):

    def __init__(self):
        super(Eltwise, self).__init__()
        self.squeeze1 = nn.Linear(2, 2, bias=False)
        self.squeeze2 = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        y = self.squeeze1(x)
        z = self.squeeze2(x)
        return z + x * y

input = Variable(torch.randn(1, 2))
model = Eltwise()
quantize_and_save_model("quantized_eltwise", input, model, wt_type="int8")

class EltwiseScalars(nn.Module):

    def __init__(self):
        super(EltwiseScalars, self).__init__()

    def forward(self, x):
        return -1.5 + 2.5 * x

input = Variable(torch.randn(1, 2, 3, 4))
model = EltwiseScalars()
quantize_and_save_model("quantized_eltwise_scalar", input, model)

class EltwiseBroadcast(nn.Module):

    def __init__(self):
        super(EltwiseBroadcast, self).__init__()
        self.a = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
        self.b = torch.tensor([[-0.9, 0.8]], dtype=torch.float32)

    def forward(self, x):
        return self.a[:, :, None, None] + self.b[:, :, None, None] * x

input = Variable(torch.randn(1, 2, 3, 4))
model = EltwiseBroadcast()
quantize_and_save_model("quantized_eltwise_broadcast", input, model)

input = Variable(torch.randn(1, 2, 3, 4))
leaky_relu = nn.LeakyReLU(negative_slope=0.25)
quantize_and_save_model("quantized_leaky_relu", input, leaky_relu)

input = Variable(torch.randn(1, 10))
sigmoid = nn.Sigmoid()
quantize_and_save_model("quantized_sigmoid", input, sigmoid)

input = Variable(torch.randn(1, 3, 6, 6))
max_pool = nn.Sequential(nn.Sigmoid(),
                         nn.MaxPool2d(kernel_size=2, stride=2))
quantize_and_save_model("quantized_maxpool", input, max_pool)

input = Variable(torch.randn(1, 3, 6, 6))
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
quantize_and_save_model("quantized_avgpool", input, avg_pool)

class SplitMax(nn.Module):

    def __init__(self):
        super(SplitMax, self).__init__()

    def forward(self, x):
        first, second = torch.split(x, (2, 4), dim=1)
        second, third = torch.split(second, (2, 2), dim=1)
        return torch.max(first, torch.max(second, third))

model = SplitMax()
input = Variable(torch.randn(1, 6, 2, 3))
quantize_and_save_model("quantized_split", input, model)

input = Variable(torch.randn(1, 3, 4, 5))
model = nn.Sequential(nn.LeakyReLU(negative_slope=0.1),
                      nn.ZeroPad2d((2, 1, 2, 1)),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReflectionPad2d(1))
quantize_and_save_model("quantized_padding", input, model)

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        return torch.reshape(self.relu(x), (1, -1))

input = Variable(torch.randn(1, 3, 4, 4))
model = Reshape()
quantize_and_save_model("quantized_reshape", input, model)

class Transpose(nn.Module):

    def __init__(self):
        super(Transpose, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return torch.t(self.relu(x))

input = Variable(torch.randn(1, 3))
model = Transpose()
quantize_and_save_model("quantized_transpose", input, model)

class Squeeze(nn.Module):

    def __init__(self):
        super(Squeeze, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return torch.squeeze(self.sigmoid(x), dim=2)

input = Variable(torch.randn(1, 3, 1, 4))
model = Squeeze()
quantize_and_save_model("quantized_squeeze", input, model)

class Unsqueeze(nn.Module):

    def __init__(self):
        super(Unsqueeze, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return torch.unsqueeze(self.sigmoid(x), dim=1)

input = Variable(torch.randn(1, 2, 3))
model = Unsqueeze()
quantize_and_save_model("quantized_unsqueeze", input, model)

input = Variable(torch.randn(1, 2, 3, 4))
resize = nn.Sequential(nn.LeakyReLU(negative_slope=0.1),
                       nn.Upsample(scale_factor=2, mode='nearest'))
quantize_and_save_model("quantized_resize_nearest", input, resize)

input = Variable(torch.randn(1, 2, 3, 4))
resize = nn.Sequential(nn.Sigmoid(),
                       nn.Upsample(size=[6, 8], mode='bilinear'))
quantize_and_save_model("quantized_resize_bilinear", input, resize)

input = Variable(torch.randn(1, 2, 3, 4))
resize = nn.Sequential(nn.Sigmoid(),
                       nn.Upsample(size=[6, 8], mode='bilinear', align_corners=True))
quantize_and_save_model("quantized_resize_bilinear_align", input, resize)

class Concatenation(nn.Module):

    def __init__(self):
        super(Concatenation, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        y = self.relu(x)
        return torch.cat([x, y], axis=1)

input = Variable(torch.randn(1, 2, 2, 2))
model = Concatenation()
quantize_and_save_model("quantized_concat", input, model)

class ConcatConstBlob(nn.Module):

    def __init__(self):
        super(ConcatConstBlob, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        y = torch.tensor([[[[0.1, -0.2], [-0.3, 0.4]]]], dtype=torch.float32)
        return torch.cat([x, y], axis=1)

input = Variable(torch.randn(1, 2, 2, 2))
model = ConcatConstBlob()
quantize_and_save_model("quantized_concat_const_blob", input, model)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(-1, 16*5*5)

model = nn.Sequential(
                nn.Conv2d(3,6,5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10)
                )
input = Variable(torch.randn(1, 3, 32, 32))
quantize_and_save_model("quantized_constant", input, model, wt_type="int8", per_channel=True)

class Gemm(nn.Module):
    def forward(self, x):
        mat1 =torch.ones(3, 3)
        return torch.mm(x, mat1)

input = Variable(torch.randn(1, 3))
model = Gemm()
quantize_and_save_model("quantized_gemm", input, model, act_type="int8", wt_type="int8", per_channel=False)

shape = [1, 10, 3]
axis = 1
name = "qlinearsoftmax_v13"
input = torch.rand(shape)
model = nn.Softmax(dim=axis)
# NOTE: this needs latest pytorch to have opset>= 13
quantize_and_save_model(name, input, model, act_type="int8", wt_type="int8", per_channel=False)

def generate_qlinearsoftmax(shape=[1, 10, 3], axis=1, opset=11, name="qlinearsoftmax", previous_name="qlinearsoftmax_v13"):
    from onnx import helper, TensorProto

    # get scales and zero points
    previous_model = onnx.load("models/{}.onnx".format(previous_name))
    previous_model_graph = previous_model.graph
    previous_model_nodes = previous_model_graph.node
    scales = []
    zero_points = []
    def get_val_from_initializers(initializers, name):
        for init in initializers:
            if init.name == name:
                if init.int32_data:
                    return init.int32_data[0]
                elif init.float_data:
                    return init.float_data[0]
                else:
                    raise NotImplementedError()
    scales.append(get_val_from_initializers(previous_model_graph.initializer, previous_model_nodes[1].input[1]))
    zero_points.append(get_val_from_initializers(previous_model_graph.initializer, previous_model_nodes[1].input[2]))
    scales.append(get_val_from_initializers(previous_model_graph.initializer, previous_model_nodes[1].input[3]))
    zero_points.append(get_val_from_initializers(previous_model_graph.initializer, previous_model_nodes[1].input[4]))


    def make_initializers(input_output_names, scales, zero_points):
        names = [
            input_output_names[0] + "_scale",
            input_output_names[0] + "_zero_point",
            input_output_names[1] + "_scale",
            input_output_names[1] + "_zero_point",
        ]

        initializers = []
        initializers.append(helper.make_tensor(names[0], TensorProto.FLOAT, [], np.array([scales[0]], dtype=np.float32)))
        initializers.append(helper.make_tensor(names[1], TensorProto.INT8, [], np.array([zero_points[0]], dtype=np.int8)))
        initializers.append(helper.make_tensor(names[2], TensorProto.FLOAT, [], np.array([scales[1]], dtype=np.float32)))
        initializers.append(helper.make_tensor(names[3], TensorProto.INT8, [], np.array([zero_points[1]], dtype=np.int8)))

        return initializers

    def make_quantize_dequantize(input_name, output_name, shape, dequantize=False):
        input_names = [
            input_name,
            input_name.split('_')[0] + "_scale",
            input_name.split('_')[0] + "_zero_point",
        ]
        output_names = [
            output_name
        ]

        inputs = []
        if dequantize:
            inputs.append(helper.make_tensor_value_info(input_names[0], TensorProto.INT8, shape))
        else:
            inputs.append(helper.make_tensor_value_info(input_names[0], TensorProto.FLOAT, shape))
        inputs.append(helper.make_tensor_value_info(input_names[1], TensorProto.FLOAT, []))
        inputs.append(helper.make_tensor_value_info(input_names[2], TensorProto.INT8,  []))

        outputs = []
        if dequantize:
            outputs.append(helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape))
        else:
            outputs.append(helper.make_empty_tensor_value_info(output_names[0]))

        node_type = "DequantizeLinear" if dequantize else "QuantizeLinear"
        node = helper.make_node(node_type, input_names, output_names)

        return node, inputs[0], outputs[0]

    def make_qlinearsoftmax(input_name, output_name, axis, opset):
        input_names = [
            input_name,
            input_name.split('_')[0] + "_scale",
            input_name.split('_')[0] + "_zero_point",
            output_name.split('_')[0] + "_scale",
            output_name.split('_')[0] + "_zero_point",
        ]
        output_names = [
            output_name
        ]

        inputs = []
        inputs.append(helper.make_empty_tensor_value_info(input_names[0]))
        inputs.append(helper.make_tensor_value_info(input_names[1], TensorProto.FLOAT, []))
        inputs.append(helper.make_tensor_value_info(input_names[2], TensorProto.INT8, []))
        inputs.append(helper.make_tensor_value_info(input_names[3], TensorProto.FLOAT, []))
        inputs.append(helper.make_tensor_value_info(input_names[4], TensorProto.INT8, []))

        outputs = []
        outputs.append(helper.make_empty_tensor_value_info(output_names[0]))

        node = helper.make_node("QLinearSoftmax", input_names, output_names, domain="com.microsoft", axis=axis, opset=opset)

        return node, inputs[0], outputs[0]

    input_names = ["input", "input_quantized", "output_quantized"]
    output_names = ["input_quantized", "output_quantized", "output"]

    shared_initializers = make_initializers([input_names[0], output_names[-1]], scales, zero_points)
    node_quantize, graph_input, _ = make_quantize_dequantize(input_names[0], output_names[0], shape)
    node_qsoftmax, _, _ = make_qlinearsoftmax(input_names[1], output_names[1], axis, opset)
    node_dequantize, _, graph_output = make_quantize_dequantize(input_names[2], output_names[2], shape, dequantize=True)

    # create graph
    name = "{}_v{}".format(name, opset)
    graph = helper.make_graph(
        [node_quantize, node_qsoftmax, node_dequantize],
        name,
        [graph_input],
        [graph_output],
        shared_initializers,
    )

    # create model
    model = helper.make_model(graph, producer_name="github.com/opencv/opencv_extra")
    model.opset_import.extend([helper.make_opsetid("com.microsoft", 1)])
    # ignore model check to create non-standard operators
    model_path = "models/{}.onnx".format(name)
    onnx.save(model, model_path)
    print("model is saved to {}".format(model_path))

    sess = rt.InferenceSession(model_path)
    # input = np.random.uniform(-1, 1, sess.get_inputs()[0].shape).astype("float32")
    input = np.load("data/input_{}.npy".format(previous_name)).astype(np.float32)
    output = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name : input})[0]

    print(name + " input has sizes",  input.shape)
    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, input.data)

    print(name + " output has sizes", output.shape)
    output_files = os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output.data))

generate_qlinearsoftmax(shape, axis, name="qlinearsoftmax", previous_name=name)
