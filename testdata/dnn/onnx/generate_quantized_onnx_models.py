from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import onnx
import onnxruntime as rt
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class DataReader(CalibrationDataReader):
    def __init__(self, model_path, batchsize=5):
        sess = rt.InferenceSession(model_path, None)
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        calib_data = np.random.uniform(-1, 1, size=[batchsize] + input_shape[1:]).astype("float32")
        self.enum_data_dicts = iter([{input_name: np.expand_dims(x, axis=0)} for x in calib_data])

    def get_next(self):
        return next(self.enum_data_dicts, None)

def quantize_and_save_model(name, input, model, act_type="uint8", wt_type="uint8", per_channel=False):
    float_model_path = os.path.join("models", "dummy.onnx")
    quantized_model_path = os.path.join("models", name + ".onnx")
    type_dict = {"uint8" : QuantType.QUInt8, "int8" : QuantType.QInt8}

    model.eval()
    torch.onnx.export(model, input, float_model_path, export_params=True, opset_version=12)

    dr = DataReader(float_model_path)
    quantize_static(float_model_path, quantized_model_path, dr, per_channel=per_channel,
                    activation_type=type_dict[act_type], weight_type=type_dict[wt_type])

    os.remove(float_model_path)
    os.remove(os.path.join("models", "dummy-opt.onnx"))
    os.remove("augmented_model.onnx")
    
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
quantize_and_save_model("quantized_conv_uint8_weights", input, conv)
quantize_and_save_model("quantized_conv_int8_weights", input, conv, wt_type="int8")
quantize_and_save_model("quantized_conv_per_channel_weights", input, conv, per_channel=True)

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