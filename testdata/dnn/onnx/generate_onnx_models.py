from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path
import onnx
import google.protobuf.text_format
import io


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


def export_to_string(model, inputs, version=None):
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, inputs, f, export_params=True, opset_version=version)
    return f.getvalue()


def save_data_and_model(name, input, model, version=None):
    model.eval()
    print(name + " input has sizes",  input.shape)
    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, input.data)

    output = model(input)

    print(name + " output has sizes", output.shape)
    print()
    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output.data))

    models_files = os.path.join("models", name + ".onnx")

    onnx_model_pb = export_to_string(model, input, version)
    model_def = assertONNXExpected(onnx_model_pb)
    with open(models_files, 'wb') as file:
        file.write(model_def.SerializeToString())

def save_onnx_data_and_model(input, output, name, operation, *args, **kwargs):
    print(name + " input has sizes",  input.shape)
    input_files = os.path.join("data", "input_" + name)
    input = input.astype(np.float32)
    np.save(input_files, input.data)

    print(name + " output has sizes", output.shape)
    print()
    output_files =  os.path.join("data", "output_" + name)
    output = output.astype(np.float32)
    np.save(output_files, np.ascontiguousarray(output.data))

    models_files = os.path.join("models", name + ".onnx")
    X = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, input.shape)
    Y = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, output.shape)
    node = onnx.helper.make_node(operation, inputs=['input'], outputs=['output'], *args, **kwargs)
    graph = onnx.helper.make_graph([node], name, [X], [Y])
    model = onnx.helper.make_model(graph, producer_name=name)
    onnx.save(model, models_files)

torch.manual_seed(0)
np.random.seed(0)

input = Variable(torch.randn(1, 3, 10, 9))
max_pool = nn.MaxPool2d(kernel_size=(5,3), stride=3, padding=1, dilation=1)
save_data_and_model("maxpooling", input, max_pool)


input = Variable(torch.randn(1, 3, 10, 10))
conv = nn.Conv2d(3, 5, kernel_size=5, stride=2, padding=1)
save_data_and_model("convolution", input, conv)

input = Variable(torch.randn(1, 3, 10, 10))
deconv = nn.ConvTranspose2d(3, 5, kernel_size=5, stride=2, padding=1)
save_data_and_model("deconvolution", input, deconv)

input = Variable(torch.randn(2, 3))
linear = nn.Linear(3, 4, bias=True)
linear.eval()
save_data_and_model("linear", input, linear)

input = Variable(torch.randn(2, 3, 12, 18))
maxpooling_sigmoid = nn.Sequential(
          nn.MaxPool2d(kernel_size=4, stride=2, padding=(1, 2), dilation=1),
          nn.Sigmoid()
        )
save_data_and_model("maxpooling_sigmoid", input, maxpooling_sigmoid)


input = Variable(torch.randn(1, 3, 10, 20))
conv2 = nn.Sequential(
          nn.Conv2d(3, 6, kernel_size=(5,3), stride=1, padding=1),
          nn.Conv2d(6, 4, kernel_size=5, stride=2, padding=(0,2))
          )
save_data_and_model("two_convolution", input, conv2)

input = Variable(torch.randn(1, 3, 10, 20))
deconv2 = nn.Sequential(
    nn.ConvTranspose2d(3, 6, kernel_size=(5,3), stride=1, padding=1),
    nn.ConvTranspose2d(6, 4, kernel_size=5, stride=2, padding=(0,2))
)
save_data_and_model("two_deconvolution", input, deconv2)

input = Variable(torch.randn(2, 3, 12, 9))
maxpool2 = nn.Sequential(
           nn.MaxPool2d(kernel_size=5, stride=1, padding=0, dilation=1),
           nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1)
           )
save_data_and_model("two_maxpooling", input, maxpool2)


input = Variable(torch.randn(1, 2, 10, 10))
relu = nn.ReLU(inplace=True)
save_data_and_model("ReLU", input, relu)


input = Variable(torch.randn(2, 3))
dropout = nn.Dropout()
dropout.eval()
save_data_and_model("dropout", input, dropout)


input = Variable(torch.randn(1, 3, 7, 5))
ave_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
save_data_and_model("average_pooling", input, ave_pool)

input = torch.randn(2, 4, 2, 3)
batch_norm = nn.BatchNorm2d(4)
batch_norm.eval()
save_data_and_model("batch_norm", input, batch_norm)


class Concatenation(nn.Module):

    def __init__(self):
        super(Concatenation, self).__init__()
        self.squeeze1 = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)
        self.squeeze2 = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.squeeze1(x)
        y = self.squeeze2(x)
        return torch.cat([x, y])


input = Variable(torch.randn(1, 2, 2, 2))
model = Concatenation()
model.eval()
save_data_and_model("concatenation", input, model)


class Mul(nn.Module):

    def __init__(self):
        super(Mul, self).__init__()
        self.squeeze1 = nn.Linear(2, 2, bias=True)

    def forward(self, x):
        x = self.squeeze1(x)
        y = self.squeeze1(x)
        return x * y


input = Variable(torch.randn(2, 2))
model = Mul()
save_data_and_model("mul", input, model)


def save_data_and_model_multy_inputs(name, model, *args, **kwargs):
    for index, input in enumerate(args, start=0):
        input_files = os.path.join("data", "input_" + name + "_" + str(index))
        np.save(input_files, input)

    output = model(*args)
    print(name + " output has sizes", output.shape)
    print()
    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, output.data)

    models_files = os.path.join("models", name + ".onnx")

    onnx_model_pb = export_to_string(model, (args), version=kwargs.get('version', None))
    model_def = assertONNXExpected(onnx_model_pb)
    with open(models_files, 'wb') as file:
        file.write(model_def.SerializeToString())


class MultyInputs(nn.Module):

    def __init__(self):
        super(MultyInputs, self).__init__()
        self.squeeze1 = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)
        self.squeeze2 = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        x = self.squeeze1(x)
        y = self.squeeze2(y)
        return x + y

input1 = Variable(torch.randn(1, 2, 2, 2))
input2 = Variable(torch.randn(1, 2, 2, 2))
model = MultyInputs()
save_data_and_model_multy_inputs("multy_inputs", model, input1, input2)

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
save_data_and_model("constant", input, model)


class Transpose(nn.Module):

    def __init__(self):
        super(Transpose, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        return torch.t(x)


input = Variable(torch.randn(2, 3))
model = Transpose()
save_data_and_model("transpose", input, model)

input = Variable(torch.randn(1, 2, 3, 4))
pad = nn.ZeroPad2d((4,3, 2,1))  # left,right, top,bottom
save_data_and_model("padding", input, pad)


class DynamicReshapeNet(nn.Module):
    def __init__(self):
        super(DynamicReshapeNet, self).__init__()

    def forward(self, image):
        batch_size = image.size(0)
        channels = image.size(1)
        image = image.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, channels)
        return image

input = Variable(torch.randn(1, 2, 3, 4))
model = DynamicReshapeNet()
save_data_and_model("dynamic_reshape", input, model)

input = Variable(torch.randn(1, 2, 3, 4))
resize = nn.Upsample(scale_factor=2, mode='nearest')
save_data_and_model("resize_nearest", input, resize)

input = Variable(torch.randn(1, 2, 3, 4))
resize = nn.Upsample(size=[6, 8], mode='bilinear')
save_data_and_model("resize_bilinear", input, resize)

if torch.__version__ == '1.4.0':
    input = Variable(torch.randn(1, 3, 4, 5))
    upsample_unfused = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.Upsample(scale_factor=2, mode='nearest')
            )
    save_data_and_model("upsample_unfused_opset9_torch1.4", input, upsample_unfused)

    input = Variable(torch.randn(1, 3, 4, 5))
    resize_nearest_unfused = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.Upsample(scale_factor=2, mode='nearest')
            )
    save_data_and_model("resize_nearest_unfused_opset11_torch1.4", input, resize_nearest_unfused, 11)

    input = Variable(torch.randn(1, 3, 4, 5))
    resize_bilinear_unfused = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
    save_data_and_model("resize_bilinear_unfused_opset11_torch1.4", input, resize_bilinear_unfused, 11)


if torch.__version__ == '1.2.0':
    input = Variable(torch.randn(1, 2, 3, 4))
    resize_nearest_unfused = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Upsample(scale_factor=2, mode='nearest')
            )
    save_data_and_model("upsample_unfused_torch1.2", input, resize_nearest_unfused)

class Unsqueeze(nn.Module):

    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return torch.unsqueeze(x, dim=1)

input = Variable(torch.randn(1, 2, 3))
model = Unsqueeze()
model.eval()
save_data_and_model("unsqueeze", input, model)

input = Variable(torch.randn(1, 2, 4, 5))
deconv_adjpad2d = nn.ConvTranspose2d(2, 3, (3, 2), stride=(1, 2), padding=(1, 2), output_padding=(0, 1))
save_data_and_model("deconv_adjpad_2d", input, deconv_adjpad2d)

input = Variable(torch.randn(1, 2, 3, 4, 5))
conv3d = nn.Conv3d(2, 3, (2, 3, 2), stride=(1, 1, 1), padding=(0, 0, 0), groups=1, dilation=(1, 1, 1), bias=False)
save_data_and_model("conv3d", input, conv3d)

input = Variable(torch.randn(1, 2, 3, 4, 5))
conv3d = nn.Conv3d(2, 3, (2, 3, 3), stride=(1, 2, 3), padding=(0, 1, 2), groups=1, dilation=(1, 2, 3), bias=True)
save_data_and_model("conv3d_bias", input, conv3d)

input = torch.randn(1, 2, 3, 4, 6)
maxpool3d = nn.MaxPool3d((3, 2, 5), stride=(2, 1, 2), padding=(1, 0, 2))
save_data_and_model("max_pool3d", input, maxpool3d)

input = torch.randn(1, 2, 3, 5, 6)
avepool3d = nn.AvgPool3d((3, 4, 3), stride=(1, 2, 3), padding=(1, 2, 0))
save_data_and_model("ave_pool3d", input, avepool3d)

input = Variable(torch.randn(1, 2, 3, 4, 5))
conv3d = nn.BatchNorm3d(2)
save_data_and_model("batch_norm_3d", input, conv3d)

class Softmax(nn.Module):

    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(x)

input = torch.randn(2, 3)
model = Softmax()
save_data_and_model("softmax", input, model)

class LogSoftmax(nn.Module):

    def __init__(self):
        super(LogSoftmax, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.log_softmax(x)

input = torch.randn(2, 3)
model = LogSoftmax()
save_data_and_model("log_softmax", input, model)

class Slice(nn.Module):

    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, x):
        return x[..., 1:-1, 0:3]

input = Variable(torch.randn(1, 2, 4, 4))
model = Slice()
save_data_and_model("slice", input, model)
save_data_and_model("slice_opset_11", input, model, opset_version=11)

class Eltwise(nn.Module):

    def __init__(self):
        super(Eltwise, self).__init__()

    def forward(self, x):
        return x + 2.7 * x

input = Variable(torch.randn(1, 1, 2, 3, 4))
model = Eltwise()
save_data_and_model("eltwise3d", input, model)

class InstanceNorm(nn.Module):

    def __init__(self):
        super(InstanceNorm, self).__init__()
        self.inorm2 = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        x = self.inorm2(x)
        return x

input = Variable(torch.rand(1, 3, 4, 4))
model = InstanceNorm()
save_data_and_model("instancenorm", input, model)

class PoolConv(nn.Module):

    def __init__(self):
        super(PoolConv, self).__init__()
        self.pool = nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv = nn.Conv3d(2, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pool(x)
        y = self.conv(x)
        return y

input = Variable(torch.randn(1, 2, 4, 4, 19))
model = PoolConv()
save_data_and_model("pool_conv_3d", input, model)

class Clip(nn.Module):

    def __init__(self):
        super(Clip, self).__init__()

    def forward(self, x):
        return torch.clamp(x, -0.1, 0.2)

model = Clip()
input = Variable(torch.rand(1, 10, 2, 2))
save_data_and_model('clip', input, model)

input = Variable(torch.randn(1, 3, 6, 6, 6))
deconv = nn.ConvTranspose3d(3, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
save_data_and_model("deconv3d", input, deconv)

input = Variable(torch.randn(1, 3, 5, 4, 4))
deconv = nn.ConvTranspose3d(3, 5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
save_data_and_model("deconv3d_bias", input, deconv)

input = Variable(torch.randn(1, 3, 5, 5, 5))
deconv = nn.ConvTranspose3d(3, 2, kernel_size=(4, 3, 3), stride=(1, 1, 1), padding=(1, 0, 1), bias=True)
save_data_and_model("deconv3d_pad", input, deconv)

input = Variable(torch.randn(1, 3, 4, 5, 3))
deconv = nn.ConvTranspose3d(3, 5, kernel_size=(2, 3, 1), stride=(2, 2, 2), padding=(1, 2, 1), output_padding=1, bias=True)
save_data_and_model("deconv3d_adjpad", input, deconv)

input = np.random.rand(1, 3, 4, 2)
output = np.mean(input, axis=(2, 3), keepdims=True)
save_onnx_data_and_model(input, output, 'reduce_mean', 'ReduceMean', axes=(2, 3), keepdims=True)

input = np.random.rand(1, 3, 4, 2, 3)
output = np.mean(input, axis=(3, 4), keepdims=True)
save_onnx_data_and_model(input, output, 'reduce_mean3d', 'ReduceMean', axes=(3, 4), keepdims=True)

class Split(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Split, self).__init__()
        self.split_size_sections = \
            kwargs.get('split_size_sections', 1)
        self.dim = kwargs.get('dim', 0)

    def forward(self, x):
        tup = torch.split(x, self.split_size_sections, self.dim)
        return torch.cat(tup)

model = Split()
input = Variable(torch.tensor([1., 2.], dtype=torch.float32))
save_data_and_model("split_1", input, model)

model = Split(dim=0)
save_data_and_model("split_2", input, model)

model = Split(split_size_sections=[1, 1])
save_data_and_model("split_3", input, model)

model = Split(dim=0, split_size_sections=[1, 1])
save_data_and_model("split_4", input, model)

class SplitMax(nn.Module):

    def __init__(self):
        super(SplitMax, self).__init__()

    def forward(self, x):
        first, second = torch.split(x, (2, 4), dim=1)
        second, third = torch.split(second, (2, 2), dim=1)
        return torch.max(first, torch.max(second, third))

model = SplitMax()
input = Variable(torch.randn(1, 6, 2, 3))
save_data_and_model("split_max", input, model)

class Squeeze(nn.Module):

    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x, dim=1)

input = Variable(torch.randn(3, 1, 2, 4))
model = Squeeze()
model.eval()
save_data_and_model("squeeze", input, model)

class Div(nn.Module):

    def __init__(self):
        super(Div, self).__init__()

    def forward(self, a, b):
        return torch.div(a, b)

a = Variable(torch.randn(1, 3, 2, 2))
b = Variable(torch.randn(1, 3, 2, 2))
model = Div()
model.eval()
save_data_and_model_multy_inputs("div", model, a, b)

class ReduceL2(nn.Module):

    def __init__(self):
        super(ReduceL2, self).__init__()

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        l2norm = torch.div(x, norm)
        return l2norm.transpose(2, 3)

input = Variable(torch.randn(1, 3, 2, 4))
model = ReduceL2()
model.eval()
save_data_and_model("reduceL2", input, model)

class SoftMaxUnfused(nn.Module):

    def __init__(self):
        super(SoftMaxUnfused, self).__init__()

    def forward(self, x):
        exp = torch.exp(x)
        sum = torch.sum(exp, dim=2, keepdim=True)
        return exp / sum

input = Variable(torch.randn(1, 2, 4, 3))
model = SoftMaxUnfused()
save_data_and_model("softmax_unfused", input, model)

class FlattenByProd(nn.Module):
    def __init__(self):
        super(FlattenByProd, self).__init__()

    def forward(self, image):
        batch_size = image.size(0)
        channels = image.size(1)
        h = image.size(2)
        w = image.size(3)
        image = image.view(batch_size, channels*h*w)
        return image

input = Variable(torch.randn(1, 2, 3, 4))
model = FlattenByProd()
save_data_and_model("flatten_by_prod", input, model, version=11)

class ReshapeByDiv(nn.Module):
    def __init__(self):
        super(ReshapeByDiv, self).__init__()

    def forward(self, image):
        batch_size = image.size(0)
        channels = image.size(1)
        h = image.size(2)
        w = image.size(3)
        image = image.view(batch_size, channels*h* (w / 2), -1)
        return image

input = Variable(torch.randn(1, 2, 3, 4))
model = ReshapeByDiv()
save_data_and_model("dynamic_reshape_opset_11", input, model, version=11)

class Broadcast(nn.Module):

    def __init__(self):
        super(Broadcast, self).__init__()

    def forward(self, x, y):
        return x * y + (x - x) / y - y

input1 = Variable(torch.randn(1, 4, 1, 2))
input2 = Variable(torch.randn(1, 4, 1, 1))
save_data_and_model_multy_inputs("channel_broadcast", Broadcast(), input1, input2)

class FlattenConst(Function):
    @staticmethod
    def symbolic(g, x):
        return g.op("Flatten", x)

    @staticmethod
    def forward(self, x):
        return torch.flatten(x)

class FlattenModel(nn.Module):
    def __init__(self):
        super(FlattenModel, self).__init__()

    def forward(self, input):
        sizes = torch.tensor(input.shape)
        flatten = FlattenConst.apply(sizes)
        return input + flatten

x = Variable(torch.rand(1, 2))
model = FlattenModel()
save_data_and_model("flatten_const", x, model)

class Cast(nn.Module):
    def __init__(self):
        super(Cast, self).__init__()

    def forward(self, x):
        return x.type(torch.FloatTensor)

x = Variable(torch.randn(1, 2))
model = Cast()
save_data_and_model("cast", x, model)

class DynamicResize(nn.Module):
    def __init__(self):
        super(DynamicResize, self).__init__()

    def forward(self, x, y):
        h = y.size(2)
        w = y.size(3)
        up = nn.Upsample(size=[h, w], mode='bilinear')
        return up(x) + y

input_0 = Variable(torch.randn(1, 3, 8, 6))
input_1 = Variable(torch.randn(1, 3, 4, 3))
model = DynamicResize()
save_data_and_model_multy_inputs("dynamic_resize", model, input_0, input_1, version=11)

class ShapeConst(nn.Module):
    def __init__(self):
        super(ShapeConst, self).__init__()

    def forward(self, x):
      x = 2 * x
      z = torch.zeros(x.shape, dtype=torch.float32)
      return z + x

x = Variable(torch.Tensor([[1, 2, 3], [1, 2, 3]]))
model = ShapeConst()
save_data_and_model("shape_of_constant", x, model, version=11)


class LSTM(nn.Module):

    def __init__(self, features, hidden, batch, num_layers=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(features, hidden, num_layers, bidirectional=bidirectional)
        self.h0 = torch.zeros(num_layers + int(bidirectional), batch, hidden)
        self.c0 = torch.zeros(num_layers + int(bidirectional), batch, hidden)

    def forward(self, x):
        return self.lstm(x, (self.h0, self.c0))[0]

batch = 5
features = 4
hidden = 3
seq_len = 2

input = Variable(torch.randn(seq_len, batch, features))
lstm = LSTM(features, hidden, batch, bidirectional=False)
save_data_and_model("lstm", input, lstm)

input = Variable(torch.randn(seq_len, batch, features))
lstm = LSTM(features, hidden, batch, bidirectional=True)
save_data_and_model("lstm_bidirectional", input, lstm)

class MatMul(nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, x):
      axis = len(x.shape)
      return x @ x.transpose(axis - 1, axis - 2)

model = MatMul()
x = Variable(torch.randn(2, 4))
save_data_and_model("matmul_2d", x, model)

x = Variable(torch.randn(3, 2, 4))
save_data_and_model("matmul_3d", x, model)

x = Variable(torch.randn(1, 3, 2, 4))
save_data_and_model("matmul_4d", x, model)

x = np.random.rand(1, 3, 2)
output = np.mean(x, axis=1, keepdims=True)
save_onnx_data_and_model(x, output, 'reduce_mean_axis1', 'ReduceMean', axes=(1), keepdims=True)

x = np.random.rand(1, 3, 2)
output = np.mean(x, axis=2, keepdims=True)
save_onnx_data_and_model(x, output, 'reduce_mean_axis2', 'ReduceMean', axes=(2), keepdims=True)

class Expand(nn.Module):
    def __init__(self, shape):
        super(Expand, self).__init__()
        self.shape = shape

    def forward(self, x):
      return x.expand(self.shape)

x = Variable(torch.randn(1, 1, 2, 2))
model = Expand(shape=[2, 1, 2, 2])
save_data_and_model("expand_batch", x, model)

x = Variable(torch.randn(1, 1, 2, 2))
model = Expand(shape=[1, 3, 2, 2])
save_data_and_model("expand_channels", x, model)

x = Variable(torch.randn(1, 2, 1, 1))
model = Expand(shape=[1, 2, 3, 4])
save_data_and_model("expand_hw", x, model)

class NormL2(nn.Module):
    def __init__(self):
        super(NormL2, self).__init__()

    def forward(self, x):
      norm = torch.norm(x, p=2, dim=1, keepdim=True)
      clip = torch.clamp(norm, min=0)
      expand = clip.expand_as(x)
      return x / expand

model = NormL2()
x = Variable(torch.randn(1, 2, 3, 4))
save_data_and_model("reduceL2_subgraph", x, model)

model = nn.ZeroPad2d(1)
model.eval()
input = torch.rand(1, 3, 2, 4)
save_data_and_model("ZeroPad2d", input, model, version = 11)

model = nn.ReflectionPad2d(1)
model.eval()
input = torch.rand(1, 3, 2, 4)
save_data_and_model("ReflectionPad2d", input, model, version = 11)

# source: https://github.com/amdegroot/ssd.pytorch/blob/master/layers/modules/l2norm.py
class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.view(1, -1, 1, 1) * x
        return x

model = L2Norm(2, 20)
x = Variable(torch.randn(1, 2, 3, 4))
save_data_and_model("reduceL2_subgraph_2", x, model)

from torchvision.ops.misc import *
n = 3
model = FrozenBatchNorm2d(n)
model.eval()
input = Variable(torch.rand( 1, 3, 2, 4 ))
save_data_and_model("frozenBatchNorm2d", input, model)

class UpsampleUnfusedTwoInput(nn.Module):

    def __init__(self):
        super(UpsampleUnfusedTwoInput, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        x = self.conv1(x)
        x = x.shape[-2:]
        y = self.conv2(y)
        y = F.interpolate(y, size=x, mode="nearest")
        return y

input_0 = Variable(torch.randn(1, 3, 4, 6))
input_1 = Variable(torch.randn(1, 3, 2, 2))
model = UpsampleUnfusedTwoInput()
save_data_and_model_multy_inputs("upsample_unfused_two_inputs_opset9_torch1.4", UpsampleUnfusedTwoInput(), input_0, input_1, version=9)
save_data_and_model_multy_inputs("upsample_unfused_two_inputs_opset11_torch1.4", UpsampleUnfusedTwoInput(), input_0, input_1, version=11)

 class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

x = Variable(torch.randn(1, 2, 3, 4))
model = FrozenBatchNorm2d(2)
save_data_and_model("batch_norm_subgraph", x, model)
