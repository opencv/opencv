from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
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


def save_data_and_model_multy_inputs(name, model, *args):
    for index, input in enumerate(args, start=0):
        input_files = os.path.join("data", "input_" + name + "_" + str(index))
        np.save(input_files, input)

    output = model(*args)
    print(name + " output has sizes", output.shape)
    print()
    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, output.data)

    models_files = os.path.join("models", name + ".onnx")

    onnx_model_pb = export_to_string(model, (args))
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
