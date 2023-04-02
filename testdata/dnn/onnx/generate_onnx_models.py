from __future__ import print_function
import torch
from torch.autograd import Variable, Function
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf # version 2.5.0
import tf2onnx # version 1.9.1
import paddle # version 2.1.1
import numpy as np
import os.path
import onnx
import onnxsim
import google.protobuf.text_format
import io
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


def export_to_string(model, inputs, version=None, export_params=False):
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, inputs, f, export_params=export_params, opset_version=version)
    return f.getvalue()


def save_data_and_model(name, input, model, version=None, export_params=False):
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

    onnx_model_pb = export_to_string(model, input, version, export_params)
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

def save_data_and_onnx_model(name, input_np, output_np, onnx_model):
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

def save_data_and_onnx_model_multy_inputs(name, input_list, output_np, onnx_model):
    for index in range(len(input_list)):
        print(name + " input  "+str(index)+" has sizes",  input_list[index].shape)
        input_files = os.path.join("data", "input_" + name + "_" + str(index))
        np.save(input_files, input_list[index])

    print(name + " output has sizes", output_np.shape)
    print()
    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output_np.data))

    models_files = os.path.join("models", name + ".onnx")

    onnx_model_pb = onnx._serialize(onnx_model)
    model_def = assertONNXExpected(onnx_model_pb)
    with open(models_files, 'wb') as file:
        file.write(model_def.SerializeToString())

def simplify(name, rename=False, **kwargs):
    model, check = onnxsim.simplify(name, **kwargs)
    assert check, "couldn't valide"
    name = name[:-5]
    if rename:
        name += '_optimized'
    onnx.save(model, name + '.onnx')

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

class PReLU_slope(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PReLU_slope, self).__init__()

    def forward(self, x):
        return nn.PReLU()(x)

model = PReLU_slope()
input_ = Variable(torch.randn(1, 1, 5, 5, dtype=torch.float32))
save_data_and_model("PReLU_slope", input_, model, export_params=True)
simplify('models/PReLU_slope.onnx', False)


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


class ConcatConstBlob(nn.Module):

    def __init__(self):
        super(ConcatConstBlob, self).__init__()
        self.squeeze = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.squeeze(x)
        y = torch.tensor([[[[0.1, -0.2], [-0.3, 0.4]]]], dtype=torch.float32)
        return torch.cat([x, y], axis=1)


input = Variable(torch.randn(1, 2, 2, 2))
model = ConcatConstBlob()
model.eval()
save_data_and_model("concat_const_blob", input, model)


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

    onnx_model_pb = export_to_string(model, (args), version=kwargs.get('version', None), export_params=kwargs.get('export_params', False))
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

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, dim = self.dim)

input = Variable(torch.randn(1, 2, 3))
model = Unsqueeze(1)
model.eval()
save_data_and_model("unsqueeze", input, model)
save_data_and_model("unsqueeze_opset_13", input, model, version=13)

model = Unsqueeze(-2)
model.eval()
save_data_and_model("unsqueeze_neg_axes", input, model)

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

    def __init__(self, custom_slice=None):
        self.custom_slice=custom_slice
        super(Slice, self).__init__()

    def forward(self, x):
        if self.custom_slice:
           return x[self.custom_slice]

        return x[..., 1:-1, 0:3]

input = Variable(torch.randn(1, 2, 4, 4))
model = Slice()
save_data_and_model("slice", input, model)
save_data_and_model("slice_opset_11", input, model, version=11)

def generate_slice_neg_starts():
    x = np.random.randn(2, 3, 4, 3).astype(np.float32)
    y = x[-1:2, -3:-1, 2:3, 1:-1]

    starts = np.array([-1, -3, 2,  1], dtype=np.int64)
    starts = onnx.numpy_helper.from_array(starts, name='starts')
    ends =   np.array([ 2, -1, 3, -1], dtype=np.int64)
    ends =   onnx.numpy_helper.from_array(ends, name='ends')

    node = onnx.helper.make_node(
        'Slice',
        inputs=['X', 'starts', 'ends'],
        outputs=['Y'],
    )

    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, list(x.shape))
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, list(y.shape))

    graph = onnx.helper.make_graph(
        [node],             # nodes
        'slice_neg_starts', # name
        [X],                # inputs
        [Y],                # outputs
    )

    graph.initializer.append(starts)
    graph.initializer.append(ends)

    model = onnx.helper.make_model(graph, producer_name='onnx')
    onnx.checker.check_model(model)

    name = 'slice_neg_starts'

    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, x.data)

    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(y.data))

    models_files = os.path.join("models", name + ".onnx")
    onnx.save(model, models_files)

generate_slice_neg_starts()

input_2 = Variable(torch.randn(6, 6))
custom_slice_list = [
    slice(1, 3, 1),
    slice(0, 3, 2)
]
model_2 = Slice(custom_slice=custom_slice_list)
save_data_and_model("slice_opset_11_steps_2d", input_2, model_2, version=11)
postprocess_model("models/slice_opset_11_steps_2d.onnx", [['height', 'width']])

input_3 = Variable(torch.randn(3, 6, 6))
custom_slice_list_3 = [
    slice(None, None, 2),
    slice(None, None, 2),
    slice(None, None, 2)
]
model_3 = Slice(custom_slice=custom_slice_list_3)
save_data_and_model("slice_opset_11_steps_3d", input_3, model_3, version=11)
postprocess_model("models/slice_opset_11_steps_3d.onnx", [[3, 'height', 'width']])

input_4 = Variable(torch.randn(1, 3, 6, 6))
custom_slice_list_4 = [
    slice(0, 5, None),
    slice(None, None, None),
    slice(1, None, 2),
    slice(None, None, None)
]
model_4 = Slice(custom_slice=custom_slice_list_4)
save_data_and_model("slice_opset_11_steps_4d", input_4, model_4, version=11)
postprocess_model("models/slice_opset_11_steps_4d.onnx", [["batch_size", 3, 'height', 'width']])

input_5 = Variable(torch.randn(1, 2, 3, 6, 6))
custom_slice_list_5 = [
    slice(None, None, None),
    slice(None, None, None),
    slice(0, None, 3),
    slice(None, None, None),
    slice(None, None, 2)
]
model_5 = Slice(custom_slice=custom_slice_list_5)
save_data_and_model("slice_opset_11_steps_5d", input_5, model_5, version=11)

########### Slice with axes ###########
def generate_slice_with_axes():
    def generate_model(name, X, Y, starts, ends, axes, steps=None):
        starts = onnx.numpy_helper.from_array(starts, name='starts')
        ends = onnx.numpy_helper.from_array(ends, name='ends')
        axes = onnx.numpy_helper.from_array(axes, name='axes')
        inputs = ['X', 'starts', 'ends', 'axes']
        if steps is not None:
            steps = onnx.numpy_helper.from_array(steps, name='steps')
            inputs.append('steps')

        node = onnx.helper.make_node(
            'Slice',
            inputs,
            outputs=['Y'],
        )

        X = onnx.helper.make_tensor_value_info(
            'X', onnx.TensorProto.FLOAT, list(x.shape))
        Y = onnx.helper.make_tensor_value_info(
            'Y', onnx.TensorProto.FLOAT, list(y.shape))

        graph = onnx.helper.make_graph(
            [node],             # nodes
            name,               # name
            [X],                # inputs
            [Y],                # outputs
        )

        graph.initializer.append(starts)
        graph.initializer.append(ends)
        graph.initializer.append(axes)
        if steps is not None:
            graph.initializer.append(steps)

        model = onnx.helper.make_model(graph, producer_name='onnx')
        onnx.checker.check_model(model)

        input_files = os.path.join("data", "input_" + name)
        np.save(input_files, x.data)

        output_files = os.path.join("data", "output_" + name)
        np.save(output_files, np.ascontiguousarray(y.data))

        models_files = os.path.join("models", name + ".onnx")
        onnx.save(model, models_files)

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([1, 2, 3], dtype=np.int64)
    ends = np.array([5, 11, 8], dtype=np.int64)
    axes = np.array([2, 0, 1], dtype=np.int64)
    y = x[2:11, 3:8, 1:5]
    generate_model("slice_nonseq_axes", x, y, starts, ends, axes)

    steps = np.array([1, 4, 2], dtype=np.int64)
    y = x[2:11:4, 3:8:2, 1:5:1]
    generate_model("slice_nonseq_axes_steps", x, y, starts, ends, axes, steps)

    starts = np.array([1, 2], dtype=np.int64)
    ends = np.array([5, 11], dtype=np.int64)
    axes = np.array([2, 0], dtype=np.int64)
    steps = np.array([1, 4], dtype=np.int64)
    y = x[2:11:4, :, 1:5:1]
    generate_model("slice_nonseq_miss_axes_steps", x, y, starts, ends, axes, steps)

    x = np.random.randn(3, 10, 8, 5).astype(np.float32)
    starts = np.array([0, 2, 3, 1], dtype=np.int64)
    ends = np.array([3, 9, 7, 4], dtype=np.int64)
    axes = np.array([-4, 1, -2, 3], dtype=np.int64)
    y = x[0:3, 2:9, 3:7, 1:4]
    generate_model("slice_neg_axes", x, y, starts, ends, axes)

    steps = np.array([1, 4, 3, 2], dtype=np.int64)
    y = x[0:3:1, 2:9:4, 3:7:3, 1:4:2]
    generate_model("slice_neg_axes_steps", x, y, starts, ends, axes, steps)

    starts = np.array([0, 3], dtype=np.int64)
    ends = np.array([3, 7], dtype=np.int64)
    axes = np.array([-4, -2], dtype=np.int64)
    steps = np.array([1, 3], dtype=np.int64)
    y = x[0:3:1, :, 3:7:3, :]
    generate_model("slice_neg_miss_axes_steps", x, y, starts, ends, axes, steps)

generate_slice_with_axes()


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

class DepthWiseAdd(nn.Module):

    def __init__(self):
        super(DepthWiseAdd, self).__init__()
        self.dconv1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0, groups=8)
        self.dconv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0, groups=8)

    def forward(self, x):
        a = self.dconv1(x)
        b = self.dconv2(x)
        z = a + b
        z = z * 2
        return z


input = Variable(torch.randn(1, 8, 32, 32))
model = DepthWiseAdd()
model.eval()
save_data_and_model("depthwiseconv_add", input, model)

class DepthWiseStride2(nn.Module):

    def __init__(self):
        super(DepthWiseStride2, self).__init__()
        self.dconv1 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1, groups=8)

    def forward(self, x):
        a = self.dconv1(x)
        return a

input = Variable(torch.randn(1, 8, 6, 6))
model = DepthWiseStride2()
model.eval()
save_data_and_model("depthwise_stride2", input, model)

class Clip(nn.Module):

    def __init__(self):
        super(Clip, self).__init__()

    def forward(self, x):
        return torch.clamp(x, -0.1, 0.2)

model = Clip()
input = Variable(torch.rand(1, 10, 2, 2))
save_data_and_model('clip', input, model)

########### clip_init ###########

operation = "Clip"
min = -0.5
max = 0.5

input = np.random.randn(3, 4, 5).astype(np.float32)
output = np.clip(input, min, max)

X = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [3, 4, 5])
MIN = onnx.helper.make_tensor_value_info('min', onnx.TensorProto.FLOAT, [1])
MAX = onnx.helper.make_tensor_value_info('max', onnx.TensorProto.FLOAT, [1])
Y = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [3, 4, 5])
MIN_INIT = onnx.helper.make_tensor("min", onnx.TensorProto.FLOAT, [1], np.array([min]))
MAX_INIT = onnx.helper.make_tensor("max", onnx.TensorProto.FLOAT, [1], np.array([max]))

name = "clip_init_min_max"
input = np.random.randn(3, 4, 5).astype(np.float32)
output = np.clip(input, min, max)

input_files = os.path.join("data", "input_" + name)
np.save(input_files, input.data)
output_files = os.path.join("data", "output_" + name)
np.save(output_files, np.ascontiguousarray(output.data))

node = onnx.helper.make_node(operation, inputs=['input', "min", "max"], outputs=['output'])
graph = onnx.helper.make_graph([node], name, [X, MIN, MAX], [Y], [MIN_INIT, MAX_INIT])
model = onnx.helper.make_model(graph, producer_name=name)
onnx.save(model, os.path.join("models", name + ".onnx"))

name = "clip_init_min"
input = np.random.randn(3, 4, 5).astype(np.float32)
output = np.clip(input, min, None)

input_files = os.path.join("data", "input_" + name)
np.save(input_files, input.data)
output_files = os.path.join("data", "output_" + name)
np.save(output_files, np.ascontiguousarray(output.data))

node = onnx.helper.make_node(operation, inputs=['input', "min", ""], outputs=['output'])
graph = onnx.helper.make_graph([node], name, [X, MIN], [Y], [MIN_INIT])
model = onnx.helper.make_model(graph, producer_name=name)
onnx.save(model, os.path.join("models", name + ".onnx"))

name = "clip_init_max"
input = np.random.randn(3, 4, 5).astype(np.float32)
output = np.clip(input, None, max)

input_files = os.path.join("data", "input_" + name)
np.save(input_files, input.data)
output_files = os.path.join("data", "output_" + name)
np.save(output_files, np.ascontiguousarray(output.data))

node = onnx.helper.make_node(operation, inputs=['input', "", "max"], outputs=['output'])
graph = onnx.helper.make_graph([node], name, [X, MAX], [Y], [MAX_INIT])
model = onnx.helper.make_model(graph, producer_name=name)
onnx.save(model, os.path.join("models", name + ".onnx"))

#################################

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

class SplitSizes(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SplitSizes, self).__init__()

    def forward(self, x):
        a, b, c, d = torch.split(x, [2, 3, 5, 10], 0)
        a = torch.mul(a, 2)
        b = torch.mul(b, 3)
        c = torch.mul(c, 5)
        d = torch.mul(d, 10)
        tup = (a, b, c, d)
        return torch.cat(tup)

model = SplitSizes()
input_ = Variable(torch.tensor(list(range(20)), dtype=torch.float32))
save_data_and_model("split_sizes", input_, model)

class SplitAxis(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SplitAxis, self).__init__()

    def forward(self, x):
        tup = torch.split(x, 2, -1)
        return torch.cat(tup, 1)

model = SplitAxis()
input_ = Variable(torch.randn(1, 10, dtype=torch.float32))
save_data_and_model("split_neg_axis", input_, model)

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
save_data_and_model("squeeze_axes_op13", input, model, version=13)

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
        image = image.view(batch_size, channels*h*(w // 2), -1)
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
save_data_and_model_multy_inputs("dynamic_resize_9", model, input_0, input_1, version=9)
save_data_and_model_multy_inputs("dynamic_resize_10", model, input_0, input_1, version=10)
save_data_and_model_multy_inputs("dynamic_resize_11", model, input_0, input_1, version=11)
save_data_and_model_multy_inputs("dynamic_resize_13", model, input_0, input_1, version=13)

class DynamicResizeScale(nn.Module):
    def forward(self, x, y):
        up = nn.Upsample(scale_factor=(0.5, 0.5), mode='bilinear')
        return up(x) + y

input_0 = Variable(torch.randn(1, 3, 8, 6))
input_1 = Variable(torch.randn(1, 3, 4, 3))
model = DynamicResizeScale()
save_data_and_model_multy_inputs("dynamic_resize_scale_9", model, input_0, input_1, version=9, export_params=True)
save_data_and_model_multy_inputs("dynamic_resize_scale_10", model, input_0, input_1, version=10, export_params=True)
save_data_and_model_multy_inputs("dynamic_resize_scale_11", model, input_0, input_1, version=11, export_params=True)
save_data_and_model_multy_inputs("dynamic_resize_scale_13", model, input_0, input_1, version=13, export_params=True)

class Resize(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Resize, self).__init__()

    def forward(self, x):
        return F.interpolate(input, [12, 12], mode="bilinear", align_corners=True)

input = Variable(torch.randn(1, 2, 6, 6))
model = Resize(input)
save_data_and_model("resize_size_opset11", input, model, 11)
save_data_and_model("resize_size_opset13", input, model, 13)

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



class HiddenLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, is_bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi_coeff = 2 if is_bidirectional else 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=is_bidirectional)

    def forward(self, t):
        h_0 = torch.ones(self.num_layers * self.bi_coeff, t.size(1),
                         self.hidden_size)
        c_0 = torch.ones(self.num_layers * self.bi_coeff, t.size(1),
                         self.hidden_size)
        return self.lstm(t, (h_0, c_0))[0]

input = torch.randn(seq_len, batch, features)
hidden_lstm = HiddenLSTM(features, hidden, num_layers=3, is_bidirectional=False)
save_data_and_model("hidden_lstm", input, hidden_lstm, version=11, export_params=True)

input = torch.randn(seq_len, batch, features)
hidden_lstm = HiddenLSTM(features, hidden, num_layers=3, is_bidirectional=True)
save_data_and_model("hidden_lstm_bi", input, hidden_lstm, version=11, export_params=True)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, is_bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi_coeff = 2 if is_bidirectional else 1
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bidirectional=is_bidirectional)

    def forward(self, t):
        h_0 = torch.ones(self.num_layers * self.bi_coeff, t.size(1),
                         self.hidden_size)
        return self.gru(t, h_0)[0]

input = torch.randn(seq_len, batch, features)
hidden_lstm = GRU(features, hidden, num_layers=3, is_bidirectional=False)
save_data_and_model("gru", input, hidden_lstm, version=11, export_params=True)

input = torch.randn(seq_len, batch, features)
hidden_lstm = GRU(features, hidden, num_layers=3, is_bidirectional=True)
save_data_and_model("gru_bi", input, hidden_lstm, version=11, export_params=True)


batch = 5
features = 4
hidden = 3
seq_len = 2
num_layers=1
bidirectional=True

class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(features, hidden, num_layers, bidirectional=bidirectional)
        self.h0 = torch.from_numpy(np.ones((num_layers + int(bidirectional), batch, hidden), dtype=np.float32))
        self.c0 = torch.from_numpy(np.ones((num_layers + int(bidirectional), batch, hidden), dtype=np.float32))

    def forward(self, x):
        a, (b, c) = self.lstm(x, (self.h0, self.c0))
        if bidirectional:
            return torch.cat((a, b, c), dim=2)
        else:
            return torch.cat((a, b, c), dim=0)


input_ = Variable(torch.randn(seq_len, batch, features))
lstm = LSTM()
save_data_and_model("lstm_cell_bidirectional", input_, lstm, export_params=True)

bidirectional = False
input_ = Variable(torch.randn(seq_len, batch, features))
lstm = LSTM()
save_data_and_model("lstm_cell_forward", input_, lstm, export_params=True)


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

########### MatMul init ###########
def generate_matmul_init(name, inputA, inputB):
    output = inputA @ inputB
    shapeA = inputA.shape
    shapeB = inputB.shape
    shapeY = output.shape

    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, inputA.data)
    output_files = os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output.data))

    A = onnx.helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, shapeA)
    B = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, shapeB)
    Y = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, shapeY)
    B_INIT = onnx.helper.make_tensor("B", onnx.TensorProto.FLOAT, shapeB, inputB)


    node = onnx.helper.make_node("MatMul", inputs=['A', "B",], outputs=['output'])
    graph = onnx.helper.make_graph([node], name, [A, B], [Y], [B_INIT])
    model = onnx.helper.make_model(graph, producer_name=name)
    onnx.save(model, os.path.join("models", name + ".onnx"))

inputA = np.random.randn(2, 3).astype(np.float32)
inputB = np.random.randn(3, 4).astype(np.float32)
generate_matmul_init("matmul_2d_init", inputA, inputB)

inputA = np.random.randn(5, 2, 3).astype(np.float32)
inputB = np.random.randn(5, 3, 4).astype(np.float32)
generate_matmul_init("matmul_3d_init", inputA, inputB)

inputA = np.random.randn(6, 2, 3, 4).astype(np.float32)
inputB = np.random.randn(6, 2, 4, 5).astype(np.float32)
generate_matmul_init("matmul_4d_init", inputA, inputB)

inputA = np.random.randn(2, 3, 4, 5).astype(np.float32)
inputB = np.random.randn(3, 5, 6).astype(np.float32)
generate_matmul_init("matmul_init_bcast", inputA, inputB)

def generate_matmul_init_2(name, inputA, inputB):
    outputY = inputA @ inputB
    shapeA = inputA.shape
    shapeB = inputB.shape
    shapeY = outputY.shape

    A = onnx.helper.make_tensor_value_info('A', onnx.TensorProto.FLOAT, shapeA)
    B = onnx.helper.make_tensor_value_info('B', onnx.TensorProto.FLOAT, shapeB)
    A_INIT = onnx.helper.make_tensor("A", onnx.TensorProto.FLOAT, shapeA, inputA)
    B_INIT = onnx.helper.make_tensor("B", onnx.TensorProto.FLOAT, shapeB, inputB)
    node1 = onnx.helper.make_node("MatMul", inputs=['A', "B",], outputs=['outputY'])

    input = outputY + np.random.rand()
    output = input + outputY
    shapeC = input.shape
    C = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, shapeC)
    O = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, shapeY)
    node2 = onnx.helper.make_node("Add", inputs=['input', "outputY",], outputs=['output'])

    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, input.data)
    output_files = os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output.data))

    graph = onnx.helper.make_graph([node1, node2], name, [A, B, C], [O], [A_INIT, B_INIT])
    model = onnx.helper.make_model(graph, producer_name=name)
    onnx.save(model, os.path.join("models", name + ".onnx"))


inputA = np.random.randn(4, 5).astype(np.float32)
inputB = np.random.randn(5, 6).astype(np.float32)
generate_matmul_init_2("matmul_init_2", inputA, inputB)

x = np.random.rand(1, 3, 2)
output = np.mean(x, axis=1, keepdims=True)
save_onnx_data_and_model(x, output, 'reduce_mean_axis1', 'ReduceMean', axes=(1), keepdims=True)

x = np.random.rand(1, 3, 2)
output = np.mean(x, axis=2, keepdims=True)
save_onnx_data_and_model(x, output, 'reduce_mean_axis2', 'ReduceMean', axes=(2), keepdims=True)

class Expand(nn.Module):
    def __init__(self):
        super(Expand, self).__init__()

    def forward(self, x):
        return x.expand(1, 3, -1, -1, -1)

input = Variable(torch.randn(1, 3, 2, 4))
model = Expand()
model.eval()
save_data_and_model("expand", input, model, export_params=True, version=12)
simplify('models/expand.onnx', False)

class ExpandIdentity(nn.Module):
    def __init__(self):
        super(ExpandIdentity, self).__init__()

    def forward(self, x):
        return x.expand(1, 3, -1, -1)

input = Variable(torch.randn(1, 3, 2, 4))
model = ExpandIdentity()
model.eval()
save_data_and_model("expand_identity", input, model, export_params=True, version=12)
simplify('models/expand_identity.onnx', False)

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

class reduceL2_subgraph2_2(nn.Module):
    def __init__(self):
        super(reduceL2_subgraph2_2, self).__init__()
        self.size = torch.Size([1, 3, 2, 4])

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        clip = torch.clamp(norm, min=0)
        expand = clip.expand([1, 3, 2, 4])
        return x / expand

input = Variable(torch.randn(1, 3, 2, 4))
model = reduceL2_subgraph2_2()
model.eval()
save_data_and_model("reduceL2_subgraph2_2", input, model, export_params=True, version=12)
simplify('models/reduceL2_subgraph2_2.onnx', False)

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

class GatherScalar(nn.Module):
    def forward(self, x):
        return x[1]

x = Variable(torch.randn(2))
model = GatherScalar()
save_data_and_model("gather_scalar", x, model)

class Gather(nn.Module):
    def forward(self, x):
        return x[..., 1]

x = Variable(torch.randn(2, 2, 2, 2))
model = Gather()
save_data_and_model("gather", x, model)

class Conv(nn.Module):
    def forward(self, x, kernel):
        out = F.conv2d(x, kernel, groups=1)
        return out

x = Variable(torch.randn(2, 2, 10, 10))
kernel = Variable(torch.randn(2, 2, 2, 2))
model = Conv()
save_data_and_model_multy_inputs("conv_variable_w", model, x, kernel)

class ConvBias(nn.Module):
    def forward(self, x, kernel, bias):
      batch = kernel.size(0)
      channel = kernel.size(1)
      x = x.view(1, batch*channel, x.size(2), x.size(3))
      kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
      conv = nn.Conv2d(batch*channel, batch*channel, kernel_size=(kernel.size(2), kernel.size(3)), bias=False, groups=batch*channel)
      conv.weight = nn.Parameter(kernel)
      conv.bias = nn.Parameter(bias)
      out = conv(x)
      out = out.view(batch, channel, out.size(2), out.size(3))
      return out

x = Variable(torch.randn(2, 2, 5, 5))
kernel = Variable(torch.randn(2, 2, 2, 2))
bias = Variable(torch.randn(4))
model = ConvBias()
save_data_and_model_multy_inputs("conv_variable_wb", model, x, kernel, bias)

x = Variable(torch.randn(1, 2, 2))
model = nn.Linear(2, 2, bias=True)
save_data_and_model("matmul_add", x, model)
input = np.random.rand(1, 3, 4, 2)
output = np.sum(input, axis=(-1), keepdims=False)
save_onnx_data_and_model(input, output, 'reduce_sum', 'ReduceSum', axes=(-1), keepdims=False)

x = Variable(torch.randn(1, 2, 2))
model = Expand(shape=[2, -1, -1, -1])
save_data_and_model("expand_neg_batch", x, model)

class LinearWithConstantInput(nn.Module):
    def __init__(self, in_dim = 2, const_dim=2, out_dim = 2):
        super(LinearWithConstantInput, self).__init__()
        self.in_dim = in_dim
        self.const_dim = const_dim
        self.lin_const = nn.Linear(const_dim, out_dim)
        self.lin_inp = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        x = x.reshape(-1, self.in_dim)
        const = torch.zeros(1, self.const_dim)
        x_projected = self.lin_inp(x)
        const_projected = self.lin_const(const)
        return x_projected*const_projected

x = Variable(torch.rand([1, 2, 2]))
model = LinearWithConstantInput()
save_data_and_model("lin_with_constant", x, model)

class MatmulWithTwoInputs(nn.Module):
    def __init__(self, in_dim = 2, const_dim=2, interm_dim = 2):
        super(MatmulWithTwoInputs, self).__init__()
        self.in_dim = in_dim
        self.const_dim = const_dim
        self.interm_dim = interm_dim
        self.linear_for_const = nn.Linear(const_dim, interm_dim)
        self.first_linear = nn.Linear(in_dim, interm_dim)
        self.second_linear = nn.Linear(interm_dim, 1)
    def forward(self, x):
        x = x.reshape(-1, self.in_dim)
        x_projected = self.first_linear(x)
        const = torch.zeros(1, self.interm_dim)
        const_projected = self.linear_for_const(const)
        const_projected = const_projected.expand(2, self.interm_dim)
        sum_tanh = torch.tanh(const_projected + x_projected)
        sum_tanh = sum_tanh.reshape(-1, self.interm_dim)
        sum_tanh_projected = self.second_linear(sum_tanh)
        sum_tanh_projected = sum_tanh_projected.reshape(1, 2)
        after_softmax = F.softmax(sum_tanh_projected, dim=1)
        return torch.matmul(after_softmax, x)

x = Variable(torch.rand([1, 2, 2]))
model = MatmulWithTwoInputs()
save_data_and_model("matmul_with_two_inputs", x, model)

class Power(nn.Module):
  def __init__(self, norm):
    super(Power, self).__init__()
    self.p = norm

  def forward(self, x):
    return x.pow(self.p)

x = Variable(torch.randn(2, 2))
model = Power(2)
save_data_and_model("pow2", x, model)

class Exp(nn.Module):
  def forward(self, x):
    return x.exp()

x = Variable(torch.randn(2, 2))
model = Exp()
save_data_and_model("exp", x, model)

class Ceil(nn.Module):
    def __init__(self):
        super(Ceil, self).__init__()

    def forward(self, x):
        return torch.ceil(x)

model = Ceil()
input = Variable(torch.randn(1, 2, 3, 4, dtype=torch.float32))
save_data_and_model("ceil", input, model, version = 11)

class Floor(nn.Module):
    def __init__(self):
        super(Floor, self).__init__()

    def forward(self, x):
        return torch.floor(x)

model = Floor()
input = Variable(torch.randn(1, 2, 3, 4, dtype=torch.float32))
save_data_and_model("floor", input, model, version = 11)

class Log(nn.Module):
    def __init__(self):
        super(Log, self).__init__()

    def forward(self, x):
        return torch.log(torch.abs(x + 0.1))

model = Log()
input = Variable(torch.randn(1, 2, 3, 4, dtype=torch.float32))
save_data_and_model("log", input, model, version = 11)

class Round(nn.Module):
    def __init__(self):
        super(Round, self).__init__()

    def forward(self, x):
        return torch.round(x)

model = Round()
input = Variable(torch.tensor([[-1.5, -1., -0.9, -0.5, -0.4, 0., 0.4, 0.5, 0.9, 1, 1.5]]))
save_data_and_model("round", input, model, version = 11)

class Sqrt(nn.Module):

    def __init__(self):
        super(Sqrt, self).__init__()

    def forward(self, a):
        return torch.sqrt(torch.FloatTensor.abs(a))

a = Variable(torch.randn(1, 3, 2, 2))
model = Sqrt()
save_data_and_model("sqrt", a, model)

class Equal(nn.Module):

    def __init__(self):
        super(Equal, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return (x == 0.5)*x

model = Equal()
input = Variable(torch.rand(1, 3, 4, 5))
save_data_and_model("equal", input, model, version = 11, export_params=True)

class EqualSameDims(nn.Module):

    def __init__(self):
        super(EqualSameDims, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x1 = x1 == x2
        return x2*x1

model = EqualSameDims()
input1 = Variable(torch.rand(1, 3, 4, 5))
input2 = Variable(torch.rand(1, 3, 4, 5))
save_data_and_model_multy_inputs("equal_same_dims", model, input1, input2, export_params=True)

class Less(nn.Module):

    def __init__(self):
        super(Less, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return (x < 0.7)*x

model = Less()
input = Variable(torch.rand(1, 3, 4, 5))
save_data_and_model("less", input, model, version = 11, export_params=True)

class LessSameDims(nn.Module):

    def __init__(self):
        super(LessSameDims, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x1 = x1 < x2
        return x2*x1

model = LessSameDims()
input1 = Variable(torch.rand(1, 3, 4, 5))
input2 = Variable(torch.rand(1, 3, 4, 5))
save_data_and_model_multy_inputs("less_same_dims", model, input1, input2, export_params=True)

class Greater(nn.Module):

    def __init__(self):
        super(Greater, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return (x > 0.5)*x

model = Greater()
input = Variable(torch.rand(1, 3, 4, 5))
save_data_and_model("greater", input, model, version = 11, export_params=True)

class GreaterOrEqual(nn.Module):

    def __init__(self):
        super(GreaterOrEqual, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return (x >= 0.5)*x

model = GreaterOrEqual()
input = Variable(torch.rand(1, 3, 4, 5))
save_data_and_model("greater_or_equal", input, model, version = 13, export_params=True)

class LessOrEqual(nn.Module):

    def __init__(self):
        super(LessOrEqual, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return (x <= 0.5)*x

model = LessOrEqual()
input = Variable(torch.rand(1, 3, 4, 5))
save_data_and_model("less_or_equal", input, model, version = 13, export_params=True)

class GreaterSameDims(nn.Module):

    def __init__(self):
        super(GreaterSameDims, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x1 = x1 > x2
        return x2*x1

model = GreaterSameDims()
input1 = Variable(torch.rand(1, 3, 4, 5))
input2 = Variable(torch.rand(1, 3, 4, 5))
save_data_and_model_multy_inputs("greater_same_dims", model, input1, input2, export_params=True)

class ReduceMaxGlobal(nn.Module):
  def forward(self, x):
    out = torch.max(x)
    return torch.unsqueeze(out, 0)

x = Variable(torch.randn(1, 3, 2, 2))
model = ReduceMaxGlobal()
save_data_and_model("reduce_max", x, model)

class ReduceMax(nn.Module):
    def __init__(self, axes):
        super(ReduceMax, self).__init__()
        self.axes = axes

    def forward(self, x):
        # torch.return_types.max(values, indices)
        out = torch.max(x, dim=self.axes, keepdim=False)[0]
        return out

x = Variable(torch.randn(1, 3, 2, 2))

model = ReduceMax(axes=0)
save_data_and_model("reduce_max_axis_0", x, model)

model = ReduceMax(axes=1)
save_data_and_model("reduce_max_axis_1", x, model)

class Min(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Min, self).__init__()

    def forward(self, a, b):
        return torch.min(a, b)

model = Min()
input_0 = Variable(torch.randn(2, 3, 4, 5, dtype=torch.float32))
input_1 = Variable(torch.randn(2, 3, 4, 5, dtype=torch.float32))
save_data_and_model_multy_inputs("min", model, input_0, input_1, export_params=True)
simplify('models/min.onnx', False)

class ResizeConv(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            bias=False)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        return x

x = Variable(torch.rand(1, 2, 2, 2))
model = ResizeConv(2, 0, 2)
save_data_and_model("resize_opset11_torch1.6", x, model, 11)

class Scale(nn.Module):
  def forward(self, x):
    w = torch.mean(x, axis=(2, 3), keepdim=True)
    return w * x

x = Variable(torch.randn(1, 3, 2, 2))
model = Scale()
save_data_and_model("scale", x, model)

class ScaleBroadcast(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ScaleBroadcast, self).__init__()

    def forward(self, x0, x1, x2):
        return torch.mul(torch.mul(x0, x1), x2)

model = ScaleBroadcast()
input_0 = Variable(torch.ones(2, 1, 4, 5, dtype=torch.float32))
input_1 = Variable(torch.ones(1, 4, 1, dtype=torch.float32))
input_2 = Variable(torch.ones(2, 1, 4, 1, dtype=torch.float32))
save_data_and_model_multy_inputs("scale_broadcast", model, input_0, input_1, input_2)

class ScaleBroadcastMid(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ScaleBroadcastMid, self).__init__()

    def forward(self, x0, x1):
        return torch.mul(x0, x1)

model = ScaleBroadcastMid()
input_0 = Variable(torch.ones(2, 1, 4, dtype=torch.float32))
input_1 = Variable(torch.ones(2, 5, 4, dtype=torch.float32))
save_data_and_model_multy_inputs("scale_broadcast_mid", model, input_0, input_1)

x = Variable(torch.randn(1, 3, 25))
conv1d = nn.Conv1d(3, 2, kernel_size=3, padding=2, stride=2, dilation=2, bias=False)
save_data_and_model("conv1d", x, conv1d)

x = Variable(torch.randn(1, 3, 25))
conv1d = nn.Conv1d(3, 2, kernel_size=3, padding=0, stride=1, dilation=1, bias=True)
save_data_and_model("conv1d_bias", x, conv1d)

class Conv1d(nn.Module):
    def forward(self, x, kernel):
        out = F.conv1d(x, kernel, groups=1)
        return out

x = Variable(torch.randn(2, 2, 10))
kernel = Variable(torch.randn(2, 2, 2))
model = Conv1d()
save_data_and_model_multy_inputs("conv1d_variable_w", model, x, kernel)

class Conv1dBias(nn.Module):
    def forward(self, x, kernel, bias):
      batch = x.size(0)
      channel = x.size(1)
      x = x.view(1, batch*channel, x.size(2))
      kernel = kernel.view(batch*channel, 1, 2)
      conv = nn.Conv1d(4, 4, kernel_size=2, bias=False, groups=4)
      conv.weight = nn.Parameter(kernel)
      conv.bias = nn.Parameter(bias)
      out = conv(x)
      out = out.view(batch, channel, out.size(2))
      return out

x = Variable(torch.randn(2, 2, 5))
kernel = Variable(torch.randn(2, 2, 2))
bias = Variable(torch.randn(4))
model = Conv1dBias()
save_data_and_model_multy_inputs("conv1d_variable_wb", model, x, kernel, bias)

class GatherMultiOutput(nn.Module):
    def __init__(self, in_dim = 2):
        super(GatherMultiOutput, self).__init__()
        self.in_dim = in_dim
        self.lin_inp = nn.Linear(in_dim, 2, bias=False)
    def forward(self, x):
        x_projected = self.lin_inp(x).long()
        x_gather = x_projected[:,0,:]
        x_float1 = x_gather.float()
        x_float2 = x_gather.float()
        x_float3 = x_gather.float()
        return x_float1+x_float2+x_float3

x = Variable(torch.zeros([1, 2, 2]))
model = GatherMultiOutput()
save_data_and_model("gather_multi_output", x, model)

def postprocess_model(model_path, inputs_shapes):
    onnx_model = onnx.load(model_path)

    def update_inputs_dims(model, input_dims):
        """
            This function updates the sizes of dimensions of the model's inputs to the values
            provided in input_dims. if the dim value provided is negative, a unique dim_param
            will be set for that dimension.
        """
        def update_dim(tensor, dim, i, j, dim_param_prefix):
            dim_proto = tensor.type.tensor_type.shape.dim[j]
            if isinstance(dim, int):
                if dim >= 0:
                    dim_proto.dim_value = dim
                else:
                    dim_proto.dim_param = dim_param_prefix + str(i) + '_' + str(j)
            elif isinstance(dim, str):
                dim_proto.dim_param = dim
            else:
                raise ValueError('Only int or str is accepted as dimension value, incorrect type: {}'.format(type(dim)))

        for i, input_dim_arr in enumerate(input_dims):
            for j, dim in enumerate(input_dim_arr):
                update_dim(model.graph.input[i], dim, i, j, 'in_')

        onnx.checker.check_model(model)
        return model

    onnx_model = update_inputs_dims(onnx_model, inputs_shapes)
    onnx.save(onnx_model, model_path)

class UnsqueezeAndConv(nn.Module):
    def __init__(self):
        super(UnsqueezeAndConv, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = x.unsqueeze(axis=0)
        out = self.conv(x)
        return out

x = Variable(torch.randn(3, 10, 10))
model = UnsqueezeAndConv()
save_data_and_model("unsqueeze_and_conv_dynamic_axes", x, model)
postprocess_model("models/unsqueeze_and_conv_dynamic_axes.onnx", [[3, 'height', 'width']])

class SqueezeAndConv(nn.Module):
    def __init__(self):
        super(SqueezeAndConv, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = x.squeeze()
        out = self.conv(x)
        return out

x = Variable(torch.randn(2, 1, 3, 3, 3))
model = SqueezeAndConv()
save_data_and_model("squeeze_and_conv_dynamic_axes", x, model)
postprocess_model("models/squeeze_and_conv_dynamic_axes.onnx", [["batch_size", 1, "channels", 'height', 'width']])

x = Variable(torch.randn(2))
model = GatherScalar()
save_data_and_model("gather_scalar_dynamic_axes", x, model)
postprocess_model("models/gather_scalar_dynamic_axes.onnx", [['shape']])

x = Variable(torch.randn(2, 2, 2, 2))
print(x)
model = Gather()
print(model(x))
print(model(x).shape)
save_data_and_model("gather_dynamic_axes", x, model)
postprocess_model("models/gather_dynamic_axes.onnx", [["batch_size", 2, 'height', 'width']])

input = Variable(torch.randn(1, 2, 4, 4))
model = Slice()
save_data_and_model("slice_dynamic_axes", input, model)
save_data_and_model("slice_opset_11_dynamic_axes", input, model, version=11)
postprocess_model("models/slice_dynamic_axes.onnx", [["batch_size", 2, 'height', 'width']])
postprocess_model("models/slice_opset_11_dynamic_axes.onnx", [["batch_size", 2, 'height', 'width']])

x = Variable(torch.rand(1, 2, 2, 2))
model = ResizeConv(2, 0, 2)
save_data_and_model("resize_opset11_torch1.6_dynamic_axes", x, model, 11)
postprocess_model("models/resize_opset11_torch1.6_dynamic_axes.onnx", [["batch_size", 2, 'height', 'width']])

maxpooling_sigmoid = nn.Sequential(
          nn.MaxPool2d(kernel_size=4, stride=2, padding=(1, 2), dilation=1),
          nn.Sigmoid()
        )
input = Variable(torch.randn(2, 3, 12, 18))
save_data_and_model("maxpooling_sigmoid_dynamic_axes", input, maxpooling_sigmoid)
postprocess_model("models/maxpooling_sigmoid_dynamic_axes.onnx", [[2, 3, 'height', 'width']])

ave_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
input = Variable(torch.randn(1, 3, 7, 5))
save_data_and_model("average_pooling_dynamic_axes", input, ave_pool)
postprocess_model("models/average_pooling_dynamic_axes.onnx", [[1, 3, 'height', 'width']])

class DynamicBatch(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DynamicBatch, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        return torch.cat((self.pool(x), torch.ones(2, 3, 1, 2)))

model = DynamicBatch()
input_ = Variable(torch.ones(2, 3, 3, 4, dtype=torch.float32))
save_data_and_model("dynamic_batch", input_, model, export_params=True)
postprocess_model("models/dynamic_batch.onnx", [['batch_size', 3, 3, 4]])

x = Variable(torch.randn(1, 3, 10))
max_pool = nn.MaxPool1d(kernel_size=(5), stride=1, padding=2, dilation=1)
save_data_and_model("maxpooling_1d", x, max_pool)

x = Variable(torch.randn(2, 3, 12))
maxpooling_sigmoid = nn.Sequential(
          nn.MaxPool1d(kernel_size=4, stride=2, padding=(2), dilation=1),
          nn.Sigmoid()
        )
save_data_and_model("maxpooling_sigmoid_1d", x, maxpooling_sigmoid)

x = Variable(torch.randn(2, 3, 12))
maxpool2 = nn.Sequential(
           nn.MaxPool1d(kernel_size=5, stride=1, padding=0, dilation=1),
           nn.MaxPool1d(kernel_size=3, stride=1, padding=0, dilation=1)
           )
save_data_and_model("two_maxpooling_1d", x, maxpool2)

x = Variable(torch.randn(1, 3, 7))
ave_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
save_data_and_model("average_pooling_1d", x, ave_pool)

class PoolConv1d(nn.Module):

    def __init__(self):
        super(PoolConv1d, self).__init__()
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)
        self.conv = nn.Conv1d(2, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pool(x)
        y = self.conv(x)
        return y

x = Variable(torch.randn(1, 2, 4))
model = PoolConv1d()
save_data_and_model("pool_conv_1d", x, model)

class Conv1ResizePoold(nn.Module):
    def __init__(self):
        super(Conv1ResizePoold, self).__init__()
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)
        self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        batch_size = x.size(0)
        channels = x.size(1)
        x = self.conv(x)
        x = x.view(batch_size, channels, -1)
        y = self.pool(x)
        return y

x = Variable(torch.randn(1, 2, 20, 20))
model = Conv1ResizePoold()
save_data_and_model("conv_resize_pool_1d", x, model)

class Mish(nn.Module):
    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

x = Variable(torch.randn([1, 2, 2, 2]))
model = Mish()
save_data_and_model("mish", x, model)

class Mish2(nn.Module):
    def forward(self, x):
        return x * (torch.tanh(torch.log(torch.exp(x) + 1)))

x = Variable(torch.randn([1, 2, 2, 2]))
model = Mish2()
save_data_and_model("mish_no_softplus", x, model)

class PadCalculation(nn.Module):
    def forward(self, x):
        y = F.max_pool2d(x, kernel_size=2)
        diff_h = x.shape[2] - y.shape[2]
        diff_w = x.shape[3] - y.shape[3]
        y = F.pad(y, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        return y

x = Variable(torch.randn([1, 1, 3, 4]))
model = PadCalculation()
save_data_and_model("calc_pads", x, model, version=11)

class NormalizeFusion(nn.Module):
    def forward(self, x):
        mul = x * x
        sum = torch.sum(mul, dim=(1), keepdim=True)
        maximum = torch.clamp(sum, min=1e-8)
        sqrt = torch.sqrt(maximum)
        reciprocal = torch.reciprocal(sqrt)
        return x * reciprocal

x = Variable(torch.randn([2, 3]))
model = NormalizeFusion()
save_data_and_model("normalize_fusion", x, model)

class CumSum(nn.Module):
    def __init__(self, dim):
        super(CumSum, self).__init__()
        self._dim = dim

    def forward(self, x):
        return torch.cumsum(x, self._dim)

x = torch.randn(2, 3)
save_data_and_model("cumsum_2d_dim_1", x, CumSum(dim=1), version=11)

x = torch.randn(2, 3, 4)
save_data_and_model("cumsum_3d_dim_2", x, CumSum(dim=2), version=11)

# tf2onnx models
def save_data_and_tf_function(tf_function, name, input):
    input = input.astype(np.float32)
    np.save(os.path.join("data", "input_" + name + ".npy"), input)
    output = tf_function(input)
    np.save(os.path.join("data", "output_" + name + ".npy"), output)
    cumsum_model = tf2onnx.convert.from_function(
        function=tf_function,
        input_signature=[tf.TensorSpec([], tf.float32)],
        opset=14)[0]
    onnx.save(cumsum_model, os.path.join("models", name + ".onnx"))

x = np.random.rand(3)

@tf.function
def cumsum_exclusive_1d(x):
    return tf.cumsum(x, exclusive=True, reverse=False)

save_data_and_tf_function(cumsum_exclusive_1d, "cumsum_1d_exclusive_1", x)

@tf.function
def cumsum_reverse(x):
    return tf.cumsum(x, exclusive=False, reverse=True)

save_data_and_tf_function(cumsum_reverse, "cumsum_1d_reverse", x)

@tf.function
def cumsum_exclusive_1d_reverse(x):
    return tf.cumsum(x, exclusive=True, reverse=True)

save_data_and_tf_function(cumsum_exclusive_1d_reverse, "cumsum_1d_exclusive_1_reverse", x)

x = np.random.rand(1, 2, 3, 4)

@tf.function
def Not(x):
    return tf.cast(tf.math.logical_not(tf.math.less(x, 0.5)), tf.float32)

save_data_and_tf_function(Not, "not", x)

#paddle2onnx model
class Resize_HumanSeg(paddle.nn.Layer):
    def __init__(self, ):
        super(Resize_HumanSeg, self).__init__()

    def forward(self, x0):
        x1 = paddle.nn.functional.interpolate(x0,size=[6,8],mode='bilinear',align_corners=False)
        return x1

def save_data_and_paddle_model(model, name, input_data):
    model.eval()
    np.save(os.path.join("data", "input_" + name + ".npy"), input_data.numpy())
    output = model(input_data)
    np.save(os.path.join("data", "output_" + name + ".npy"), output.numpy())
    inputs = [paddle.static.InputSpec(shape=input_data.shape, dtype="float32")]
    paddle.onnx.export(model, "models/" + name,
                       input_spec=inputs,
                       opset_version=11)

input_shape = [1, 2, 3, 4]
x = paddle.rand(input_shape, dtype="float32")
save_data_and_paddle_model(Resize_HumanSeg(), "resize_humanseg", x)

class SubFromConstBroadcast(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SubFromConstBroadcast, self).__init__()
        self.const = torch.randn(1, 3, dtype=torch.float32)

    def forward(self, x):
        return self.const - x

model = SubFromConstBroadcast()
input_ = Variable(torch.randn(2, 3, dtype=torch.float32))
save_data_and_model("sub_from_const_broadcast", input_, model)

class SubFromConstEltWise(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SubFromConstEltWise, self).__init__()
        self.const = torch.randn(1, 2, 3, 4, dtype=torch.float32)

    def forward(self, x):
        return self.const - x

model = SubFromConstEltWise()
input_ = Variable(torch.randn(1, 2, 3, 4, dtype=torch.float32))
save_data_and_model("sub_from_const_eltwise", input_, model)

class SubFromConst1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SubFromConst1, self).__init__()

    def forward(self, x):
        return 1 - x

model = SubFromConst1()
input_ = Variable(torch.randn(1, 2, 3, 4, dtype=torch.float32))
save_data_and_model("sub_from_const1", input_, model)

class ArgMax(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ArgMax, self).__init__()

    def forward(self, x):
        return torch.argmax(x, dim=2, keepdims=False).to(torch.float32)

model = ArgMax()
input_ = Variable(torch.randn(2, 3, 4, 5, dtype=torch.float32))
save_data_and_model("argmax", input_, model)

class ArgMin(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ArgMin, self).__init__()

    def forward(self, x):
        return torch.argmin(x, dim=-1, keepdims=True).to(torch.float32)

model = ArgMin()
input_ = Variable(torch.randn(2, 3, 4, 5, dtype=torch.float32))
save_data_and_model("argmin", input_, model)

########################## const / x ##########################

node = onnx.helper.make_node('Div', inputs=['x', 'y'], outputs=['z'])

x = np.array([2]).astype(np.float32)
y = np.array([[4, 4], [4, 4]]).astype(np.float32)
name = 'div_const'
input_files = os.path.join("data", "input_" + name)
np.save(input_files, x.data)
np.save(input_files, y.data)

z = (x / y).astype(np.float32)
output_files =  os.path.join("data", "output_" + name)
np.save(output_files, np.ascontiguousarray(z.data))

X = onnx.helper.make_tensor('x', onnx.TensorProto.FLOAT, x.shape, x)
Y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, y.shape)
Z = onnx.helper.make_tensor_value_info('z', onnx.TensorProto.FLOAT, z.shape)

graph = onnx.helper.make_graph([node], 'div_const', [Y], [Z], initializer=[X])
model = onnx.helper.make_model(graph, producer_name=name)
models_files = os.path.join("models", name + ".onnx")
onnx.save(model, models_files)

########################## const / x ##########################

class OutputRegistration(nn.Module):
    def __init__(self):
        super(OutputRegistration, self).__init__()
        self.c = torch.randn(2, 2)

    def forward(self, a, b):
        return (a + b) + self.c

a = Variable(torch.randn(2, 2))
b = Variable(torch.randn(2, 2))
model = OutputRegistration()
save_data_and_model_multy_inputs('output_registration', model, a, b)
model = onnx.load('models/output_registration.onnx')
model.graph.node[0].name = model.graph.output[0].name
onnx.save(model, 'models/output_registration.onnx')

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
inputs = [onnx.helper.make_tensor_value_info("input1", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_np.dtype], shape=input_np.shape)]

weight_np = np.random.rand(10, 3).astype("float32")
weight_tensor = onnx.helper.make_tensor('weight_tensor', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight_np.dtype], dims=weight_np.shape, vals=weight_np)

outputs = [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape=(2, 3))]

nodes = [onnx.helper.make_node("Gemm", ["input1", "weight_tensor"], ["output"])]

graph = onnx.helper.make_graph(nodes,
                            "gemm_test",
                            inputs,
                            outputs, initializer=[weight_tensor])
gemm_model = onnx.helper.make_model(graph)
output_np = gemm_reference_implementation(input_np, weight_np)
save_data_and_onnx_model("gemm_no_transB", input_np, output_np, gemm_model)

## gemm with transB = 0

nodes2 = [onnx.helper.make_node("Gemm", ["input1", "weight_tensor"], ["output"], transB=0)]
graph2 = onnx.helper.make_graph(nodes2,
                            "gemm_test",
                            inputs,
                            outputs, initializer=[weight_tensor])
gemm_model2 = onnx.helper.make_model(graph2)
output_np = gemm_reference_implementation(input_np, weight_np)
save_data_and_onnx_model("gemm_transB_0", input_np, output_np, gemm_model2)

## gemm with transA = 1, transB = 1 and the first input is constance.

weight_np = np.random.rand(10, 2).astype("float32")
weight_tensor = helper.make_tensor('weight_tensor', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight_np.dtype], dims=weight_np.shape, vals=weight_np)

input_np = np.random.rand(3, 10).astype("float32")
inputs = [helper.make_tensor_value_info("input1", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_np.dtype], shape=input_np.shape)]

outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape=(2, 3))]

nodes = [helper.make_node("Gemm", ["weight_tensor", "input1"], ["output"], transA=1, transB=1)]

graph = helper.make_graph(nodes,
                            "gemm_test",
                            inputs,
                            outputs, initializer=[weight_tensor])
gemm_model = helper.make_model(graph)
output_np = gemm_reference_implementation(weight_np.T, input_np.T)
save_data_and_model("gemm_first_const", input_np, output_np, gemm_model)

## gemm with bias
def generate_gemm_bias(name, inputA, inputB, inputC):
    outputY = gemm_reference_implementation(inputA, inputB, inputC)

    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, inputA.shape)
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, inputB.shape)
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, inputC.shape)
    B_INIT = onnx.helper.make_tensor("B", onnx.TensorProto.FLOAT, inputB.shape, inputB)
    C_INIT = onnx.helper.make_tensor("C", onnx.TensorProto.FLOAT, inputC.shape, inputC)
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, outputY.shape)
    node = onnx.helper.make_node("Gemm", inputs=["A", "B", "C"], outputs=["Y"])
    graph = onnx.helper.make_graph([node], name, [A, B, C], [Y], [B_INIT, C_INIT])
    model = onnx.helper.make_model(graph, producer_name=name)
    onnx.save(model, os.path.join("models", name + ".onnx"))

    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, inputA.data)
    output_files = os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(outputY.data))

inputA = np.random.ranf([3, 6]).astype(np.float32)
inputB = np.random.ranf([6, 4]).astype(np.float32)
inputC = np.random.ranf([1, 4]).astype(np.float32)
generate_gemm_bias("gemm_vector_bias", inputA, inputB, inputC)

# ########################## ReduceSum with Dynamic Batch ##########################
input_np = np.random.rand(2, 4, 4, 4).astype("float32")
inputs = [onnx.helper.make_tensor_value_info("input1", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_np.dtype], shape=('?', 4, 4, 4))]

axis_np = np.array([1]).astype(np.int64)
axis_tensor = onnx.helper.make_tensor('axis_tensor', data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[axis_np.dtype], dims=axis_np.shape, vals=axis_np)

outputs = [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape=(2, 1, 4, 4))]

nodes = [onnx.helper.make_node("ReduceSum", ["input1", "axis_tensor"], ["output"], keepdims=1)]

graph = onnx.helper.make_graph(nodes,
                            "reduce_sum",
                            inputs,
                            outputs, initializer=[axis_tensor])
onnx_model = onnx.helper.make_model(graph)

output_np = np.sum(input_np, axis=1, keepdims=1)
save_data_and_onnx_model("reduce_sum_axis_dynamic_batch", input_np, output_np, onnx_model)


# ########################## DivBroadcast ##########################
input_np = np.random.rand(1, 4).astype("float32")
input2_np = np.random.rand(1, 1).astype(np.float32)
inputs = [onnx.helper.make_tensor_value_info("input1", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_np.dtype], shape=input_np.shape), \
    onnx.helper.make_tensor_value_info("input2", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input2_np.dtype], shape=input2_np.shape)]

outputs = [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape=(1, 4))]

nodes = [onnx.helper.make_node("Div", ["input1", "input2"], ["output"])]

graph = onnx.helper.make_graph(nodes,
                            "div_test",
                            inputs,
                            outputs)
onnx_model = onnx.helper.make_model(graph)

output_np = input_np/input2_np
save_data_and_onnx_model_multy_inputs("div_test_1x1", [input_np, input2_np], output_np, onnx_model)


###################### GatherMulti #################################
N, C, H, W = 2, 3, 8, 7
axis = 2 # H

input = np.random.rand(N, C, H, W).astype(np.float32)
idx = np.random.randint(low=1, high=H, size=(3, 4), dtype=np.int32)
output = np.take(input, idx, axis=axis)

inputs = [onnx.helper.make_tensor_value_info("x", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input.dtype], shape=input.shape)]
outputs = [onnx.helper.make_tensor_value_info("y", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[output.dtype], shape=output.shape)]

nodes = [onnx.helper.make_node("Constant", [], ["idx"], value=onnx.numpy_helper.from_array(idx)), onnx.helper.make_node("Gather", ["x", "idx"], ["y"], axis=axis)]
graph = onnx.helper.make_graph(nodes, "gather_multi", inputs, outputs)
onnx_model = onnx.helper.make_model(graph)

save_data_and_onnx_model("gather_multi", input, output, onnx_model)

###################### Tile #################################

# input & output are taken from onnx conformance test 'test_tile'

def generate_onnx_single_operator(single_op, onnx_name, save_prefix="./models"):
    # Create inputs (ValueInfoProto)
    inputs = []
    input_names = []
    initializers = []
    for name, prop in single_op.get("inputs").items():
        input_names.append(name)

        dtype = prop.get("dtype")
        shape = prop.get("shape")
        inputs.append(
            helper.make_tensor_value_info(name, dtype, shape)
        )
        initializer = prop.get("initializer")
        if initializer is not None:
            initializers.append(
                helper.make_tensor(name, dtype, shape, initializer)
            )

    # Create outputs (ValueInfoProto)
    outputs = []
    output_names = []
    for name, prop in single_op.get("outputs").items():
        output_names.append(name)

        dtype = prop.get("dtype")
        shape = prop.get("shape")
        outputs.append(
            helper.make_tensor_value_info(name, dtype, shape)
        )

    # Create a node (NodeProto)
    attributes = single_op.get("attributes", {})
    node_def = onnx.helper.make_node(
        single_op.get("op_name"),
        inputs=input_names,
        outputs=output_names,
        **attributes,
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],        # nodes
        onnx_name,         # name
        inputs,            # inputs
        outputs,           # outputs
        initializers       # initializer
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name="github.com/opencv/opencv_extra")
    onnx.checker.check_model(model_def)
    onnx.save(model_def, "models/{}.onnx".format(onnx_name))

    return True

tile=dict(
    op_name="Tile",
    inputs=dict(
        input=dict(
            dtype=TensorProto.FLOAT,
            shape=[2, 3, 4, 5],
        ),
        repeats=dict(
            dtype=TensorProto.INT64,
            shape=[4],
            initializer=np.array([7, 6, 4, 2], dtype=np.int64)
        ),
    ),
    outputs=dict(
        y=dict(
            dtype=TensorProto.FLOAT,
            shape=[14, 18, 16, 10]
        )
    )
)

generate_onnx_single_operator(tile, "tile")

def gen_layer_norm_expanded(input_shape=[1, 4, 5], axis=-1, constant_as_initializers=False):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, input_shape)
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, input_shape)
    nodes = []
    initializers = []

    class NodeNameManager:
        def __init__(self):
            self.name_dict = dict()

        def get_name(self, op_type):
            if op_type in self.name_dict:
                self.name_dict[op_type] += 1
            else:
                self.name_dict[op_type] = 0

            return "{}.{}".format(op_type, self.name_dict[op_type])

    node_name_manager = NodeNameManager()

    def make_node(op_type, inputs=None, outputs=None, *args, **kwargs):
        nonlocal node_name_manager, nodes
        node_name = node_name_manager.get_name(op_type)

        if inputs is None:
            inputs = [nodes[-1].output[0]]
        if outputs is None:
            outputs = ["{}.out".format(node_name)]
        return [onnx.helper.make_node(op_type, inputs, outputs, *args, **kwargs)]

    def make_node_with_constant(op_type, constant_value, inputs=None, outputs=None, is_constant_scalar=False):
        nonlocal node_name_manager, nodes, initializers, constant_as_initializers
        node_name = node_name_manager.get_name(op_type)

        constant_shape = [] if is_constant_scalar else constant_value.shape
        tensor = onnx.helper.make_tensor(
            "Const.{}.tensor".format(node_name),
            onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[constant_value.dtype],
            constant_shape,
            vals=constant_value
        )
        if inputs is None:
            inputs = [nodes[-1].output[0]]
        if outputs is None:
            outputs = ["{}.out".format(node_name)]

        if constant_as_initializers:
            inputs = inputs + [tensor.name]
            initializers += [tensor]
            return [onnx.helper.make_node(op_type, inputs, outputs, node_name)]
        else:
            node_const = onnx.helper.make_node("Constant", [], ["Const.{}.out".format(node_name)], value=tensor)
            inputs = inputs + [node_const.output[0]]
            return [node_const, onnx.helper.make_node(op_type, inputs, outputs, node_name)]

    #   -> ReduceMean ->     -> Pow(2) -> ReduceMean -> Add(epsilon) -> Sqrt ->
    # x                  Sub                                                    Div -> Mul(weight) -> Add(bias)
    #   --------------->     ------------------------------------------------->

    nodes += make_node("ReduceMean", inputs=["X"], axes=np.array([axis], dtype=np.int64))
    nodes += make_node("Sub", inputs=["X", nodes[-1].output[0]])
    node_sub_0_outname = nodes[-1].output[0]
    nodes += make_node_with_constant("Pow", np.array([2], dtype=np.float32), is_constant_scalar=True)
    nodes += make_node("ReduceMean", axes=np.array([axis], dtype=np.int64))
    nodes += make_node_with_constant("Add", np.array([1e-5], dtype=np.float32), is_constant_scalar=True)
    nodes += make_node("Sqrt")
    nodes += make_node("Div", inputs=[node_sub_0_outname, nodes[-1].output[0]])
    nodes += make_node_with_constant("Mul", np.random.rand(*input_shape[axis:]).astype(np.float32))
    nodes += make_node_with_constant("Add", np.random.rand(*input_shape[axis:]).astype(np.float32), outputs=["Y"])

    graph_name = "layer_norm_expanded"
    if constant_as_initializers:
        graph_name += "with_initializers"
    graph_def = onnx.helper.make_graph(
        nodes,
        graph_name,
        [X],
        [Y],
        initializers
    )
    model_def = onnx.helper.make_model(graph_def, producer_name="github.com/opencv/opencv_extra")
    onnx.checker.check_model(model_def)
    shape_inferred_model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.save(shape_inferred_model_def, "models/{}.onnx".format(graph_name))

    # infer & save data
    input_blob = np.random.rand(*input_shape).astype(np.float32)

    import onnxruntime as ort
    sess = ort.InferenceSession("models/{}.onnx".format(graph_name))
    output_blobs = sess.run(["Y"], {"X": input_blob})
    np.save("data/input_{}.npy".format(graph_name), input_blob)
    np.save("data/output_{}.npy".format(graph_name), output_blobs[0])

gen_layer_norm_expanded()
gen_layer_norm_expanded(constant_as_initializers=True)

################# GELU #################

x = torch.randn(1, 5, 20)
gelu = nn.GELU()
save_data_and_model("gelu", x, gelu)

gelu_approximation = nn.GELU('tanh')
save_data_and_model("gelu_approximation", x, gelu_approximation)

# Test data for a part of model: https://huggingface.co/openai/clip-vit-base-patch32
input = np.random.standard_normal((1, 1, 3)).astype(np.float32)
embedding = np.array([4, 5, 6], dtype=np.float32)
data = np.random.standard_normal((2, 3)).astype(np.float32)
indices = np.array([[0, 1]], dtype=np.int64)

output = np.concatenate((embedding.reshape(1, 1, 3), input), axis=1) + np.take(data, indices, axis=0)

embedding = onnx.numpy_helper.from_array(embedding, name='embedding')
X = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, input.shape)
Y = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, output.shape)

shape = np.array([1, 1, -1], dtype=np.int32)
shape = onnx.numpy_helper.from_array(shape, name='shape')
expand = onnx.helper.make_node("Expand", inputs=['embedding', 'shape'], outputs=['expand'])

one = np.array([1, 1, 1], dtype=np.float32)
one = onnx.numpy_helper.from_array(one, name='one')
mul = onnx.helper.make_node("Mul", inputs=['input', 'one'], outputs=['input_mul'])

concat = onnx.helper.make_node("Concat", inputs=['expand', 'input_mul'], outputs=['concat'], axis=1)

data = onnx.numpy_helper.from_array(data, name='data')
indices = onnx.numpy_helper.from_array(indices, name='indices')

gather = onnx.helper.make_node("Gather", inputs=['data', 'indices'], outputs=['gather'])
add = onnx.helper.make_node("Add", inputs=['concat', 'gather'], outputs=['output'])

name = "clip-vit-base-head"
graph = onnx.helper.make_graph([mul, expand, concat, gather, add], name, [X], [Y], [embedding, data, indices, shape, one])

model = onnx.helper.make_model(graph, producer_name=name)
onnx.save(model, os.path.join("models", name + ".onnx"))

input_files = os.path.join("data", "input_" + name)
np.save(input_files, input.data)
output_files = os.path.join("data", "output_" + name)
np.save(output_files, np.ascontiguousarray(output.data))
