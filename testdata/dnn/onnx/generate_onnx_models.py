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


def export_to_string(model, inputs):
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, inputs, f)
    return f.getvalue()


def save_data_and_model(name, input, model):
    model.eval()
    print name + " input has sizes",  input.shape
    input_files = os.path.join("data", "input_" + name)
    np.save(input_files, input.data)

    output = model(input)

    print name + " output has sizes", output.shape
    print
    output_files =  os.path.join("data", "output_" + name)
    np.save(output_files, np.ascontiguousarray(output.data))

    models_files = os.path.join("models", name + ".onnx")

    onnx_model_pb = export_to_string(model, input)
    model_def = assertONNXExpected(onnx_model_pb)
    with open(models_files, 'wb') as file:
        file.write(model_def.SerializeToString())

torch.manual_seed(0)

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
    print name + " output has sizes", output.shape
    print
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
