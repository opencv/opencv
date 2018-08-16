// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "test_precomp.hpp"
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace opencv_test { namespace {

template<typename TString>
static std::string _tf(TString filename)
{
    String rootFolder = "dnn/onnx/";
    return findDataFile(rootFolder + filename, false);
}

class Test_ONNX_layers : public DNNTestLayer
{
public:
    enum Extension
    {
        npy,
        pb
    };

    void testONNXModels(const String& basename, const Extension ext = npy,
                                const double l1 = 1e-5, const float lInf = 1e-4)
    {
        String onnxmodel = _tf("models/" + basename + ".onnx");
        Mat inp, ref;
        if (ext == npy) {
            inp = blobFromNPY(_tf("data/input_" + basename + ".npy"));
            ref = blobFromNPY(_tf("data/output_" + basename + ".npy"));
        }
        else if (ext == pb) {
            inp = readTensorFromONNX(_tf("data/input_" + basename + ".pb"));
            ref = readTensorFromONNX(_tf("data/output_" + basename + ".pb"));
        }
        else
            CV_Error(Error::StsUnsupportedFormat, "Unsupported extension");

        checkBackend(&inp, &ref);
        Net net = readNetFromONNX(onnxmodel);
        ASSERT_FALSE(net.empty());

        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);

        net.setInput(inp);
        Mat out = net.forward();
        normAssert(ref, out, "", l1, lInf);
    }
};

TEST_P(Test_ONNX_layers, MaxPooling)
{
    testONNXModels("maxpooling");
    testONNXModels("two_maxpooling");
}

TEST_P(Test_ONNX_layers, Convolution)
{
    testONNXModels("convolution");
    testONNXModels("two_convolution");
}

TEST_P(Test_ONNX_layers, Dropout)
{
    testONNXModels("dropout");
}

TEST_P(Test_ONNX_layers, Linear)
{
    if (backend == DNN_BACKEND_OPENCV && target == DNN_TARGET_OPENCL_FP16)
        throw SkipTestException("");
    testONNXModels("linear");
}

TEST_P(Test_ONNX_layers, ReLU)
{
    testONNXModels("ReLU");
}

TEST_P(Test_ONNX_layers, MaxPooling_Sigmoid)
{
    testONNXModels("maxpooling_sigmoid");
}

TEST_P(Test_ONNX_layers, Concatenation)
{
    if (backend == DNN_BACKEND_INFERENCE_ENGINE &&
         (target == DNN_TARGET_OPENCL_FP16 || target == DNN_TARGET_OPENCL))
        throw SkipTestException("");
    testONNXModels("concatenation");
}

TEST_P(Test_ONNX_layers, AveragePooling)
{
    testONNXModels("average_pooling");
}

TEST_P(Test_ONNX_layers, BatchNormalization)
{
    testONNXModels("batch_norm");
}

TEST_P(Test_ONNX_layers, Multiplication)
{
    testONNXModels("mul");
}

TEST_P(Test_ONNX_layers, MultyInputs)
{
    const String model =  _tf("models/multy_inputs.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp1 = blobFromNPY(_tf("data/input_multy_inputs_0.npy"));
    Mat inp2 = blobFromNPY(_tf("data/input_multy_inputs_1.npy"));
    Mat ref  = blobFromNPY(_tf("data/output_multy_inputs.npy"));
    checkBackend(&inp1, &ref);

    net.setInput(inp1, "0");
    net.setInput(inp2, "1");
    Mat out = net.forward();

    normAssert(ref, out, "", default_l1,  default_lInf);
}


INSTANTIATE_TEST_CASE_P(/*nothing*/, Test_ONNX_layers, dnnBackendsAndTargets());

class Test_ONNX_nets : public Test_ONNX_layers {};
TEST_P(Test_ONNX_nets, Alexnet)
{
    const String model =  _tf("models/bvlc_alexnet.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat inp = imread(_tf("../grace_hopper_227.png"));
    Mat ref = blobFromNPY(_tf("../caffe_alexnet_prob.npy"));
    checkBackend(&inp, &ref);

    net.setInput(blobFromImage(inp, 1.0f, Size(227, 227), Scalar(), false));
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(out, ref, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, Squeezenet)
{
    const String model =  _tf("models/squeezenet.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat image = imread(_tf("../googlenet_0.png"));
    Mat inp = blobFromImage(image, 1.0f, Size(227,227), Scalar(), false);
    Mat ref = blobFromNPY(_tf("../squeezenet_v1.1_prob.npy"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();
    out = out.reshape(1, 1);

    normAssert(ref, out, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, Googlenet)
{
    const String model = _tf("models/googlenet.onnx");

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    std::vector<Mat> images;
    images.push_back( imread(_tf("../googlenet_0.png")) );
    images.push_back( imread(_tf("../googlenet_1.png")) );
    Mat inp = blobFromImages(images, 1.0f, Size(), Scalar(), false);
    Mat ref = blobFromNPY(_tf("../googlenet_prob.npy"));
    checkBackend(&inp, &ref);

    net.setInput(inp);
    ASSERT_FALSE(net.empty());
    Mat out = net.forward();

    normAssert(ref, out, "", default_l1,  default_lInf);
}

TEST_P(Test_ONNX_nets, CaffeNet)
{
    testONNXModels("caffenet", pb);
}

TEST_P(Test_ONNX_nets, RCNN_ILSVRC13)
{
    testONNXModels("rcnn_ilsvrc13", pb);
}

TEST_P(Test_ONNX_nets, VGG16)
{
    testONNXModels("vgg16", pb);
}

TEST_P(Test_ONNX_nets, VGG16_bn)
{
    testONNXModels("vgg16-bn", pb);
}

TEST_P(Test_ONNX_nets, ZFNet)
{
    testONNXModels("zfnet512", pb);
}

TEST_P(Test_ONNX_nets, ResNet18v1)
{
    testONNXModels("resnet18v1", pb);
}

TEST_P(Test_ONNX_nets, ResNet50v1)
{
    testONNXModels("resnet50v1", pb);
}

TEST_P(Test_ONNX_nets, ResNet101_DUC_HDC)
{
    testONNXModels("resnet101_duc_hdc", pb);
}

TEST_P(Test_ONNX_nets, TinyYolov2)
{
    testONNXModels("tiny_yolo2", pb);
}

TEST_P(Test_ONNX_nets, CNN_MNIST)
{
    testONNXModels("cnn_mnist", pb, 4.2e-4, 1e-3);
}

TEST_P(Test_ONNX_nets, MobileNet_v2)
{
    testONNXModels("mobilenetv2", pb, 5e-5, 5e-4);
}

TEST_P(Test_ONNX_nets, LResNet100E_IR)
{
    testONNXModels("LResNet100E_IR", pb);
}

INSTANTIATE_TEST_CASE_P(/**/, Test_ONNX_nets, dnnBackendsAndTargets());

}} // namespace
